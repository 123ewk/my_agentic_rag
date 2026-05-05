"""
长期记忆V2管理器(PostgreSQL + pgvector)

相比V1的核心改进:
1. 四类型分类: user_profile / fact / experience / preference
2. 写入前价值评估: 低于阈值直接丢弃
3. 写入前去重: 高相似度合并更新而非新增
4. 分层检索: 类型预筛→向量搜索→综合排序→数量限制
5. 时间衰减: 按类型差异化半衰期
6. 访问统计: 记录检索命中次数，影响衰减和排序
7. 容量控制: 低价值记忆定期淘汰
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, delete, func, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.embeddings import Embeddings

from ..models.long_term_model_v2 import long_term_memories_v2
from .memory_evaluator import (
    MemoryType,
    MemoryCandidate,
    MEMORY_VALUE_THRESHOLD,
    calculate_decay_score,
    INTENT_TO_MEMORY_TYPES,
)
from .memory_compressor import extract_memory_candidate
from loguru import logger


# 单次检索最多返回的长期记忆数量
MAX_LONG_TERM_MEMORIES = 3


class LongTermMemoryV2:
    """
    长期记忆V2管理器

    核心改进: 价值筛选写入 + 四类型分类 + 分层检索 + 时间衰减
    """

    def __init__(
        self,
        embeddings: Embeddings,
        llm=None,
        database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/agentic_rag",
        k: int = 5,
        similarity_threshold: float = 0.7,
        pool_size: int = 10,
        max_overflow: int = 20,
        max_memories_per_user: int = 500,
    ):
        """
        初始化长期记忆V2

        Args:
            embeddings: 嵌入模型实例
            llm: 大语言模型实例(用于记忆提取和压缩)
            database_url: PostgreSQL连接字符串
            k: 检索相似记忆数量
            similarity_threshold: 相似度阈值(0-1)
            pool_size: 连接池基础大小
            max_overflow: 连接池最大溢出
            max_memories_per_user: 每用户最大记忆数量
        """
        self.database_url = database_url
        self.embeddings = embeddings
        self.llm = llm
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.max_memories_per_user = max_memories_per_user

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((SQLAlchemyError, ConnectionError)),
        reraise=True,
    )
    async def connect(self):
        """带自动重试的数据库连接"""
        try:
            self.engine = create_async_engine(
                self.database_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=30,
                pool_recycle=1800,
                echo=False,
            )

            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            async with self.engine.begin():
                logger.info("✅ 长期记忆V2数据库连接成功")

        except SQLAlchemyError as e:
            logger.error("❌ 长期记忆V2数据库连接失败，即将重试：{}", str(e))
            raise
        except Exception as e:
            logger.error("❌ 未知错误：{}", str(e))
            raise

    # ====================== 写入流程 ======================

    async def save_from_conversation(
        self,
        user_id: str,
        question: str,
        answer: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        从对话中提取并保存高价值记忆

        完整写入流程:
        1. LLM提取候选记忆
        2. 硬性过滤 + 价值评估
        3. 类型分类
        4. 去重检查(合并/跳过/新增)
        5. 生成embedding + 写入

        Args:
            user_id: 用户标识
            question: 用户问题
            answer: AI回答
            session_id: 会话标识
            context: 额外上下文

        Returns:
            记忆ID，未写入返回None
        """
        if not self.llm:
            logger.warning("LLM未配置，无法提取记忆，跳过")
            return None

        if not answer or len(answer.strip()) < 10:
            return None

        # Step 1: LLM提取候选记忆
        candidate = await extract_memory_candidate(
            llm=self.llm,
            question=question,
            answer=answer,
            context=context,
        )
        if candidate is None:
            logger.debug("对话未产生值得记忆的信息")
            return None

        # Step 2: 去重检查
        action = await self._deduplicate_before_write(user_id, candidate)
        if action == "skip":
            logger.debug(f"重复记忆，跳过: {candidate.summary[:50]}")
            return None
        elif action == "merge":
            logger.debug(f"相似记忆，合并更新: {candidate.summary[:50]}")
            return await self._merge_and_update(user_id, candidate, session_id)

        # Step 3: 写入新记忆
        return await self._insert_new_memory(user_id, candidate, session_id)

    async def _deduplicate_before_write(
        self, user_id: str, candidate: MemoryCandidate
    ) -> str:
        """
        写入前去重检查

        在同类型记忆中搜索相似内容:
        - sim > 0.95 → 跳过(几乎完全重复)
        - sim > 0.85 → 合并更新
        - 否则 → 新增

        Args:
            user_id: 用户标识
            candidate: 候选记忆

        Returns:
            "skip" / "merge" / "insert"
        """
        try:
            similar = await self.search(
                user_id=user_id,
                query=candidate.summary,
                k=5,
                metadata_filter={"memory_type": candidate.memory_type.value},
            )

            if not similar:
                return "insert"

            max_similarity = max(m["similarity"] for m in similar)

            if max_similarity > 0.95:
                return "skip"
            elif max_similarity > 0.85:
                return "merge"

            return "insert"
        except Exception as e:
            logger.warning(f"去重检查失败，默认新增: {e}")
            return "insert"

    async def _merge_and_update(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        合并记忆: 更新而非新增

        合并策略按类型不同:
        - user_profile: 字段级合并，提升confidence
        - fact: 新事实替代旧事实(标记supersedes_id)
        - experience: 累加occurrence_count
        - preference: 矛盾时累计contradict_count

        Args:
            user_id: 用户标识
            candidate: 候选记忆
            session_id: 会话标识

        Returns:
            更新后的记忆ID
        """
        try:
            similar = await self.search(
                user_id=user_id,
                query=candidate.summary,
                k=1,
                metadata_filter={"memory_type": candidate.memory_type.value},
            )

            if not similar:
                return await self._insert_new_memory(user_id, candidate, session_id)

            target = similar[0]
            target_id = target["id"]
            existing_meta = target.get("metadata", {})

            if candidate.memory_type == MemoryType.USER_PROFILE:
                existing_meta["confidence"] = min(
                    1.0, existing_meta.get("confidence", 0.5) + 0.1
                )
            elif candidate.memory_type == MemoryType.EXPERIENCE:
                existing_meta["occurrence_count"] = (
                    existing_meta.get("occurrence_count", 1) + 1
                )
            elif candidate.memory_type == MemoryType.PREFERENCE:
                if _is_contradictory(target["content"], candidate.content):
                    existing_meta["contradict_count"] = (
                        existing_meta.get("contradict_count", 0) + 1
                    )

            # 合并内容: 保留更丰富的描述
            merged_content = _merge_content(target["content"], candidate.content)
            merged_summary = candidate.summary or target.get("summary", "")

            # 重新生成embedding
            new_embedding = self.embeddings.embed_query(merged_content)

            # 更新数据库
            async with self.async_session() as session:
                stmt = (
                    update(long_term_memories_v2)
                    .where(long_term_memories_v2.c.id == target_id)
                    .values(
                        content=merged_content,
                        summary=merged_summary,
                        embedding=new_embedding,
                        value_score=max(
                            target.get("value_score", 0.5), candidate.value_score
                        ),
                        metadata=existing_meta,
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                await session.execute(stmt)
                await session.commit()

                logger.info(f"合并记忆: {merged_summary[:50]}")
                return str(target_id)

        except Exception as e:
            logger.error(f"合并记忆失败: {e}")
            return None

    async def _insert_new_memory(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        写入新记忆

        Args:
            user_id: 用户标识
            candidate: 候选记忆
            session_id: 会话标识

        Returns:
            新记忆ID
        """
        try:
            embedding = self.embeddings.embed_query(candidate.content)

            source_session_ids = [session_id] if session_id else []

            async with self.async_session() as session:
                stmt = (
                    insert(long_term_memories_v2)
                    .values(
                        user_id=user_id,
                        memory_type=candidate.memory_type.value,
                        content=candidate.content,
                        summary=candidate.summary,
                        embedding=embedding,
                        value_score=candidate.value_score,
                        metadata=candidate.metadata,
                        source_session_ids=source_session_ids,
                    )
                    .returning(long_term_memories_v2.c.id)
                )

                result = await session.execute(stmt)
                await session.commit()

                memory_id = str(result.scalar())
                logger.info(
                    f"写入新记忆[{candidate.memory_type.value}](score={candidate.value_score:.2f}): {candidate.summary[:50]}"
                )
                return memory_id

        except Exception as e:
            logger.error(f"写入新记忆失败: {e}")
            raise

    # ====================== 检索流程 ======================

    async def search(
        self,
        user_id: str,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        memory_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索相似记忆(向量检索)

        Args:
            user_id: 用户标识
            query: 查询文本
            k: 检索数量
            metadata_filter: 元数据过滤条件
            memory_types: 限定检索的记忆类型列表

        Returns:
            相似记忆列表(含相似度分数)
        """
        try:
            actual_k = k or self.k
            query_embedding = self.embeddings.embed_query(query)

            async with self.async_session() as session:
                # 构建类型过滤条件
                type_clause = ""
                if memory_types:
                    type_values = ", ".join([f"'{t}'" for t in memory_types])
                    type_clause = f"AND memory_type IN ({type_values})"

                # 构建元数据过滤条件
                metadata_clause = ""
                params = {
                    "user_id": user_id,
                    "query_embedding": str(query_embedding),
                    "threshold": self.similarity_threshold,
                    "k": actual_k,
                }

                if metadata_filter:
                    filter_conditions = []
                    for key, value in metadata_filter.items():
                        filter_conditions.append(f"metadata->>'{key}' = :filter_{key}")
                        params[f"filter_{key}"] = str(value)
                    metadata_clause = "AND " + " AND ".join(filter_conditions)

                sql = text(f"""
                    SELECT
                        id, user_id, memory_type, content, summary,
                        value_score, metadata, access_count,
                        last_accessed_at, created_at, updated_at,
                        1 - (embedding <=> :query_embedding) as similarity
                    FROM long_term_memories_v2
                    WHERE user_id = :user_id
                        AND 1 - (embedding <=> :query_embedding) >= :threshold
                        {type_clause}
                        {metadata_clause}
                    ORDER BY embedding <=> :query_embedding
                    LIMIT :k
                """)

                result = await session.execute(sql, params)
                rows = result.fetchall()

                return [
                    {
                        "id": str(row.id),
                        "memory_type": row.memory_type,
                        "content": row.content,
                        "summary": row.summary,
                        "value_score": float(row.value_score),
                        "metadata": row.metadata,
                        "access_count": row.access_count,
                        "last_accessed_at": row.last_accessed_at,
                        "created_at": row.created_at,
                        "similarity": float(row.similarity),
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"❌ 搜索长期记忆V2失败: {e}")
            raise

    async def search_with_decay(
        self,
        user_id: str,
        query: str,
        k: Optional[int] = None,
        intent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        分层检索: 类型预筛→向量搜索→综合排序→数量限制

        综合排序公式: final = 0.5*sim + 0.25*value + 0.25*decay

        Args:
            user_id: 用户标识
            query: 查询文本
            k: 检索数量
            intent: 查询意图(用于类型预筛)

        Returns:
            排序后的记忆列表(最多MAX_LONG_TERM_MEMORIES条)
        """
        try:
            # 第一层: 类型预筛选
            memory_types = None
            if intent and intent in INTENT_TO_MEMORY_TYPES:
                memory_types = INTENT_TO_MEMORY_TYPES[intent]

            # 第二层: 向量相似度检索(多取一些用于后续排序)
            candidates = await self.search(
                user_id=user_id,
                query=query,
                k=min(10, (k or self.k) * 2),
                memory_types=memory_types,
            )

            if not candidates:
                return []

            # 第三层: 综合排序(sim + value + decay)
            now = datetime.now(timezone.utc)
            for mem in candidates:
                decay = calculate_decay_score(
                    memory_type=mem["memory_type"],
                    created_at=mem["created_at"],
                    access_count=mem.get("access_count", 0),
                    current_time=now,
                )
                mem["decay_score"] = decay
                mem["final_score"] = (
                    0.5 * mem["similarity"]
                    + 0.25 * mem.get("value_score", 0.5)
                    + 0.25 * decay
                )

            # 按综合分数排序
            sorted_memories = sorted(
                candidates, key=lambda m: m["final_score"], reverse=True
            )

            # 第四层: 数量限制
            result = sorted_memories[:MAX_LONG_TERM_MEMORIES]

            # 更新访问统计
            for mem in result:
                await self._update_access_stats(mem["id"])

            return result

        except Exception as e:
            logger.error(f"❌ 分层检索失败: {e}")
            raise

    async def _update_access_stats(self, memory_id: str):
        """
        更新记忆的访问统计(检索命中次数+1，更新最后访问时间)

        Args:
            memory_id: 记忆ID
        """
        try:
            async with self.async_session() as session:
                stmt = (
                    update(long_term_memories_v2)
                    .where(long_term_memories_v2.c.id == memory_id)
                    .values(
                        access_count=long_term_memories_v2.c.access_count + 1,
                        last_accessed_at=datetime.now(timezone.utc),
                    )
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as e:
            logger.warning(f"更新访问统计失败: {e}")

    # ====================== 兼容旧接口 ======================

    async def save_memory(
        self,
        user_id: str,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        兼容旧接口的保存方法

        当没有LLM时直接保存(降级为V1行为)，有LLM时走完整提取流程

        Args:
            user_id: 用户标识
            content: 记忆内容
            session_id: 会话标识
            metadata: 额外元数据

        Returns:
            记忆ID
        """
        if self.llm:
            # 有LLM时，content被视为原始对话内容，走提取流程
            question = metadata.get("question", "") if metadata else ""
            context = metadata.get("context") if metadata else None
            result = await self.save_from_conversation(
                user_id=user_id,
                question=question,
                answer=content,
                session_id=session_id,
                context=context,
            )
            return result or ""

        # 无LLM时降级: 直接保存原始内容
        try:
            embedding = self.embeddings.embed_query(content)
            source_session_ids = [session_id] if session_id else []

            async with self.async_session() as session:
                stmt = (
                    insert(long_term_memories_v2)
                    .values(
                        user_id=user_id,
                        memory_type="fact",
                        content=content,
                        summary=content[:30],
                        embedding=embedding,
                        value_score=0.5,
                        metadata=metadata or {},
                        source_session_ids=source_session_ids,
                    )
                    .returning(long_term_memories_v2.c.id)
                )
                result = await session.execute(stmt)
                await session.commit()
                return str(result.scalar())
        except Exception as e:
            logger.error(f"❌ 保存长期记忆V2失败: {e}")
            raise

    # ====================== 管理操作 ======================

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """
        删除指定记忆

        Args:
            memory_id: 记忆ID
            user_id: 用户标识(确保只能删除自己的记忆)

        Returns:
            是否删除成功
        """
        try:
            async with self.async_session() as session:
                stmt = delete(long_term_memories_v2).where(
                    long_term_memories_v2.c.id == memory_id,
                    long_term_memories_v2.c.user_id == user_id,
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"❌ 删除长期记忆V2失败: {e}")
            raise

    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        获取记忆统计信息

        Args:
            user_id: 用户标识

        Returns:
            统计信息字典(含各类型数量)
        """
        try:
            async with self.async_session() as session:
                # 总记忆数
                count_stmt = select(func.count()).select_from(
                    long_term_memories_v2
                ).where(long_term_memories_v2.c.user_id == user_id)
                total = (await session.execute(count_stmt)).scalar()

                # 各类型数量
                type_stmt = (
                    select(
                        long_term_memories_v2.c.memory_type,
                        func.count().label("count"),
                    )
                    .where(long_term_memories_v2.c.user_id == user_id)
                    .group_by(long_term_memories_v2.c.memory_type)
                )
                type_result = (await session.execute(type_stmt)).fetchall()
                type_counts = {row.memory_type: row.count for row in type_result}

                # 最近记忆时间
                latest_stmt = select(func.max(long_term_memories_v2.c.created_at)).where(
                    long_term_memories_v2.c.user_id == user_id
                )
                latest = (await session.execute(latest_stmt)).scalar()

                # 平均价值分数
                avg_stmt = select(func.avg(long_term_memories_v2.c.value_score)).where(
                    long_term_memories_v2.c.user_id == user_id
                )
                avg_score = (await session.execute(avg_stmt)).scalar()

                return {
                    "total_memories": total,
                    "type_counts": type_counts,
                    "latest_memory_at": latest,
                    "avg_value_score": round(float(avg_score or 0), 3),
                }
        except Exception as e:
            logger.error(f"❌ 获取长期记忆V2统计失败: {e}")
            raise

    async def get_all_user_ids(self) -> List[str]:
        """获取所有有记忆的用户ID列表"""
        try:
            async with self.async_session() as session:
                stmt = select(long_term_memories_v2.c.user_id).distinct()
                result = await session.execute(stmt)
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error(f"❌ 获取用户ID列表失败: {e}")
            raise

    async def close(self):
        """关闭连接池"""
        try:
            if self.engine:
                await self.engine.dispose()
            logger.info("✅ 长期记忆V2数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接时出错: {e}")

    # ====================== 定时清理 ======================

    async def cleanup_old_memories(self, user_id: str, retention_days: int = 90) -> int:
        """
        清理指定用户的旧记忆(超过保留天数且价值分数低)

        Args:
            user_id: 用户标识
            retention_days: 记忆保留天数

        Returns:
            删除的记忆数量
        """
        try:
            async with self.async_session() as session:
                delete_date = f"now() - interval '{retention_days} days'"
                # 只删除低价值的旧记忆，高价值的保留更久
                stmt = text(f"""
                    DELETE FROM long_term_memories_v2
                    WHERE user_id = :user_id
                        AND created_at < {delete_date}
                        AND value_score < 0.7
                """)
                result = await session.execute(stmt, {"user_id": user_id})
                await session.commit()

                deleted_count = result.rowcount
                logger.info(
                    f"用户 {user_id} 清理了 {deleted_count} 条低价值旧记忆"
                )
                return deleted_count
        except Exception as e:
            logger.error(f"❌ 清理旧长期记忆V2失败: {e}")
            raise

    async def cleanup_duplicates(self, user_id: str, similarity_threshold: float = 0.95) -> int:
        """
        清理指定用户的重复记忆

        Args:
            user_id: 用户标识
            similarity_threshold: 相似度阈值

        Returns:
            删除的重复记忆数量
        """
        try:
            async with self.async_session() as session:
                stmt = text("""
                    DELETE FROM long_term_memories_v2
                    WHERE id IN (
                        SELECT DISTINCT m1.id
                        FROM long_term_memories_v2 m1
                        JOIN long_term_memories_v2 m2
                            ON m1.user_id = m2.user_id
                            AND m1.memory_type = m2.memory_type
                            AND m1.id > m2.id
                            AND 1 - (m1.embedding <=> m2.embedding) >= :threshold
                        WHERE m1.user_id = :user_id
                    )
                """)
                result = await session.execute(stmt, {
                    "user_id": user_id,
                    "threshold": similarity_threshold,
                })
                await session.commit()

                deleted_count = result.rowcount
                logger.info(f"用户 {user_id} 清理了 {deleted_count} 条重复记忆V2")
                return deleted_count
        except Exception as e:
            logger.error(f"❌ 清理重复记忆V2失败: {e}")
            raise

    async def cleanup_low_value(self, user_id: str) -> int:
        """
        清理低价值记忆(value_score < 0.4)

        Args:
            user_id: 用户标识

        Returns:
            删除的记忆数量
        """
        try:
            async with self.async_session() as session:
                stmt = delete(long_term_memories_v2).where(
                    long_term_memories_v2.c.user_id == user_id,
                    long_term_memories_v2.c.value_score < 0.4,
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount
        except Exception as e:
            logger.error(f"❌ 清理低价值记忆失败: {e}")
            raise

    async def enforce_memory_limit(self, user_id: str) -> int:
        """
        执行容量控制: 超出限制时按综合分数淘汰末位10%

        Args:
            user_id: 用户标识

        Returns:
            淘汰的记忆数量
        """
        try:
            stats = await self.get_stats(user_id)
            total = stats["total_memories"]

            if total <= self.max_memories_per_user:
                return 0

            # 淘汰末位10%
            to_delete = max(1, int(total * 0.1))

            async with self.async_session() as session:
                # 按综合分数排序，删除最低的
                now = datetime.now(timezone.utc)
                stmt = text("""
                    DELETE FROM long_term_memories_v2
                    WHERE id IN (
                        SELECT id FROM long_term_memories_v2
                        WHERE user_id = :user_id
                        ORDER BY value_score ASC, created_at ASC
                        LIMIT :limit
                    )
                """)
                result = await session.execute(stmt, {
                    "user_id": user_id,
                    "limit": to_delete,
                })
                await session.commit()

                deleted = result.rowcount
                logger.info(f"用户 {user_id} 容量控制淘汰了 {deleted} 条记忆")
                return deleted

        except Exception as e:
            logger.error(f"❌ 容量控制失败: {e}")
            raise


def _is_contradictory(existing: str, new: str) -> bool:
    """
    检查新内容是否与已有内容矛盾

    简单启发式: 检查是否存在对立表述

    Args:
        existing: 已有内容
        new: 新内容

    Returns:
        True表示存在矛盾
    """
    contradiction_pairs = [
        ("喜欢", "不喜欢"), ("偏好", "不用"), ("用", "不用"),
        ("推荐", "不推荐"), ("支持", "不支持"), ("可以", "不可以"),
    ]
    for pos, neg in contradiction_pairs:
        if (pos in existing and neg in new) or (neg in existing and pos in new):
            return True
    return False


def _merge_content(existing: str, new: str) -> str:
    """
    合并两条记忆内容

    策略: 保留更丰富的描述，如果新内容有额外信息则追加

    Args:
        existing: 已有内容
        new: 新内容

    Returns:
        合并后的内容(不超过200字)
    """
    if new in existing:
        return existing

    if existing in new:
        return new

    # 新内容有额外信息，追加
    merged = f"{existing}；{new}"
    if len(merged) > 200:
        merged = merged[:200]
    return merged
