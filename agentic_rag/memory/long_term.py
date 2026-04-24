"""
长期记忆实现(PostgreSQL + pgvector)

生产环境优化：
1. 使用pgvector进行向量相似度搜索,无需外部向量数据库
2. 支持按用户隔离，确保数据隐私
3. 批量插入优化，减少数据库往返
4. 支持元数据过滤查询
5. HNSW索引加速检索
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, delete, func, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.embeddings import Embeddings

from ..models.long_term_model import long_term_memories
from loguru import logger



class LongTermMemory:
    """长期记忆管理器(PostgreSQL + pgvector) 1024维向量"""
    
    def __init__(
        self,
        embeddings: Embeddings,
        database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/agentic_rag",
        k: int = 5,
        similarity_threshold: float = 0.7,
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        """
        初始化长期记忆
        
        Args:
            database_url: PostgreSQL连接字符串
            embeddings: 嵌入模型实例
            k: 检索相似记忆数量
            similarity_threshold: 相似度阈值(0-1)
            pool_size: 连接池基础大小
            max_overflow: 连接池最大溢出
        """
        self.database_url = database_url
        self.embeddings = embeddings
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.pool_size = pool_size
        self.max_overflow = max_overflow


    @retry(
        stop=stop_after_attempt(5),  # 最多重试 5 次
        wait=wait_exponential(multiplier=1, min=1, max=10),  # 指数退避：1s → 2s → 4s → 8s...最大10s
        retry=retry_if_exception_type((SQLAlchemyError, ConnectionError)),  # 只重试数据库/连接类异常
        reraise=True,  # 重试失败后继续抛出错误
    )
    async def connect(self):
        try:
            # 创建异步引擎
            self.engine = create_async_engine(
                self.database_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=30,
                pool_recycle=1800, # 30分钟回收连接
                echo=False
            )
            
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # 测试连接是否真的可用（关键！）
            async with self.engine.begin():
                logger.info("✅ 数据库连接成功")

        except SQLAlchemyError as e:
            logger.error("❌ 数据库连接失败，即将重试：{}", str(e))
            raise  # 抛出让 tenacity 重试
        except Exception as e:
            logger.error("❌ 未知错误：{}", str(e))
            raise
    
    async def save_memory(
        self,
        user_id: str,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        保存长期记忆
        
        Args:
            user_id: 用户标识
            content: 记忆内容
            session_id: 会话标识（可选）
            metadata: 额外元数据
            
        Returns:
            记忆ID
        """
        # 生成嵌入向量
        try:
            embedding = self.embeddings.embed_query(content)
            
            async with self.async_session() as session:
                # 插入记忆
                stmt = insert(long_term_memories).values(
                    user_id=user_id,
                    session_id=session_id,
                    content=content,
                    embedding=embedding,
                    metadata=metadata or {}
                ).returning(long_term_memories.c.id) # 让这条 INSERT 语句执行成功后，把新插入行的 id 字段值返回给你。
                
                result = await session.execute(stmt)
                await session.commit()
                
                memory_id = result.scalar()
                return str(memory_id)
        except Exception as e:
            logger.error("❌ 保存长期记忆失败:{}", str(e))
            raise
    
    async def save_batch(
        self,
        user_id: str,
        contents: List[str],
        session_id: Optional[str] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        批量保存记忆（优化：一次性生成所有嵌入）
        
        Args:
            user_id: 用户标识
            contents: 记忆内容列表
            session_id: 会话标识
            metadata_list: 元数据列表
            
        Returns:
            记忆ID列表
        """
        try:
            # 批量生成嵌入（比逐个调用快3-5倍）
            embeddings = self.embeddings.embed_documents(contents)
        
            async with self.async_session() as session:
                memory_ids = []
                
                for content, embedding in zip(contents, embeddings):
                    metadata = metadata_list.pop(0) if metadata_list else {}
                    
                    stmt = insert(long_term_memories).values(
                        user_id=user_id,
                        session_id=session_id,
                        content=content,
                        embedding=embedding,
                        metadata=metadata
                    ).returning(long_term_memories.c.id)
                    
                    result = await session.execute(stmt)
                    memory_ids.append(str(result.scalar()))
                
                await session.commit()
                return memory_ids
        except Exception as e:
            logger.error("❌ 批量保存长期记忆失败:{}", str(e))
            raise
    
    async def search(
        self,
        user_id: str,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似记忆
        
        Args:
            user_id: 用户标识
            query: 查询文本
            k: 检索数量
            metadata_filter: 元数据过滤条件
            
        Returns:
            相似记忆列表（含相似度分数）
        """
        try:
            
            actual_k = k or self.k
            
            # 生成查询向量
            query_embedding = self.embeddings.embed_query(query)
            
            async with self.async_session() as session:
                # 使用pgvector的余弦相似度搜索
                # 注意：这里使用原生SQL因为SQLAlchemy的vector支持有限
                # <=> 表示余弦距离（Cosine Distance），计算两个向量之间的 “距离”
                # 余弦距离的取值范围是 [0, 2]
                # 距离越接近 0，两个向量越相似；
                # 1 - (embedding <=> :query_embedding) as similarity, 这是把余弦距离转换成余弦相似度的常用写法
                # 转换后，值的范围变成 [-1, 1]，越接近 1 代表两个向量越相似

                sql = text("""
                    SELECT 
                        id, user_id, content, metadata, created_at,
                        1 - (embedding <=> :query_embedding) as similarity
                    FROM long_term_memories
                    WHERE user_id = :user_id
                        AND 1 - (embedding <=> :query_embedding) >= :threshold
                        {metadata_filter}
                    ORDER BY embedding <=> :query_embedding
                    LIMIT :k
                """)  # :(sql占位符) 防 SQL 注入  距离越小（越相似）的记忆，排在越前面,再配合后面的 LIMIT :k，就能取出Top K 个最相似的记忆 

                # 构建元数据过滤条件
                metadata_clause = ""
                params = {
                    "user_id": user_id,
                    "query_embedding": str(query_embedding),
                    "threshold": self.similarity_threshold,
                    "k": actual_k
                }
                
                if metadata_filter:
                    filter_conditions = []
                    for key, value in metadata_filter.items():
                        filter_conditions.append(f"metadata->>'{key}' = :filter_{key}")
                        params[f"filter_{key}"] = str(value)
                    metadata_clause = "AND " + " AND ".join(filter_conditions)
                
                final_sql = text(str(sql).format(metadata_filter=metadata_clause))
                
                result = await session.execute(final_sql, params)
                rows = result.fetchall()
                
                return [
                    {
                        "id": str(row.id),
                        "content": row.content,
                        "metadata": row.metadata,
                        "created_at": row.created_at,
                        "similarity": float(row.similarity)
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error("❌ 搜索长期记忆失败:{}", str(e))
            raise
    
    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """
        删除指定记忆
        
        Args:
            memory_id: 记忆ID
            user_id: 用户标识（确保只能删除自己的记忆）
            
        Returns:
            是否删除成功
        """
        try:
            async with self.async_session() as session:
                stmt = delete(long_term_memories).where(
                    long_term_memories.c.id == memory_id,
                    long_term_memories.c.user_id == user_id
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        except Exception as e:
            logger.error("❌ 删除长期记忆失败:{}", str(e))
            raise
    
    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Args:
            user_id: 用户标识
            
        Returns:
            统计信息字典
        """
        try:
            async with self.async_session() as session:
                # 总记忆数
                count_stmt = select(func.count()).select_from(long_term_memories).where(
                    long_term_memories.c.user_id == user_id
                )
                total = (await session.execute(count_stmt)).scalar() 
                
                # 最近记忆时间
                latest_stmt = select(func.max(long_term_memories.c.created_at)).where(
                    long_term_memories.c.user_id == user_id
                )
                latest = (await session.execute(latest_stmt)).scalar()
                
                return {
                    "total_memories": total,
                    "latest_memory_at": latest
                }
        except Exception as e:
            logger.error("❌ 获取长期记忆统计失败:{}", str(e))
            raise
    
    async def close(self):
        """关闭连接池"""
        try:
            if self.engine:
                await self.engine.dispose()
            logger.info("✅ 数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接时出错: {e}")
           
