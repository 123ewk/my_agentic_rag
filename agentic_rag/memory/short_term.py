"""
短期记忆实现(PostgreSQL持久化)

生产环境优化：
1. 异步数据库连接池，支持高并发
2. 按会话隔离，支持多用户并发
3. 支持TTL自动过期清理
4. 基于token数量智能截断,而非简单消息数量
5. 提供分页查询能力
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select, delete, func
from sqlalchemy.dialects.postgresql import insert
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from loguru import logger

from ..models.short_term_model import conversation_sessions

class ShortTermMemory:
    """短期记忆管理器(PostgreSQL持久化)"""
    
    def __init__(
        self,
        database_url: str,
        max_messages: int = 20,
        max_tokens: int = 5000,
        ttl_hours: int = 24,
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        """
        初始化短期记忆
        
        Args:
            database_url: PostgreSQL连接字符串,如 'postgresql+asyncpg://user:pass@host:5432/db'
            max_messages: 最大保留消息数
            max_tokens: 最大token数(用于截断上下文)
            ttl_hours: 会话过期时间（小时）
            pool_size: 连接池基础大小
            max_overflow: 连接池最大溢出
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.ttl_hours = ttl_hours
        
        # 创建异步引擎和连接池
        # 数据库引擎（先不创建，放到异步方法里重试创建）
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.engine = None
        
        # 创建异步会话工厂,用来生成数据库会话对象，后面增删改查都要靠它。
        self.async_session = None


    # ====================== 【核心：带重试的数据库连接】 ======================
    @retry(
        stop=stop_after_attempt(5),  # 最多重试 5 次
        wait=wait_exponential(multiplier=1, min=1, max=10),  # 指数退避：1s → 2s → 4s → 8s...最大10s
        retry=retry_if_exception_type((SQLAlchemyError, ConnectionError)),  # 只重试数据库/连接类异常
        reraise=True,  # 重试失败后继续抛出错误
    )
    async def connect(self):
        """
        带自动重试的数据库连接方法
        """
        try:
            # 创建异步引擎
            self.engine = create_async_engine(
                self.database_url,
                pool_size=self.pool_size, # 避免每次请求都新建连接，复用连接减少开销
                max_overflow=self.max_overflow, # 应对突发高流量，防止请求排队超时
                pool_timeout=30,    # 30秒超时
                pool_recycle=1800,  # 30分钟回收连接,解决数据库主动断开闲置连接导致的 “死连接” 问题
                echo=False  # 生产环境关闭SQL日志,避免日志泄露 SQL 语句，影响性能
            )

            # 创建异步会话工厂,用来生成数据库会话对象，后面增删改查都要靠它。
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,   # 明确指定使用异步会话类，确保所有操作都是异步的，不会阻塞事件循环。
                expire_on_commit=False  # 禁用自动过期提交,设置为 False 后，提交后对象依然可用，避免不必要的重复查询，提升性能。
            )

            # 测试连接是否真的可用（关键！）
            async with self.engine.begin():
                logger.info("✅ 数据库连接成功")

        except SQLAlchemyError as e:
            logger.error("❌ 数据库连接失败，即将重试：{}", str(e))
            raise  # 抛出让 tenacity 重试
        except Exception as e:
            logger.error("❌ 短期记忆,数据库连接未知错误：{}", str(e))
            raise
    
    async def add_message(
        self,
        session_id: str,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        添加对话消息到短期记忆
        
        Args:
            session_id: 会话唯一标识
            question: 用户问题
            answer: AI回答
            metadata: 额外元数据（如意图、工具使用等）
            
        Returns:
            消息索引号
        """
        try:
            expires_at = datetime.now() + timedelta(hours=self.ttl_hours)
            
            async with self.async_session() as session:
                # 获取当前消息索引
                result = await session.execute(
                    select(func.max(conversation_sessions.c.message_index))
                    .where(conversation_sessions.c.session_id == session_id)
                )
                max_index = result.scalar() or -1  # 拿到结果，没有则返回 -1
                next_index = max_index + 1
                
                # 插入用户消息
                user_msg = insert(conversation_sessions).values(
                    session_id=session_id,
                    message_index=next_index,
                    role='user',
                    content=question,
                    metadata=metadata or {},
                    token_count=self._estimate_tokens(question),
                    expires_at=expires_at
                )
                await session.execute(user_msg)
                
                # 插入AI回复
                ai_msg = insert(conversation_sessions).values(
                    session_id=session_id,
                    message_index=next_index + 1,
                    role='assistant',
                    content=answer,
                    metadata=metadata or {},
                    token_count=self._estimate_tokens(answer),
                    expires_at=expires_at
                )
                await session.execute(ai_msg)
                
                await session.commit()
                return next_index
        except Exception as e:
            logger.error("❌ 添加消息到短期记忆失败：{}", str(e))
            raise
    
    async def get_message(
        self,
        session_id: str,
        limit:Optional[int] = None
    ) -> List[BaseMessage]:
        """
        获取会话历史消息
        
        Args:
            session_id: 会话唯一标识
            limit: 限制返回数量(None表示使用max_messages)
            
        Returns:
            LangChain消息列表
        """
        try:
            actual_limit = limit or self.max_messages

            async with self.async_session() as session:
                # 查询最近N条消息
                stmt = (
                    select(conversation_sessions)
                    .where(conversation_sessions.c.session_id == session_id)
                    .order_by(conversation_sessions.c.message_index.desc())
                    .limit(actual_limit)
                )
                result = await session.execute(stmt)
                rows = result.fetchall() # 拿到所有结果
                
                # 反转并按角色转换为LangChain消息
                messages = []
                for row in reversed(rows):
                    if row.role == 'user':
                        messages.append(HumanMessage(content=row.content))
                    elif row.role == 'assistant':
                        messages.append(AIMessage(content=row.content))
                
                return messages
        except Exception as e:
            logger.error("❌ 获取会话历史消息失败：{}", str(e))
            raise

    async def get_context(
        self,
        session_id: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        获取记忆上下文(基于token限制)
        
        Args:
            session_id: 会话唯一标识
            max_tokens: 最大token数(默认使用实例配置)
            
        Returns:
            格式化的上下文字符串
        """
        try:
            token_limit = max_tokens or self.max_tokens
            messages = await self.get_message(session_id)
            
            # 从后往前累加，直到达到token限制
            context_messages = []
            total_tokens = 0
            
            for msg in reversed(messages):
                msg_tokens = self._estimate_tokens(msg.content)
                if total_tokens + msg_tokens > token_limit:
                    break
                context_messages.insert(0, msg)
                total_tokens += msg_tokens
            
            # 格式化为字符串
            context_lines = []
            for msg in context_messages:
                role = "用户" if isinstance(msg, HumanMessage) else "助手"
                context_lines.append(f"{role}: {msg.content}")
            
            return "\n".join(context_lines)
        except Exception as e:
            logger.error("❌ 获取记忆上下文失败：{}", str(e))
            raise
    
    async def clear_session(self, session_id: str) -> int:
        """
        清空指定会话的记忆
        
        Args:
            session_id: 会话唯一标识
            
        Returns:
            删除的消息数量
        """
        try:
            async with self.async_session() as session:
                stmt = delete(conversation_sessions).where(
                    conversation_sessions.c.session_id == session_id
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount
        except Exception as e:
            logger.error("❌ 清空会话记忆失败：{}", str(e))
            raise
    
    async def cleanup_expired(self) -> int:
        """
        清理过期会话（可由定时任务调用）
        
        Returns:
            删除的会话数量
        """
        try:
            async with self.async_session() as session:
                stmt = delete(conversation_sessions).where(
                    conversation_sessions.c.expires_at < datetime.now()
                )
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount
        except Exception as e:
            logger.error("❌ 清理过期会话失败：{}", str(e))
            raise



    # ====================== 优雅关闭连接 ======================
    async def close(self):
        """关闭数据库连接池"""
        try:
            if self.engine:
                await self.engine.dispose()
                logger.info("✅ 数据库连接已关闭")
        except Exception as e:
            logger.error("❌ 关闭数据库连接失败：{}", str(e))
            raise
        


    @staticmethod
    def _estimate_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
        """
        本地计算Token,无需模型、无需API
        """
        # 第一次运行会自动下载几十KB的分词规则，秒完成
        enc = tiktoken.encoding_for_model(model_name)
        return len(enc.encode(text))






# # 1. 创建实例
# memory = ShortTermMemoryManager(database_url="postgresql+asyncpg://user:pass@localhost/db")

# # 2. 调用带重试的连接方法
# await memory.connect()

# # 之后正常使用
# async with memory.async_session() as session:
#     ...  # 你的业务逻辑