from loguru import logger
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Any
import asyncio



# ==================== PostgreSQL 锁 ====================

class PostgresLockManager:
    """
    PostgreSQL锁管理器
    
    使用场景：
    1. 会话记忆更新:防止并发写入导致数据覆盖
    2. 文档索引去重:基于数据库的行级锁
    3. 事务保护:确保多表操作的原子性
    
    支持的锁类型:
    - 行级锁(SELECT FOR UPDATE):锁定特定行（防止并发修改）
    - 表级锁(LOCK TABLE):锁定整个表（防止并发修改）
    - 咨询锁(pg_advisory_lock):用于在多个数据库实例之间协调锁，确保一致的锁行为
    """
    
    def __init__(self, async_session):
        """
        初始化PostgreSQL锁管理器
        
        Args:
            async_session: SQLAlchemy异步会话工厂
        """
        self.async_session = async_session
    
    async def lock_row(
        self,
        table,
        row_id: str,
        nowait: bool = False
    ) -> Optional[Any]:
        """
        行级锁：锁定特定行（防止并发修改）
        
        使用场景：更新会话记忆时防止多个请求同时修改同一会话
        
        Args:
            table: SQLAlchemy表对象
            row_id: 行ID
            nowait: 如果锁不可用是否立即返回(True不等待,False等待)
            
        Returns:
            锁定的行,获取失败返回None
        """
        async with self.async_session() as session:
            try:
                # SELECT ... FOR UPDATE [NOWAIT]
                if nowait:
                    stmt = select(table).where(
                        table.c.id == row_id
                    ).with_for_update(nowait=True)  # nowait=True 的特殊效果:nowait=True 会让它不等待、直接报错
                    # .with_for_update(nowait=True) 是 SQLAlchemy 中实现悲观锁（排他行锁）的方法，
                else:
                    stmt = select(table).where(
                        table.c.id == row_id
                    ).with_for_update() # 默认情况下，如果目标行已经被其他事务锁住，当前查询会一直阻塞等待锁释放；
                
                result = await session.execute(stmt)
                row = result.scalar_one_or_none() # scalar_one_or_none() 是 SQLAlchemy 中 异步查询结果解析 的核心方法。
                
                if row:
                    return row
                
                return None
                
            except Exception as e:
                # 锁冲突或行不存在
                logger.error(f"行级锁获取失败: {e}")
                return None
    
    async def lock_session_messages(
        self,
        session_id: str
    ) -> Optional[Any]:
        """
        锁定会话消息（用于追加消息时的并发保护）
        
        使用场景:多个用户请求同时向同一会话追加消息时，
        确保消息索引不会冲突
        
        Args:
            session_id: 会话ID
            
        Returns:
            当前最大消息索引,获取失败返回None
        """
        async with self.async_session() as session:
            try:
                # 锁定会话的最新一行消息
                stmt = (
                    select(func.max(conversation_sessions.c.message_index))
                    .where(conversation_sessions.c.session_id == session_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                max_index = result.scalar()
                
                return max_index or -1
                
            except Exception as e:
                print(f"会话消息锁获取失败: {e}")
                return None
    
    async def advisory_lock_context(
        self,
        lock_id: int,
        blocking: bool = True,
        timeout: float = 10.0
    ):
        """
        获取咨询锁的上下文管理器(确保锁的获取和释放在同一个session中)
        
        特点：
        - 不关联任何具体数据行
        - 由应用程序自行决定锁的语义
        - 支持跨会话锁定
        - 通过上下文管理器确保 session 生命周期一致，避免锁泄漏
        
        使用场景：定时任务互斥、全局操作保护
        
        使用示例：
        ```python
        lock_manager = PostgresLockManager(async_session)
        
        # 非阻塞模式
        async with lock_manager.advisory_lock_context(user_id) as acquired:
            if acquired:
                # 获取到锁，执行任务
                await process_user_task(user_id)
        
        # 阻塞模式（等待锁释放）
        async with lock_manager.advisory_lock_context(user_id, blocking=True) as acquired:
            if acquired:
                await process_user_task(user_id)
        ```
        
        Args:
            lock_id: 锁ID(64位整数,可用用户ID实现"同一用户同时只能执行一个任务",用订单ID实现"同一订单并发操作互斥")
            blocking: 是否阻塞等待锁释放(默认True)
            timeout: 阻塞模式下的超时时间(秒)，仅在 blocking=True 时有效
            
        Returns:
            AdvisoryLockContext: 上下文管理器，进入时返回是否获取成功
        """
        return AdvisoryLockContext(self, lock_id, blocking, timeout)


class AdvisoryLockContext:
    """
    PostgreSQL咨询锁上下文管理器
    
    通过上下文管理器确保锁的获取和释放在同一个数据库会话中进行，
    避免因 session 生命周期不匹配导致的锁泄漏问题。
    
    使用方式：
    ```python
    lock_manager = PostgresLockManager(async_session)
    
    # 非阻塞模式：立即返回获取结果
    async with lock_manager.advisory_lock_context(user_id, blocking=False) as acquired:
        if acquired:
            await process_user_task(user_id)
    
    # 阻塞模式：等待直到获取锁或超时
    async with lock_manager.advisory_lock_context(user_id, blocking=True, timeout=5.0) as acquired:
        if acquired:
            await process_user_task(user_id)
    ```
    
    注意：
    - PostgreSQL 咨询锁与获取它的数据库会话绑定
    - 锁会在会话结束时自动释放（但显式释放更安全）
    - 阻塞模式下使用 pg_advisory_lock()，非阻塞模式使用 pg_try_advisory_lock()
    """
    
    def __init__(
        self,
        lock_manager: PostgresLockManager,
        lock_id: int,
        blocking: bool = True,
        timeout: float = 10.0
    ):
        """
        初始化咨询锁上下文管理器
        
        Args:
            lock_manager: PostgresLockManager 实例
            lock_id: 锁ID(64位整数)
            blocking: 是否阻塞等待锁释放(默认True)
            timeout: 阻塞超时时间(秒)，仅在 blocking=True 时有效
        """
        self.lock_manager = lock_manager
        self.lock_id = lock_id
        self.blocking = blocking
        self.timeout = timeout
        self.acquired = False
        self.session = None
    
    async def __aenter__(self) -> bool:
        """
        进入上下文：尝试获取咨询锁
        
        Returns:
            是否成功获取锁
        
        Raises:
            asyncio.TimeoutError: 阻塞模式下等待超时
        """
        try:
            self.session = self.lock_manager.async_session()
            
            if self.blocking:
                result = await self.session.execute(
                    text("SELECT pg_advisory_lock(:lock_id)"),
                    {"lock_id": self.lock_id}
                )
                self.acquired = result.scalar_one() is not None
            else:
                result = await self.session.execute(
                    text("SELECT pg_try_advisory_lock(:lock_id)"),
                    {"lock_id": self.lock_id}
                )
                self.acquired = result.scalar_one()
            
            return self.acquired
            
        except asyncio.TimeoutError:
            logger.warning(f"咨询锁 {self.lock_id} 获取超时（等待 {self.timeout} 秒）")
            await self._cleanup_session()
            raise
        except Exception as e:
            logger.error(f"咨询锁 {self.lock_id} 获取失败: {e}")
            await self._cleanup_session()
            return False
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文：释放咨询锁
        
        Args:
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常回溯
        
        Returns:
            False: 不抑制异常
        """
        if self.session:
            try:
                await self.session.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"),
                    {"lock_id": self.lock_id}
                )
                logger.debug(f"咨询锁 {self.lock_id} 已释放")
            except Exception as e:
                logger.error(f"咨询锁 {self.lock_id} 释放失败: {e}")
            finally:
                await self._cleanup_session()
        
        return False
    
    async def _cleanup_session(self):
        """清理数据库会话资源"""
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.warning(f"会话清理时出现异常: {e}")
            finally:
                self.session = None





if __name__ == "__main__":
    async def example_session_lock():   
        f"""示例:PostgreSQL会话锁(使用咨询锁上下文管理器)"""
        
        from memory.short_term import ShortTermMemory
        
        # 获取数据库会话
        short_term_memory = ShortTermMemory(database_url="...")
        pg_lock = PostgresLockManager(short_term_memory.async_session)
        
        user_id = "user_123"
        session_id = "user_123_session"
        
        # 方式1：使用咨询锁保护用户操作（非阻塞模式）
        # 确保同一用户同时只能执行一个任务
        async with pg_lock.advisory_lock_context(user_id, blocking=False) as acquired:
            if acquired:
                # 获取到锁，执行任务
                print(f"用户 {user_id} 开始执行任务")
                # ... 业务逻辑 ...
            else:
                # 未获取到锁，跳过或等待
                print(f"用户 {user_id} 正在执行其他任务，跳过")
        
        # 方式2：使用咨询锁保护会话消息（阻塞模式）
        # 确保同一会话的消息追加操作互斥
        async with pg_lock.advisory_lock_context(hash(session_id), blocking=True, timeout=5.0) as acquired:
            if acquired:
                # 获取到会话锁，防止并发修改消息索引
                max_index = await pg_lock.lock_session_messages(session_id)
                next_index = (max_index or -1) + 1
                
                # 安全地插入消息
                await short_term_memory.add_message(
                    session_id=session_id,
                    question="你好",
                    answer="你好！有什么我可以帮你的？",
                    message_index=next_index
                )
                print(f"消息已添加到会话 {session_id}，索引: {next_index}")
            else:
                print(f"会话 {session_id} 已被锁定，等待超时")
        
        # 方式3：使用行级锁（SELECT FOR UPDATE）
        # 锁定特定行进行更新
        async with pg_lock.advisory_lock_context(hash(f"row:{session_id}"), blocking=True) as acquired:
            if acquired:
                # 锁定会话，防止并发修改
                row = await pg_lock.lock_row(session_id)
                if row:
                    print(f"已锁定会话行: {session_id}")
                    # ... 执行更新操作 ...