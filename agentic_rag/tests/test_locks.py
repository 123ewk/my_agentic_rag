"""
锁机制模块测试
测试PostgreSQL锁和Redis分布式锁
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional


class TestPostgresLockManagerMocked:
    """PostgreSQL锁管理器测试（模拟）"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """测试初始化"""
        from agentic_rag.lock.postgresql_lock import PostgresLockManager
        
        mock_session = MagicMock()
        lock_manager = PostgresLockManager(mock_session)
        
        assert lock_manager.async_session is not None

    @pytest.mark.asyncio
    async def test_advisory_lock_context_creation(self):
        """测试咨询锁上下文创建"""
        from agentic_rag.lock.postgresql_lock import PostgresLockManager, AdvisoryLockContext
        
        mock_session = MagicMock()
        lock_manager = PostgresLockManager(mock_session)
        
        context = lock_manager.advisory_lock_context(lock_id=12345)
        
        assert isinstance(context, AdvisoryLockContext)
        assert context.lock_id == 12345
        assert context.blocking is True
        assert context.timeout == 10.0

    @pytest.mark.asyncio
    async def test_advisory_lock_context_non_blocking(self):
        """测试非阻塞咨询锁上下文"""
        from agentic_rag.lock.postgresql_lock import PostgresLockManager
        
        mock_session = MagicMock()
        lock_manager = PostgresLockManager(mock_session)
        
        context = lock_manager.advisory_lock_context(
            lock_id=12345,
            blocking=False
        )
        
        assert context.blocking is False


class TestAdvisoryLockContextMocked:
    """咨询锁上下文管理器测试（模拟）"""

    @pytest.mark.asyncio
    async def test_context_initialization(self):
        """测试上下文初始化"""
        from agentic_rag.lock.postgresql_lock import AdvisoryLockContext, PostgresLockManager
        
        mock_session = MagicMock()
        lock_manager = PostgresLockManager(mock_session)
        
        context = AdvisoryLockContext(
            lock_manager=lock_manager,
            lock_id=12345,
            blocking=True,
            timeout=5.0
        )
        
        assert context.lock_id == 12345
        assert context.blocking is True
        assert context.timeout == 5.0
        assert context.acquired is False
        assert context.session is None

    @pytest.mark.asyncio
    async def test_context_aenter_non_blocking(self):
        """测试非阻塞模式进入上下文"""
        from agentic_rag.lock.postgresql_lock import AdvisoryLockContext, PostgresLockManager
        
        mock_session_factory = MagicMock()
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session
        
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = True
        mock_session.execute.return_value = mock_result
        
        lock_manager = PostgresLockManager(mock_session_factory)
        
        context = AdvisoryLockContext(
            lock_manager=lock_manager,
            lock_id=12345,
            blocking=False
        )
        
        result = await context.__aenter__()
        
        assert result is True
        assert context.acquired is True

    @pytest.mark.asyncio
    async def test_context_aexit_release(self):
        """测试退出上下文释放锁"""
        from agentic_rag.lock.postgresql_lock import AdvisoryLockContext, PostgresLockManager
        
        mock_session_factory = MagicMock()
        mock_session = MagicMock()
        mock_session_factory.return_value = mock_session
        
        lock_manager = PostgresLockManager(mock_session_factory)
        
        context = AdvisoryLockContext(
            lock_manager=lock_manager,
            lock_id=12345
        )
        context.session = mock_session
        context.acquired = True
        
        await context.__aexit__(None, None, None)
        
        mock_session.execute.assert_called()
        mock_session.close.assert_called()


class TestRedisDistributedLockMocked:
    """Redis分布式锁测试（模拟）"""

    def test_initialization(self):
        """测试初始化"""
        from agentic_rag.lock.redis_lock import RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        lock = RedisDistributedLock(
            redis_cache=mock_redis_cache,
            lock_timeout=30,
            retry_interval=0.1,
            max_retry=50
        )
        
        assert lock.lock_timeout == 30
        assert lock.retry_interval == 0.1
        assert lock.max_retry == 50

    def test_default_values(self):
        """测试默认值"""
        from agentic_rag.lock.redis_lock import RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        lock = RedisDistributedLock(redis_cache=mock_redis_cache)
        
        assert lock.lock_timeout == 30
        assert lock.retry_interval == 0.1
        assert lock.max_retry == 50


class TestDistributedLockContextMocked:
    """分布式锁上下文管理器测试（模拟）"""

    @pytest.mark.asyncio
    async def test_context_initialization(self):
        """测试上下文初始化"""
        from agentic_rag.lock.redis_lock import DistributedLockContext, RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        distributed_lock = RedisDistributedLock(mock_redis_cache)
        
        context = DistributedLockContext(
            distributed_lock=distributed_lock,
            lock_key="test_lock",
            blocking=True
        )
        
        assert context.lock_key == "test_lock"
        assert context.blocking is True
        assert context.lock_value is None

    @pytest.mark.asyncio
    async def test_context_aenter_acquires_lock(self):
        """测试进入上下文获取锁"""
        from agentic_rag.lock.redis_lock import DistributedLockContext, RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        distributed_lock = RedisDistributedLock(mock_redis_cache)
        
        mock_redis_cache.redis_client.set = AsyncMock(return_value="lock_value_123")
        
        context = DistributedLockContext(
            distributed_lock=distributed_lock,
            lock_key="test_lock"
        )
        
        result = await context.__aenter__()
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_context_aexit_releases_lock(self):
        """测试退出上下文释放锁"""
        from agentic_rag.lock.redis_lock import DistributedLockContext, RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        distributed_lock = RedisDistributedLock(mock_redis_cache)
        
        mock_redis_cache.redis_client.eval = AsyncMock(return_value=1)
        
        context = DistributedLockContext(
            distributed_lock=distributed_lock,
            lock_key="test_lock"
        )
        context.lock_value = "test_value"
        
        await context.__aexit__(None, None, None)
        
        mock_redis_cache.redis_client.eval.assert_called()


class TestLockPatterns:
    """锁模式测试（模拟）"""

    @pytest.mark.asyncio
    async def test_non_blocking_acquire(self):
        """测试非阻塞获取锁"""
        from agentic_rag.lock.redis_lock import RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        mock_redis_cache.redis_client.set = AsyncMock(return_value=None)
        
        lock = RedisDistributedLock(mock_redis_cache, max_retry=1)
        result = await lock.acquire("test_key", blocking=False)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_blocking_acquire_success(self):
        """测试阻塞获取锁成功"""
        from agentic_rag.lock.redis_lock import RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        mock_redis_cache.redis_client.set = AsyncMock(return_value="success")
        
        lock = RedisDistributedLock(mock_redis_cache, max_retry=2)
        result = await lock.acquire("test_key", blocking=True)
        
        assert result == "success"

    @pytest.mark.asyncio
    async def test_release_success(self):
        """测试释放锁成功"""
        from agentic_rag.lock.redis_lock import RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        mock_redis_cache.redis_client.eval = AsyncMock(return_value=1)
        
        lock = RedisDistributedLock(mock_redis_cache)
        result = await lock.release("test_key", "test_value")
        
        assert result is True

    @pytest.mark.asyncio
    async def test_release_wrong_value(self):
        """测试释放锁时值不匹配"""
        from agentic_rag.lock.redis_lock import RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        mock_redis_cache.redis_client.eval = AsyncMock(return_value=0)
        
        lock = RedisDistributedLock(mock_redis_cache)
        result = await lock.release("test_key", "wrong_value")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_extend_success(self):
        """测试延长锁超时成功"""
        from agentic_rag.lock.redis_lock import RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        mock_redis_cache.redis_client.eval = AsyncMock(return_value=1)
        
        lock = RedisDistributedLock(mock_redis_cache)
        result = await lock.extend("test_key", "test_value", additional_time=60)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_is_locked_true(self):
        """测试锁被占用"""
        from agentic_rag.lock.redis_lock import RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        mock_redis_cache.redis_client.exists = AsyncMock(return_value=1)
        
        lock = RedisDistributedLock(mock_redis_cache)
        result = await lock.is_locked("test_key")
        
        assert result is True

    @pytest.mark.asyncio
    async def test_is_locked_false(self):
        """测试锁未占用"""
        from agentic_rag.lock.redis_lock import RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        mock_redis_cache.redis_client.exists = AsyncMock(return_value=0)
        
        lock = RedisDistributedLock(mock_redis_cache)
        result = await lock.is_locked("test_key")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_lock_owner(self):
        """测试获取锁持有者"""
        from agentic_rag.lock.redis_lock import RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        mock_redis_cache.redis_client.get = AsyncMock(return_value="owner_value")
        
        lock = RedisDistributedLock(mock_redis_cache)
        result = await lock.get_lock_owner("test_key")
        
        assert result == "owner_value"

    @pytest.mark.asyncio
    async def test_get_lock_owner_not_locked(self):
        """测试获取不存在的锁持有者"""
        from agentic_rag.lock.redis_lock import RedisDistributedLock
        
        mock_redis_cache = MagicMock()
        mock_redis_cache.redis_client.get = AsyncMock(return_value=None)
        
        lock = RedisDistributedLock(mock_redis_cache)
        result = await lock.get_lock_owner("test_key")
        
        assert result is None
