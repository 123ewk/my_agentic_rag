from loguru import logger

# ==================== 分布式锁 ====================

class RedisDistributedLock:
    """
    Redis分布式锁(基于SET NX EX)
    
    使用场景：
    1. 文档索引去重:防止多实例同时索引同一文档
    2. 缓存更新保护:防止缓存击穿时多个请求同时回源
    3. 定时任务互斥:防止多个调度器实例重复执行
    4. 会话状态保护:防止并发修改导致数据不一致
    
    特性：
    - 支持超时自动释放(防止死锁)
    - 支持可重入(同一线程可多次获取)
    - 支持看门狗续期(长任务自动延长锁超时)
    """
    
    def __init__(
        self,
        redis_cache: RedisCache,
        lock_timeout: int = 30,  # 锁超时时间（秒）
        retry_interval: float = 0.1,  # 重试间隔
        max_retry: int = 50  # 最大重试次数
    ):
        """
        初始化分布式锁
        
        Args:
            redis_cache: Redis缓存实例
            lock_timeout: 锁自动超时时间（防止死锁）
            retry_interval: 获取锁失败时的重试间隔
            max_retry: 最大重试次数
        """
        self.redis = redis_cache
        self.lock_timeout = lock_timeout
        self.retry_interval = retry_interval
        self.max_retry = max_retry
    
    async def acquire(
        self,
        lock_key: str,
        lock_value: Optional[str] = None,
        blocking: bool = True
    ) -> Optional[str]:
        """
        获取/设置分布式锁
        
        Args:
            lock_key: 锁的键名
            lock_value: 锁的值(用于释放时验证,默认生成UUID)
            blocking: 是否阻塞等待
            
        Returns:
            锁的标识值(用于后续释放),获取失败返回None
        """
        import uuid
        try:
        
            lock_value = lock_value or str(uuid.uuid4())
            attempts = 0
            
            while True:
                # SET key value NX EX timeout （原子操作）
                # NX: 仅当key不存在时设置
                # EX: 设置过期时间
                result = await self.redis.redis_client.set(
                    f"lock:{lock_key}",
                    lock_value,
                    nx=True,
                    ex=self.lock_timeout
                )
                
                if result:
                    return lock_value
                
                if not blocking:
                    return None
                
                attempts += 1
                if attempts >= self.max_retry:
                    return None
                
                await asyncio.sleep(self.retry_interval)
                logger.info(f"获取锁失败 {lock_key} with value {lock_value}, 尝试 {attempts} 次")
        except Exception as e:
            logger.error(f"获取锁失败 {lock_key} with value {lock_value}: {e}")
            return None
    
    async def release(self, lock_key: str, lock_value: str) -> bool:
        """
        释放分布式锁(Lua脚本保证原子性)
        
        注意:必须传入获取锁时返回的lock_value,防止误释放其他客户端的锁
        
        Args:
            lock_key: 锁的键名
            lock_value: 锁的标识值(acquire返回的值)
            
        Returns:
            是否释放成功
        """
        # Lua脚本：先检查值是否匹配，再删除（原子操作）
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        try:
        
            result = await self.redis.redis_client.eval(
                lua_script,
                1,
                f"lock:{lock_key}",
                lock_value
            )
            
            return result == 1

        except Exception as e:
            logger.error(f"释放锁失败 {lock_key} with value {lock_value}: {e}")
            return False
    
    async def extend(self, lock_key: str, lock_value: str, additional_time: int = None) -> bool:
        """
        延长锁的超时时间（看门狗续期）
        
        Args:
            lock_key: 锁的键名
            lock_value: 锁的标识值
            additional_time: 延长时间(秒),默认使用lock_timeout
            
        Returns:
            是否续期成功
        """
        try:
        
            additional_time = additional_time or self.lock_timeout
            
            # Lua脚本：检查值是否匹配，匹配则更新过期时间
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("expire", KEYS[1], ARGV[2])
            else
                return 0
            end
            """
            
            result = await self.redis.redis_client.eval(
                lua_script,
                1,
                f"lock:{lock_key}",
                lock_value,
                str(additional_time)
            )
            
            return result == 1
            
        except Exception as e:
            logger.error(f"续期锁失败 {lock_key} with value {lock_value}: {e}")
            return False
    
    async def is_locked(self, lock_key: str) -> bool:
        """
        检查锁是否被占用
        
        Args:
            lock_key: 锁的键名
            
        Returns:
            是否被锁定
        """
        try:
            return await self.redis.redis_client.exists(f"lock:{lock_key}") > 0
        except Exception as e:
            logger.error(f"检查锁是否被占用失败 {lock_key}: {e}")
            return False
    
    async def get_lock_owner(self, lock_key: str) -> Optional[str]:
        """
        获取锁的持有者
        
        Args:
            lock_key: 锁的键名
            
        Returns:
            锁的标识值,未锁定返回None
        """
        try:
            return await self.redis.redis_client.get(f"lock:{lock_key}")
        except Exception as e:
            logger.error(f"获取锁持有者失败 {lock_key}: {e}")
            return None



# 上下文管理器：方便使用 with 语句
class DistributedLockContext:
    """
    分布式锁上下文管理器
    
    使用方式：
    lock = RedisDistributedLock(redis_cache)
    async with DistributedLockContext(lock, "document_index:doc_123") as lock_value:
        if lock_value:
            # 获取到锁，执行任务
            pass
        else:
            # 未获取到锁，跳过
            pass
    """
    
    def __init__(
        self,
        distributed_lock: RedisDistributedLock,  # 分布式锁实例
        lock_key: str, 
        blocking: bool = True  # 是否阻塞等待锁
    ):
        self.lock = distributed_lock
        self.lock_key = lock_key
        self.blocking = blocking
        self.lock_value = None
    
    async def __aenter__(self) -> Optional[str]:
        """
        进入上下文管理器，尝试获取锁
        
        Returns:
            锁的标识值,获取成功返回,否则返回None
        """

        self.lock_value = await self.lock.acquire(
            lock_key=self.lock_key,
            blocking=self.blocking
        )
        return self.lock_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文管理器，尝试释放锁
        
        Args:
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常跟踪信息（可选）
            
        Returns:
            是否释放成功
        """
        if self.lock_value:
            await self.lock.release(
                lock_key=self.lock_key,
                lock_value=self.lock_value
            )
        return True




if __name__ == "__main__":
# ==================== 使用示例 ====================

    async def example_document_indexing_with_lock():
        """示例：文档索引时的分布式锁使用"""
        
        # 初始化
        redis_cache = RedisCache()
        doc_lock = RedisDistributedLock(redis_cache, lock_timeout=120)  # 索引可能需要较长时间
        
        document_id = "doc_20240101_001"
        lock_key = f"document_index:{document_id}"
        
        # 方式1：使用上下文管理器
        async with DistributedLockContext(doc_lock, lock_key) as lock_value:
            if lock_value:
                # 获取到锁，执行索引
                print(f"开始索引文档: {document_id}")
                # ... 索引逻辑 ...
                print(f"文档索引完成: {document_id}")
            else:
                print(f"文档正在被其他实例索引，跳过: {document_id}")
        
        # 方式2：手动获取和释放
        lock_value = await doc_lock.acquire(lock_key, blocking=False)
        if lock_value:
            try:
                # ... 索引逻辑 ...
                pass
            finally:
                await doc_lock.release(lock_key, lock_value)
        else:
            print("获取锁失败")
        
        await redis_cache.close()


    async def example_cache_update_with_lock():
        """示例：缓存更新防击穿"""
        
        redis_cache = RedisCache()
        cache_lock = RedisDistributedLock(redis_cache, lock_timeout=10)
        
        query = "人工智能是什么"
        cache_key = f"vector_search:{hash(query)}"
        lock_key = f"cache_update:{cache_key}"
        
        # 1. 先查缓存
        cached = await redis_cache.get(cache_key)
        if cached:
            return cached
        
        # 2. 缓存未命中，获取锁防止多个请求同时回源
        lock_value = await cache_lock.acquire(lock_key, blocking=False)
        
        if lock_value:
            try:
                # 双重检查：可能其他请求已经更新了缓存
                cached = await redis_cache.get(cache_key)
                if cached:
                    return cached
                
                # 执行实际查询（向量检索、LLM调用等）
                result = await expensive_query(query)
                
                # 存入缓存
                await redis_cache.set(cache_key, result, ttl=3600)
                
                return result
            finally:
                await cache_lock.release(lock_key, lock_value)
        else:
            # 未获取到锁，等待并重试
            await asyncio.sleep(0.5)
            return await redis_cache.get(cache_key)