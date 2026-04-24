"""
Redis缓存层(生产级)

缓存策略：
1. LLM响应缓存:相同问题+上下文直接返回缓存
2. 向量检索缓存:相同query的检索结果复用
3. 限流计数器:多实例共享的限流状态
4. 会话状态缓存:热会话Redis,冷会话PostgreSQL

多级缓存架构：
  客户端请求
    → Redis缓存(毫秒级响应)
    → PostgreSQL(百毫秒级查询)
    → 重新生成(秒级LLM调用)
"""
from typing import Optional, Any, Dict, List
from datetime import timedelta
import time
import json
import hashlib
import redis.asyncio as redis
from functools import wraps

from loguru import logger




class RedisCache:
    """Redis缓存管理器(异步)"""
    
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        max_connections: int = 20,  # 最大连接数
        default_ttl: int = 3600,  # 默认1小时,单位秒
        socket_timeout: int = 5,  # 读写超时,单位秒
        socket_connect_timeout: int = 5,  # 连接超时,单位秒
        retry_on_timeout: bool = True  # 超时重试,默认True,超时时重试 1 次
    ):
        """
        初始化Redis连接池
        
        Args:
            url: Redis连接字符串,支持 redis:// 或 rediss://
            max_connections: 最大连接数
            default_ttl: 默认过期时间（秒）
            socket_timeout: Socket超时,单位秒
            socket_connect_timeout: 连接超时,单位秒
            retry_on_timeout: 超时重试,默认True,超时时重试 1 次
        """
        # 创建连接池
        self.redis_pool = redis.ConnectionPool.from_url(
            url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
            decode_responses=True  # 自动解码字符串
        )
        
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)
        self.default_ttl = default_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值,不存在则返回None
        """
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except redis.RedisError as e:
            # Redis故障时降级处理，不影响主流程
            logger.error(f"Redis GET失败: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值(任意可JSON序列化对象)
            ttl: 过期时间(秒),None使用默认值
            
        Returns:
            是否设置成功
        """
        try:
            serialized = json.dumps(value, ensure_ascii=False) # 把任意 Python 对象序列化成 JSON 字符串，方便存入 Redis 这类只支持字符串 / 字节的存储中。
            await self.redis_client.set(
                key,
                serialized,
                ex=ttl or self.default_ttl
            )
            return True
        except redis.RedisError as e:
            logger.error(f"Redis SET失败: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        try:
            await self.redis_client.delete(key)
            return True
        except redis.RedisError as e:
            logger.error(f"Redis DELETE失败: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在
        """
        try:
            return await self.redis_client.exists(key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis EXISTS失败: {e}")
            return False
    
    async def increment(
        self,
        key: str,
        amount: int = 1,
        ttl: Optional[int] = None
    ) -> int:
        """
        原子递增计数器（用于限流）
        
        Args:
            key: 计数器键
            amount: 递增量
            ttl: 过期时间
            
        Returns:
            当前计数值
        """
        try:
            value = await self.redis_client.incr(key, amount)
            # 首次创建时设置过期时间
            if value == amount and ttl:
                await self.redis_client.expire(key, ttl)
            return value
        except redis.RedisError as e:
            logger.error(f"Redis INCR失败: {e}")
            return 0
    
    async def close(self):
        """关闭连接池"""
        if self.redis_pool:
            await self.redis_pool.aclose()
            logger.info("Redis连接池已关闭")
        else:
            logger.warning("Redis连接池未初始化")
    
    @staticmethod
    def generate_cache_key(prefix: str, **kwargs) -> str:
        """
        生成缓存键（基于参数哈希）
        
        Args:
            prefix: 键前缀，如 "llm_response"
            **kwargs: 键值对参数
            
        Returns:
            格式化的缓存键
        """
        # 将参数排序后生成哈希
        sorted_params = sorted(kwargs.items())
        params_str = json.dumps(sorted_params, ensure_ascii=False)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"{prefix}:{params_hash}"



# ==================== 缓存装饰器 ====================

def cache_llm_response(ttl: int = 3600):
    """
    LLM响应缓存装饰器
    
    使用方法：
    @cache_llm_response(ttl=3600)
    async def generate_answer(question, context):
        # LLM调用逻辑
        return answer
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = RedisCache.generate_cache_key(
                "llm_response",
                func_name=func.__name__,
                args=str(args),
                kwargs=str(kwargs)
            )
            
            # 尝试从缓存获取
            cache = RedisCache()
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行实际函数
            result = await func(*args, **kwargs)
            
            # 存入缓存
            await cache.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator


def cache_vector_search(ttl: int = 1800):
    """
    向量检索缓存装饰器
    
    相同query的检索结果直接复用,避免重复向量计算
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = RedisCache.generate_cache_key(
                "vector_search",
                func_name=func.__name__,
                query=kwargs.get("query", "")
            )
            
            cache = RedisCache()
            cached = await cache.get(cache_key)
            if cached:
                return cached
            
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator


# ==================== 限流器 ====================

class RedisRateLimiter:
    """基于Redis的分布式限流器(滑动窗口)"""
    
    def __init__(self, redis_cache: RedisCache):
        self.redis = redis_cache
    
    async def is_allowed(
        self,
        client_id: str,
        max_requests: int = 60,
        window_seconds: int = 60
    ) -> bool:
        """
        检查请求是否允许
        
        Args:
            client_id: 客户端标识(如IP、用户ID)
            max_requests: 窗口内最大请求数
            window_seconds: 窗口大小（秒）
            
        Returns:
            是否允许请求
        """
        # 使用Redis Sorted Set实现滑动窗口
        now = time.time()
        window_start = now - window_seconds
        
        key = f"rate_limit:{client_id}"
        
        # 使用Lua脚本保证原子性
        # Redis 中执行 Lua 脚本时，整个脚本会作为一个原子操作执行，中间不会被其他请求打断，完美解决了并发问题，所以限流、分布式锁这类需要原子性的场景，Lua 脚本是 Redis 里的常用方案
        # 用原生Redis命令:这里有致命问题：不是原子操作
        # 而Redis 执行 Lua 脚本的规则 = 强制原子性

        lua_script = """
        -- 清理过期记录
        redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ARGV[1])
        -- 获取当前窗口内请求数
        local current = redis.call('ZCARD', KEYS[1])
        -- 检查是否超限
        if current < tonumber(ARGV[2]) then
            -- 允许请求，记录时间戳
            redis.call('ZADD', KEYS[1], ARGV[3], ARGV[3])
            -- 设置过期时间
            redis.call('EXPIRE', KEYS[1], ARGV[4])
            return 1
        else
            return 0
        end
        """
        
        result = await self.redis.redis_client.eval(
            lua_script,
            1,
            key,
            str(window_start),
            str(max_requests),
            str(now),
            str(window_seconds)
        )
        
        return result == 1
    
    async def get_remaining(
        self,
        client_id: str,
        max_requests: int = 60,
        window_seconds: int = 60
    ) -> int:
        """
        获取剩余请求次数
        
        Args:
            client_id: 客户端标识
            max_requests: 窗口内最大请求数
            window_seconds: 窗口大小
            
        Returns:
            剩余请求次数
        """
        now = time.time()
        window_start = now - window_seconds
        key = f"rate_limit:{client_id}"
        
        # 清理过期记录
        await self.redis.redis_client.zremrangebyscore(key, 0, window_start)
        
        # 获取当前计数
        current = await self.redis.redis_client.zcard(key)
        remaining = max_requests - current
        
        return max(0, remaining)


# ==================== 缓存统计 ====================

class CacheStats:
    """缓存统计(命中率、缓存大小等)"""
    
    def __init__(self, redis_cache: RedisCache):
        self.redis = redis_cache
    
    async def get_hit_rate(self, prefix: str) -> Dict[str, float]:
        """
        获取缓存命中率(需要外部记录hit/miss)
        缓存命中数(hits)和未命中数(misses) prefix: str:缓存统计的前缀
        这里提供架构示例,实际hit/miss应在业务层记录
        """
        hits = await self.redis.get(f"stats:{prefix}:hits") or 0
        misses = await self.redis.get(f"stats:{prefix}:misses") or 0
        
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0
        
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate
        }
    
    async def get_cache_size(self, prefix: str) -> int:
        """获取指定前缀的缓存键数量"""
        keys = await self.redis.redis_client.keys(f"{prefix}:*")
        return len(keys)
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """获取Redis内存使用情况"""
        # 获取 Redis 内存相关的所有状态数据
        info = await self.redis.redis_client.info("memory")
        return {
            "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
            "peak_memory_mb": info.get("used_memory_peak", 0) / 1024 / 1024,
            "memory_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0)
        }

