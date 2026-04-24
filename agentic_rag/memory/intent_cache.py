"""
意图识别缓存模块
用于缓存意图识别结果，减少重复的LLM调用
"""
from typing import Optional, Dict, Any
from collections import OrderedDict
import threading
import time
import hashlib
from loguru import logger


class IntentCache:
    """
    意图识别缓存，使用LRU策略管理缓存，支持可选的Redis持久化
    """

    def __init__(
        self,
        max_size: int = 200,
        ttl_seconds: int = 1800,
        redis_cache=None,
        redis_enabled: bool = False
    ):
        """
        初始化缓存

        参数:
            max_size: 内存缓存最大容量
            ttl_seconds: 缓存过期时间（秒）
            redis_cache: Redis缓存实例（可选）
            redis_enabled: 是否启用Redis持久化（默认关闭）
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._redis = redis_cache
        self._redis_enabled = redis_enabled

    def _generate_key(self, question: str) -> str:
        """
        生成缓存键

        参数:
            question: 用户问题

        返回:
            缓存键的哈希值
        """
        content = question.strip().lower()
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    async def _get_from_redis(self, key: str) -> Optional[str]:
        """从Redis获取缓存"""
        if not self._redis_enabled or not self._redis:
            return None
        try:
            result = await self._redis.get(f"intent_cache:{key}")
            return result
        except Exception as e:
            logger.warning(f"Redis缓存获取失败: {e}")
            return None

    async def _set_to_redis(self, key: str, intent: str):
        """设置缓存到Redis"""
        if not self._redis_enabled or not self._redis:
            return
        try:
            await self._redis.set(f"intent_cache:{key}", intent, ttl=self.ttl_seconds)
        except Exception as e:
            logger.warning(f"Redis缓存设置失败: {e}")

    def get(self, question: str) -> Optional[str]:
        """
        获取缓存的意图识别结果

        参数:
            question: 用户问题

        返回:
            缓存的意图类型，如果没有缓存则返回None
        """
        key = self._generate_key(question)

        with self._lock:
            if key not in self._cache:
                return None

            if time.time() - self._timestamps.get(key, 0) > self.ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                return None

            self._cache.move_to_end(key)
            logger.debug(f"意图缓存命中: {question[:30]}...")
            return self._cache[key].get("intent")

    def set(self, question: str, intent: str):
        """
        设置缓存

        参数:
            question: 用户问题
            intent: 识别的意图类型
        """
        key = self._generate_key(question)

        with self._lock:
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            self._cache[key] = {
                "question": question,
                "intent": intent,
                "created_at": time.time()
            }
            self._timestamps[key] = time.time()
            self._cache.move_to_end(key)

            logger.debug(f"意图缓存已保存: {question[:30]}...")

    async def get_async(self, question: str) -> Optional[str]:
        """异步获取缓存，优先查内存，再查Redis"""
        intent = self.get(question)
        if intent:
            return intent

        key = self._generate_key(question)
        redis_intent = await self._get_from_redis(key)
        if redis_intent:
            self.set(question, redis_intent)
            return redis_intent
        return None

    async def set_async(self, question: str, intent: str):
        """异步设置缓存，同时写内存和Redis"""
        self.set(question, intent)
        await self._set_to_redis(self._generate_key(question), intent)

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "redis_enabled": self._redis_enabled
            }


# 全局意图缓存实例
_intent_cache: Optional[IntentCache] = None


def get_intent_cache(
    max_size: int = 200,
    ttl_seconds: int = 1800,
    redis_cache=None,
    redis_enabled: bool = False
) -> IntentCache:
    """
    获取全局意图缓存实例（单例模式）

    参数:
        max_size: 缓存最大容量
        ttl_seconds: 缓存过期时间
        redis_cache: Redis缓存实例
        redis_enabled: 是否启用Redis持久化

    返回:
        IntentCache实例
    """
    global _intent_cache
    if _intent_cache is None:
        _intent_cache = IntentCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            redis_cache=redis_cache,
            redis_enabled=redis_enabled
        )
    return _intent_cache


def reset_intent_cache():
    """重置全局缓存实例（用于测试）"""
    global _intent_cache
    _intent_cache = None
