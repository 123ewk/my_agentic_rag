"""
生成结果缓存模块
用于缓存完整问答结果，减少重复的LLM调用和检索开销
"""
from typing import Optional, Dict, Any
from collections import OrderedDict
import threading
import time
import hashlib
from loguru import logger


class GenerationCache:
    """
    生成结果缓存，使用LRU策略管理缓存，支持可选的Redis持久化

    缓存键基于：问题文本 + 意图类型（相同问题+意图返回相同答案）
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 3600,
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
            hits: 缓存命中次数（默认0）
            misses: 缓存未命中次数（默认0）
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._redis = redis_cache
        self._redis_enabled = redis_enabled
        self._hits = 0
        self._misses = 0

    def _generate_key(self, question: str, intent: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """
        生成缓存键

        设计原则：相同问题 + 相同意图 = 相同答案
        - question: 问题文本（归一化后MD5）
        - intent: 意图类型（同一意图下的改写查询应命中缓存）
        - session_id: 可选，用于区分不同会话的相同问题

        返回:
            缓存键的哈希值
        """
        content = question.strip().lower()
        if intent:
            content += f"|{intent}"
        # 注意：session_id 不加入缓存键，因为同一问题在不同session应有相同答案
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    async def _get_from_redis(self, key: str) -> Optional[Dict]:
        """从Redis获取缓存"""
        if not self._redis_enabled or not self._redis:
            return None
        try:
            result = await self._redis.get(f"gen_cache:{key}")
            return result
        except Exception as e:
            logger.warning(f"Redis生成缓存获取失败: {e}")
            return None

    async def _set_to_redis(self, key: str, data: Dict):
        """设置缓存到Redis"""
        if not self._redis_enabled or not self._redis:
            return
        try:
            await self._redis.set(f"gen_cache:{key}", data, ttl=self.ttl_seconds)
        except Exception as e:
            logger.warning(f"Redis生成缓存设置失败: {e}")

    def get(self, question: str, intent: Optional[str] = None) -> Optional[Dict]:
        """
        获取缓存的生成结果

        缓存查找顺序（优先返回最早命中的缓存）：
        1. question + intent 作为键
        2. question 作为键（intent=None 的老缓存兼容）

        参数:
            question: 用户问题
            intent: 意图类型

        返回:
            包含答案和元数据的字典，如果没有缓存则返回None
        """
        # 优先查找精确匹配（question + intent）
        key = self._generate_key(question, intent)
        if key in self._cache and self._is_valid(key):
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug(f"生成缓存命中(精确): {question[:30]}...")
            return self._cache[key]

        # 回退查找：question only（兼容老缓存或规则命中的意图）
        key_question_only = self._generate_key(question, None)
        if key_question_only in self._cache and self._is_valid(key_question_only):
            self._cache.move_to_end(key_question_only)
            self._hits += 1
            logger.debug(f"生成缓存命中(回退): {question[:30]}...")
            return self._cache[key_question_only]

        self._misses += 1
        return None

    def _is_valid(self, key: str) -> bool:
        """检查缓存是否有效（未过期）"""
        if key not in self._timestamps:
            return False
        return time.time() - self._timestamps.get(key, 0) <= self.ttl_seconds

    def set(self, question: str, response: str, intent: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        设置缓存

        优化策略：
        - 同一问题在不同意图下产生不同答案，但缓存键不包含 session_id
        - 缓存写入时会同时用 (question, intent) 和 (question, None) 两种键，
          以提高回退查找时的命中率

        参数:
            question: 用户问题
            response: LLM生成的响应
            intent: 意图类型
            metadata: 其他元数据（如检索文档摘要等）
        """
        # 同时用两种键写入，确保回退查找能命中
        keys_to_write = []
        if intent:
            keys_to_write.append(self._generate_key(question, intent))
        # 也写入 question only 的键（兼容老缓存）
        keys_to_write.append(self._generate_key(question, None))

        with self._lock:
            # LRU淘汰：如果缓存满了且要写入新键，删除最老的
            if len(self._cache) >= self.max_size:
                # 检查要写入的键是否已存在
                existing_keys = set(k for k in keys_to_write if k in self._cache)
                if not existing_keys:
                    # 需要淘汰：删除最老的键
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    del self._timestamps[oldest_key]
                    logger.debug(f"缓存淘汰(空间不足): {oldest_key[:16]}...")

            # 写入所有键（相同数据）
            for key in keys_to_write:
                self._cache[key] = {
                    "question": question,
                    "response": response,
                    "intent": intent,
                    "metadata": metadata or {},
                    "created_at": time.time()
                }
                self._timestamps[key] = time.time()
                self._cache.move_to_end(key)

            logger.debug(f"生成缓存已保存: {question[:30]}... (写入{len(keys_to_write)}个键)")

    async def get_async(self, question: str, intent: Optional[str] = None) -> Optional[Dict]:
        """异步获取缓存，优先查内存，再查Redis"""
        result = self.get(question, intent)
        if result:
            return result

        key = self._generate_key(question, intent)
        redis_result = await self._get_from_redis(key)
        if redis_result:
            self.set(question, redis_result.get("response", ""), intent, redis_result.get("metadata"))
            return redis_result
        return None

    async def set_async(self, question: str, response: str, intent: Optional[str] = None, metadata: Optional[Dict] = None):
        """异步设置缓存，同时写内存和Redis"""
        self.set(question, response, intent, metadata)
        await self._set_to_redis(self._generate_key(question, intent), {
            "response": response,
            "intent": intent,
            "metadata": metadata or {}
        })

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "redis_enabled": self._redis_enabled
            }


# 全局生成缓存实例
_gen_cache: Optional[GenerationCache] = None


def get_generation_cache(
    max_size: int = 100,
    ttl_seconds: int = 3600,
    redis_cache=None,
    redis_enabled: bool = False
) -> GenerationCache:
    """
    获取全局生成缓存实例（单例模式）

    参数:
        max_size: 缓存最大容量
        ttl_seconds: 缓存过期时间
        redis_cache: Redis缓存实例
        redis_enabled: 是否启用Redis持久化

    返回:
        GenerationCache实例
    """
    global _gen_cache
    if _gen_cache is None:
        _gen_cache = GenerationCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            redis_cache=redis_cache,
            redis_enabled=redis_enabled
        )
    return _gen_cache


def reset_generation_cache():
    """重置全局缓存实例（用于测试）"""
    global _gen_cache
    _gen_cache = None
