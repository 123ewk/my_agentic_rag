"""
缓存模块测试
测试意图缓存、生成缓存、LLM缓存等功能
"""
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from agentic_rag.memory.intent_cache import (
    IntentCache,
    get_intent_cache,
    reset_intent_cache
)
from agentic_rag.memory.gen_cache import (
    GenerationCache,
    get_generation_cache,
    reset_generation_cache
)
from agentic_rag.memory.cache import (
    RedisCache,
    RedisRateLimiter,
    CacheStats
)


class TestIntentCache:
    """意图缓存测试"""

    def setup_method(self):
        """每个测试前重置全局缓存"""
        reset_intent_cache()

    def teardown_method(self):
        """每个测试后清理"""
        reset_intent_cache()

    def test_cache_set_and_get(self):
        """测试缓存设置和获取"""
        cache = IntentCache(max_size=10, ttl_seconds=3600)
        
        cache.set("什么是Python", "factual")
        result = cache.get("什么是Python")
        
        assert result == "factual"

    def test_cache_miss(self):
        """测试缓存未命中"""
        cache = IntentCache()
        
        result = cache.get("不存在的问题")
        
        assert result is None

    def test_cache_lru_eviction(self):
        """测试LRU淘汰策略"""
        cache = IntentCache(max_size=2, ttl_seconds=3600)
        
        cache.set("问题1", "factual")
        cache.set("问题2", "multi_hop")
        cache.set("问题3", "summary")
        
        assert cache.get("问题1") is None
        assert cache.get("问题2") == "multi_hop"
        assert cache.get("问题3") == "summary"

    def test_cache_ttl_expiration(self):
        """测试TTL过期"""
        cache = IntentCache(max_size=10, ttl_seconds=1)
        
        cache.set("测试问题", "factual")
        time.sleep(1.1)
        
        result = cache.get("测试问题")
        assert result is None

    def test_cache_same_question_different_case(self):
        """测试相同问题不同大小写"""
        cache = IntentCache()
        
        cache.set("什么是Python", "factual")
        result1 = cache.get("什么是Python")
        result2 = cache.get("什么是PYTHON")
        result3 = cache.get("什么是 python")
        
        assert result1 == result2 == result3 == "factual"

    def test_cache_clear(self):
        """测试清空缓存"""
        cache = IntentCache()
        
        cache.set("问题1", "factual")
        cache.set("问题2", "multi_hop")
        
        cache.clear()
        
        assert cache.get("问题1") is None
        assert cache.get("问题2") is None
        assert len(cache.get_stats()["size"]) == 0

    def test_cache_stats(self):
        """测试缓存统计"""
        cache = IntentCache(max_size=100, ttl_seconds=3600)
        
        cache.set("问题1", "factual")
        cache.set("问题2", "multi_hop")
        
        stats = cache.get_stats()
        
        assert stats["size"] == 2
        assert stats["max_size"] == 100
        assert stats["ttl_seconds"] == 3600

    def test_cache_order_maintained(self):
        """测试访问顺序维护"""
        cache = IntentCache(max_size=3, ttl_seconds=3600)
        
        cache.set("问题1", "factual")
        cache.set("问题2", "multi_hop")
        cache.set("问题3", "summary")
        
        cache.get("问题1")
        
        cache.set("问题4", "reasoning")
        
        assert cache.get("问题1") == "factual"
        assert cache.get("问题2") is None


class TestGenerationCache:
    """生成缓存测试"""

    def setup_method(self):
        """每个测试前重置全局缓存"""
        reset_generation_cache()

    def teardown_method(self):
        """每个测试后清理"""
        reset_generation_cache()

    def test_cache_set_and_get(self):
        """测试缓存设置和获取"""
        cache = GenerationCache(max_size=10, ttl_seconds=3600)
        
        cache.set("什么是Python", "Python是一种编程语言", intent="factual")
        result = cache.get("什么是Python", intent="factual")
        
        assert result is not None
        assert result["response"] == "Python是一种编程语言"
        assert result["intent"] == "factual"

    def test_cache_miss_without_intent(self):
        """测试无意图时的缓存未命中"""
        cache = GenerationCache()
        
        result = cache.get("不存在的问题")
        
        assert result is None

    def test_cache_miss_with_different_intent(self):
        """测试不同意图的缓存未命中"""
        cache = GenerationCache()
        
        cache.set("测试问题", "回答1", intent="factual")
        result = cache.get("测试问题", intent="multi_hop")
        
        assert result is None

    def test_cache_hit_rate_tracking(self):
        """测试命中率跟踪"""
        cache = GenerationCache(max_size=10, ttl_seconds=3600)
        
        cache.set("问题1", "回答1", intent="factual")
        cache.set("问题2", "回答2", intent="multi_hop")
        
        cache.get("问题1")
        cache.get("不存在")
        cache.get("问题2")
        
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    def test_cache_lru_eviction(self):
        """测试LRU淘汰策略"""
        cache = GenerationCache(max_size=2, ttl_seconds=3600)
        
        cache.set("问题1", "回答1", intent="factual")
        cache.set("问题2", "回答2", intent="multi_hop")
        cache.set("问题3", "回答3", intent="summary")
        
        assert cache.get("问题1", intent="factual") is None
        assert cache.get("问题2", intent="multi_hop") is not None
        assert cache.get("问题3", intent="summary") is not None

    def test_cache_with_metadata(self):
        """测试带元数据的缓存"""
        cache = GenerationCache()
        
        metadata = {"sources": ["doc1", "doc2"], "confidence": 0.9}
        cache.set("测试问题", "测试回答", intent="factual", metadata=metadata)
        
        result = cache.get("测试问题", intent="factual")
        
        assert result is not None
        assert result["metadata"]["sources"] == ["doc1", "doc2"]
        assert result["metadata"]["confidence"] == 0.9

    def test_cache_clear_resets_stats(self):
        """测试清空缓存重置统计"""
        cache = GenerationCache()
        
        cache.set("问题1", "回答1", intent="factual")
        cache.get("问题1")
        cache.get("不存在")
        
        cache.clear()
        
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_singleton_pattern(self):
        """测试单例模式"""
        cache1 = get_generation_cache(max_size=50, ttl_seconds=1800)
        cache2 = get_generation_cache(max_size=100, ttl_seconds=3600)
        
        assert cache1 is cache2


class TestRedisCacheMocked:
    """Redis缓存测试（模拟）"""

    @pytest.mark.asyncio
    async def test_generate_cache_key(self):
        """测试缓存键生成"""
        key1 = RedisCache.generate_cache_key("test", param1="value1", param2="value2")
        key2 = RedisCache.generate_cache_key("test", param2="value2", param1="value1")
        
        assert key1 == key2
        assert key1.startswith("test:")

    @pytest.mark.asyncio
    async def test_generate_cache_key_different_params(self):
        """测试不同参数生成不同键"""
        key1 = RedisCache.generate_cache_key("test", param1="value1")
        key2 = RedisCache.generate_cache_key("test", param1="value2")
        
        assert key1 != key2


class TestRedisRateLimiterMocked:
    """限流器测试（模拟）"""

    @pytest.mark.asyncio
    async def test_rate_limiter_structure(self):
        """测试限流器结构"""
        mock_redis_cache = MagicMock()
        limiter = RedisRateLimiter(mock_redis_cache)
        
        assert limiter.redis is not None


class TestCacheStatsMocked:
    """缓存统计测试（模拟）"""

    @pytest.mark.asyncio
    async def test_cache_stats_structure(self):
        """测试缓存统计结构"""
        mock_redis_cache = MagicMock()
        stats = CacheStats(mock_redis_cache)
        
        assert stats.redis is not None
