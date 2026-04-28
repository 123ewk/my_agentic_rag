"""
缓存模块测试
测试缓存逻辑（不依赖外部库）
"""
import pytest
import time
from collections import OrderedDict
import threading
import hashlib


class MockIntentCache:
    """模拟意图缓存"""

    def __init__(self, max_size=200, ttl_seconds=3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}

    def _generate_key(self, question):
        content = question.strip().lower()
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get(self, question):
        key = self._generate_key(question)
        if key not in self._cache:
            return None
        if time.time() - self._timestamps.get(key, 0) > self.ttl_seconds:
            del self._cache[key]
            del self._timestamps[key]
            return None
        self._cache.move_to_end(key)
        return self._cache[key].get("intent")

    def set(self, question, intent):
        key = self._generate_key(question)
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

    def clear(self):
        self._cache.clear()
        self._timestamps.clear()


class TestIntentCacheLogic:
    """意图缓存逻辑测试"""

    def test_cache_set_and_get(self):
        """测试缓存设置和获取"""
        cache = MockIntentCache(max_size=10, ttl_seconds=3600)

        cache.set("什么是Python", "factual")
        result = cache.get("什么是Python")

        assert result == "factual"

    def test_cache_miss(self):
        """测试缓存未命中"""
        cache = MockIntentCache()

        result = cache.get("不存在的问题")

        assert result is None

    def test_cache_lru_eviction(self):
        """测试LRU淘汰策略"""
        cache = MockIntentCache(max_size=2, ttl_seconds=3600)

        cache.set("问题1", "factual")
        cache.set("问题2", "multi_hop")
        cache.set("问题3", "summary")

        assert cache.get("问题1") is None
        assert cache.get("问题2") == "multi_hop"
        assert cache.get("问题3") == "summary"

    def test_cache_ttl_expiration(self):
        """测试TTL过期"""
        cache = MockIntentCache(max_size=10, ttl_seconds=1)

        cache.set("测试问题", "factual")
        time.sleep(1.1)

        result = cache.get("测试问题")
        assert result is None

    def test_cache_same_question_different_case(self):
        """测试相同问题不同大小写"""
        cache = MockIntentCache()

        cache.set("什么是Python", "factual")
        result1 = cache.get("什么是Python")
        result2 = cache.get("什么是PYTHON")
        result3 = cache.get("什么是 python")

        assert result1 == result2 == result3 == "factual"

    def test_cache_clear(self):
        """测试清空缓存"""
        cache = MockIntentCache()

        cache.set("问题1", "factual")
        cache.set("问题2", "multi_hop")

        cache.clear()

        assert cache.get("问题1") is None
        assert cache.get("问题2") is None


class MockGenerationCache:
    """模拟生成缓存"""

    def __init__(self, max_size=100, ttl_seconds=3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._hits = 0
        self._misses = 0

    def _generate_key(self, question, intent=None):
        content = question.strip().lower()
        if intent:
            content = content + "|" + intent
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get(self, question, intent=None):
        key = self._generate_key(question, intent)
        if key not in self._cache:
            self._misses += 1
            return None
        if time.time() - self._timestamps.get(key, 0) > self.ttl_seconds:
            del self._cache[key]
            del self._timestamps[key]
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return self._cache[key]

    def set(self, question, response, intent=None, metadata=None):
        key = self._generate_key(question, intent)
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
        self._cache[key] = {
            "question": question,
            "response": response,
            "intent": intent,
            "metadata": metadata or {},
            "created_at": time.time()
        }
        self._timestamps[key] = time.time()
        self._cache.move_to_end(key)

    def clear(self):
        self._cache.clear()
        self._timestamps.clear()
        self._hits = 0
        self._misses = 0


class TestGenerationCacheLogic:
    """生成缓存逻辑测试"""

    def test_cache_set_and_get(self):
        """测试缓存设置和获取"""
        cache = MockGenerationCache(max_size=10, ttl_seconds=3600)

        cache.set("什么是Python", "Python是一种编程语言", intent="factual")
        result = cache.get("什么是Python", intent="factual")

        assert result is not None
        assert result["response"] == "Python是一种编程语言"
        assert result["intent"] == "factual"

    def test_cache_miss_without_intent(self):
        """测试无意图时的缓存未命中"""
        cache = MockGenerationCache()

        result = cache.get("不存在的问题")

        assert result is None

    def test_cache_miss_with_different_intent(self):
        """测试不同意图的缓存未命中"""
        cache = MockGenerationCache()

        cache.set("测试问题", "回答1", intent="factual")
        result = cache.get("测试问题", intent="multi_hop")

        assert result is None

    def test_cache_hit_rate_tracking(self):
        """测试命中率跟踪"""
        cache = MockGenerationCache(max_size=10, ttl_seconds=3600)

        cache.set("问题1", "回答1", intent="factual")
        cache.set("问题2", "回答2", intent="multi_hop")

        cache.get("问题1")
        cache.get("不存在")
        cache.get("问题2")

        assert cache._hits == 2
        assert cache._misses == 1

    def test_cache_lru_eviction(self):
        """测试LRU淘汰策略"""
        cache = MockGenerationCache(max_size=2, ttl_seconds=3600)

        cache.set("问题1", "回答1", intent="factual")
        cache.set("问题2", "回答2", intent="multi_hop")
        cache.set("问题3", "回答3", intent="summary")

        assert cache.get("问题1", intent="factual") is None
        assert cache.get("问题2", intent="multi_hop") is not None
        assert cache.get("问题3", intent="summary") is not None

    def test_cache_with_metadata(self):
        """测试带元数据的缓存"""
        cache = MockGenerationCache()

        metadata = {"sources": ["doc1", "doc2"], "confidence": 0.9}
        cache.set("测试问题", "测试回答", intent="factual", metadata=metadata)

        result = cache.get("测试问题", intent="factual")

        assert result is not None
        assert result["metadata"]["sources"] == ["doc1", "doc2"]
        assert result["metadata"]["confidence"] == 0.9

    def test_cache_clear_resets_stats(self):
        """测试清空缓存重置统计"""
        cache = MockGenerationCache()

        cache.set("问题1", "回答1", intent="factual")
        cache.get("问题1")
        cache.get("不存在")

        cache.clear()

        assert len(cache._cache) == 0
        assert cache._hits == 0
        assert cache._misses == 0


class TestCacheKeyGeneration:
    """缓存键生成测试"""

    def test_key_generation_consistency(self):
        """测试键生成一致性"""
        def generate_key(text):
            content = text.strip().lower()
            return hashlib.md5(content.encode('utf-8')).hexdigest()

        key1 = generate_key("测试")
        key2 = generate_key("测试")
        key3 = generate_key("TEST")

        assert key1 == key2
        assert key1 != key3

    def test_key_with_intent(self):
        """测试带意图的键生成"""
        def generate_key(question, intent=None):
            content = question.strip().lower()
            if intent:
                content = content + "|" + intent
            return hashlib.md5(content.encode('utf-8')).hexdigest()

        key1 = generate_key("测试", "factual")
        key2 = generate_key("测试", "multi_hop")
        key3 = generate_key("测试", "factual")

        assert key1 != key2
        assert key1 == key3


class TestCacheStats:
    """缓存统计测试"""

    def test_stats_structure(self):
        """测试统计数据结构"""
        stats = {
            "size": 5,
            "max_size": 100,
            "ttl_seconds": 3600,
            "hits": 10,
            "misses": 2,
            "hit_rate": 10 / 12
        }

        assert stats["size"] == 5
        assert stats["max_size"] == 100
        assert "hit_rate" in stats

    def test_hit_rate_calculation(self):
        """测试命中率计算"""
        hits = 80
        misses = 20
        total = hits + misses

        hit_rate = hits / total if total > 0 else 0.0

        assert hit_rate == 0.8
