"""
LLM响应缓存模块
用于缓存相似问题的LLM响应,减少重复API调用
"""
from typing import Optional, Dict, Any, List
import hashlib
import time
from collections import OrderedDict
import threading
from loguru import logger


class LLMCache:
    """
    LLM响应缓存
    
    使用LRU(最近最少使用)策略管理缓存,
    支持基于问题哈希的精确匹配和相似度匹配
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        初始化缓存
        
        参数:
            max_size: 缓存最大容量
            ttl_seconds: 缓存过期时间(秒)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, question: str, context_hash: Optional[str] = None) -> str:
        """
        生成缓存键
        
        参数:
            question: 用户问题
            context_hash: 上下文哈希(可选)
        
        返回:
            缓存键的哈希值
        """
        content = question.strip().lower()
        if context_hash:
            content += f"|{context_hash}"
        
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:32]
    
    def get(self, question: str, context_hash: Optional[str] = None) -> Optional[str]:
        """
        获取缓存的响应
        
        参数:
            question: 用户问题
            context_hash: 上下文哈希
        
        返回:
            缓存的响应内容,如果没有缓存则返回None
        """
        key = self._generate_key(question, context_hash)
        
        with self._lock:
            if key not in self._cache:
                return None
            
            # 检查是否过期
            if time.time() - self._timestamps.get(key, 0) > self.ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                return None
            
            # 移到末尾(LRU策略)
            self._cache.move_to_end(key)
            
            cache_entry = self._cache[key]
            logger.debug(f"缓存命中: {question[:50]}...")
            return cache_entry.get("response")
    
    def set(self, question: str, response: str, context_hash: Optional[str] = None):
        """
        设置缓存
        
        参数:
            question: 用户问题
            response: LLM响应
            context_hash: 上下文哈希
        """
        key = self._generate_key(question, context_hash)
        
        with self._lock:
            # 如果缓存已满,删除最旧的项
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = {
                "question": question,
                "response": response,
                "context_hash": context_hash,
                "created_at": time.time()
            }
            self._timestamps[key] = time.time()
            self._cache.move_to_end(key)
            
            logger.debug(f"缓存已保存: {question[:50]}...")
    
    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        返回:
            包含缓存大小、命中率等信息的字典
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds
            }


# 全局缓存实例
llm_cache = LLMCache(max_size=100, ttl_seconds=3600)


def get_llm_cache() -> LLMCache:
    """
    获取全局LLM缓存实例
    
    返回:
        LLMCache实例
    """
    return llm_cache
