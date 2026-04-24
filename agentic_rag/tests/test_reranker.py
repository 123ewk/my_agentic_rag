"""
重排器测试
测试SimpleReranker等重排功能
"""
import pytest
from langchain_core.documents import Document

from agentic_rag.retrieval.reranker import (
    SimpleReranker,
    get_reranker
)


class TestSimpleReranker:
    """简单重排器测试"""

    def test_rerank_basic(self):
        """测试基础重排功能"""
        reranker = SimpleReranker()
        
        docs = [
            Document(page_content="Python是一种编程语言", metadata={"id": "1"}),
            Document(page_content="Java是一种编程语言", metadata={"id": "2"}),
            Document(page_content="Python机器学习", metadata={"id": "3"}),
        ]
        
        result = reranker.rerank("Python", docs, top_k=2)
        
        assert len(result) <= 2
        assert all(isinstance(item, tuple) for item in result)
        assert all(isinstance(item[0], Document) for item in result)

    def test_rerank_with_scores(self):
        """测试带分数的重排"""
        reranker = SimpleReranker()
        
        docs = [
            Document(page_content="Python编程", metadata={"id": "1"}),
            Document(page_content="Java编程", metadata={"id": "2"}),
        ]
        
        result = reranker.rerank("Python", docs, top_k=2)
        
        for doc, score in result:
            assert 0 <= score <= 1

    def test_rerank_empty_docs(self):
        """测试空文档列表"""
        reranker = SimpleReranker()
        
        result = reranker.rerank("测试", [], top_k=3)
        
        assert result == []

    def test_rerank_top_k_larger_than_docs(self):
        """测试top_k大于文档数量"""
        reranker = SimpleReranker()
        
        docs = [
            Document(page_content="文档1", metadata={"id": "1"}),
        ]
        
        result = reranker.rerank("测试", docs, top_k=10)
        
        assert len(result) <= len(docs)

    def test_rerank_keyword_overlap(self):
        """测试关键词重叠计算"""
        reranker = SimpleReranker()
        
        docs = [
            Document(page_content="Python编程语言非常流行", metadata={"id": "1"}),
            Document(page_content="JavaScript用于网页开发", metadata={"id": "2"}),
        ]
        
        result = reranker.rerank("Python编程", docs, top_k=2)
        
        assert len(result) == 2
        doc1_score = next(score for doc, score in result if doc.metadata["id"] == "1")
        doc2_score = next(score for doc, score in result if doc.metadata["id"] == "2")
        assert doc1_score >= doc2_score


class TestGetReranker:
    """重排器工厂函数测试"""

    def test_get_simple_reranker(self):
        """测试获取简单重排器"""
        reranker = get_reranker("simple")
        
        assert isinstance(reranker, SimpleReranker)

    def test_get_unknown_type(self):
        """测试获取未知类型重排器"""
        with pytest.raises(ValueError) as exc_info:
            get_reranker("unknown")
        
        assert "未知的重排序类型" in str(exc_info.value)

    def test_normalize_score(self):
        """测试分数归一化"""
        from agentic_rag.retrieval.reranker import BGGReranker
        
        reranker = BGGReranker(api_key="test", base_url="http://test.com")
        
        positive_score = BGGReranker.normalize_score(5.0)
        assert 0 < positive_score < 1
        
        zero_score = BGGReranker.normalize_score(0.0)
        assert 0 < zero_score < 1
        
        negative_score = BGGReranker.normalize_score(-5.0)
        assert 0 < negative_score < 1