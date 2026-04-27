"""
文档处理模块测试
测试文档加载和分块功能
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from agentic_rag.document_processing.splitters import (
    TextSplitter,
    RecursiveChunker,
    SemanticChunker,
    AdaptiveChunker,
    get_splitter
)


class TestRecursiveChunker:
    """递归字符分块器测试"""

    def test_basic_split(self):
        """测试基础分块功能"""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        
        docs = [
            Document(
                page_content="这是一个测试文档。\n\n它包含多个段落。\n\n用于测试分块功能。",
                metadata={"source": "test"}
            )
        ]
        
        result = chunker.split_documents(docs)
        
        assert len(result) > 0
        assert all(isinstance(doc, Document) for doc in result)

    def test_chunk_size_respected(self):
        """测试分块大小限制"""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        
        docs = [
            Document(
                page_content="这是一个很长的文档内容。" * 20,
                metadata={"source": "test"}
            )
        ]
        
        result = chunker.split_documents(docs)
        
        assert len(result) > 1
        for doc in result:
            assert len(doc.page_content) <= 50 + 10  # 允许一定的重叠

    def test_empty_documents(self):
        """测试空文档列表"""
        chunker = RecursiveChunker()
        
        result = chunker.split_documents([])
        
        assert result == []

    def test_custom_separators(self):
        """测试自定义分隔符"""
        chunker = RecursiveChunker(
            chunk_size=100,
            chunk_overlap=10,
            separators=["\n\n", "\n", "|"]
        )
        
        docs = [
            Document(
                page_content="第一部分|第二部分|第三部分",
                metadata={"source": "test"}
            )
        ]
        
        result = chunker.split_documents(docs)
        
        assert len(result) > 0


class TestSemanticChunker:
    """语义分块器测试"""

    def test_paragraph_and_sentence_split(self):
        """测试段落和句子分割"""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
        
        docs = [
            Document(
                page_content="第一段落的句子一。第一段落的句子二。\n\n第二段落的句子一。第二段落的句子二。",
                metadata={"source": "test"}
            )
        ]
        
        result = chunker.split_documents(docs)
        
        assert len(result) > 0
        for doc in result:
            assert "chunk_id" in doc.metadata
            assert "total_chunks" in doc.metadata

    def test_sentence_split(self):
        """测试纯句子分割"""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20, split_by="sentence")
        
        docs = [
            Document(
                page_content="这是第一个句子。这是第二个句子。这是第三个句子。",
                metadata={"source": "test"}
            )
        ]
        
        result = chunker.split_documents(docs)
        
        assert len(result) > 0

    def test_paragraph_split(self):
        """测试纯段落分割"""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20, split_by="paragraph")
        
        docs = [
            Document(
                page_content="第一段内容。\n\n第二段内容。\n\n第三段内容。",
                metadata={"source": "test"}
            )
        ]
        
        result = chunker.split_documents(docs)
        
        assert len(result) > 0

    def test_empty_documents(self):
        """测试空文档列表"""
        chunker = SemanticChunker()
        
        result = chunker.split_documents([])
        
        assert result == []

    def test_metadata_preservation(self):
        """测试元数据保留"""
        chunker = SemanticChunker()
        
        original_metadata = {"source": "test", "author": "tester"}
        docs = [
            Document(
                page_content="测试内容。",
                metadata=original_metadata
            )
        ]
        
        result = chunker.split_documents(docs)
        
        for doc in result:
            assert doc.metadata["source"] == "test"
            assert doc.metadata["author"] == "tester"


class TestAdaptiveChunker:
    """自适应分块器测试"""

    def test_pdf_document_chunking(self):
        """测试PDF文档分块"""
        chunker = AdaptiveChunker(chunk_size=500, chunk_overlap=50)
        
        docs = [
            Document(
                page_content="PDF文档内容。" * 50,
                metadata={"type": "pdf"}
            )
        ]
        
        result = chunker.split_documents(docs)
        
        assert len(result) > 0

    def test_code_document_chunking(self):
        """测试代码文档分块"""
        chunker = AdaptiveChunker(chunk_size=500, chunk_overlap=50)
        
        code_content = """
def function1():
    pass

def function2():
    pass

class MyClass:
    def method(self):
        pass
"""
        docs = [
            Document(
                page_content=code_content,
                metadata={"type": "code"}
            )
        ]
        
        result = chunker.split_documents(docs)
        
        assert len(result) > 0

    def test_unknown_type_uses_default(self):
        """测试未知类型使用默认策略"""
        chunker = AdaptiveChunker(chunk_size=500, chunk_overlap=50)
        
        docs = [
            Document(
                page_content="普通文档内容。",
                metadata={"type": "unknown"}
            )
        ]
        
        result = chunker.split_documents(docs)
        
        assert len(result) > 0

    def test_empty_documents(self):
        """测试空文档列表"""
        chunker = AdaptiveChunker()
        
        result = chunker.split_documents([])
        
        assert result == []


class TestGetSplitter:
    """分块器工厂函数测试"""

    def test_get_recursive_splitter(self):
        """测试获取递归分块器"""
        splitter = get_splitter("recursive", chunk_size=200)
        
        assert isinstance(splitter, RecursiveChunker)
        assert splitter.chunk_size == 200

    def test_get_semantic_splitter(self):
        """测试获取语义分块器"""
        splitter = get_splitter("semantic", chunk_size=300)
        
        assert isinstance(splitter, SemanticChunker)
        assert splitter.chunk_size == 300

    def test_get_adaptive_splitter(self):
        """测试获取自适应分块器"""
        splitter = get_splitter("adaptive", chunk_size=400)
        
        assert isinstance(splitter, AdaptiveChunker)
        assert splitter.chunk_size == 400

    def test_get_unknown_type_raises_error(self):
        """测试获取未知类型抛出异常"""
        with pytest.raises(ValueError) as exc_info:
            get_splitter("unknown_type")
        
        assert "Unknown splitter type" in str(exc_info.value)

    def test_default_splitter_type(self):
        """测试默认分块器类型"""
        splitter = get_splitter()
        
        assert isinstance(splitter, SemanticChunker)
