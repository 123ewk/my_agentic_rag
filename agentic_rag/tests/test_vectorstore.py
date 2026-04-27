"""
向量存储模块测试
测试嵌入模型和向量存储功能
"""
import pytest
from unittest.mock import MagicMock, patch
import re

from agentic_rag.vectorstore.embeddings import (
    BaseEmbeddings,
    HuggingFaceEmbeddings,
    ZhiPuEmbeddings,
    get_embeddings,
    _clean_text
)


class TestCleanText:
    """文本清洗函数测试"""

    def test_remove_control_characters(self):
        """测试移除控制字符"""
        text = "Hello\x00World\x07Test"
        result = _clean_text(text)
        
        assert "\x00" not in result
        assert "\x07" not in result

    def test_remove_zero_width_characters(self):
        """测试移除零宽字符"""
        text = "Hello\u200bWorld\u200cTest"
        result = _clean_text(text)
        
        assert "\u200b" not in result
        assert "\u200c" not in result

    def test_normalize_whitespace(self):
        """测试空白字符规范化"""
        text = "Hello   World\n\nTest  \t  Tab"
        result = _clean_text(text)
        
        assert "   " not in result
        assert "\n\n" not in result

    def test_strip_whitespace(self):
        """测试去除首尾空白"""
        text = "  Hello World  "
        result = _clean_text(text)
        
        assert result == "Hello World"

    def test_empty_text(self):
        """测试空文本"""
        result = _clean_text("")
        
        assert result == ""

    def test_mixed_invalid_characters(self):
        """测试混合无效字符"""
        text = "  Hello\x00World\u200b  \n\nTest  "
        result = _clean_text(text)
        
        assert "Hello" in result
        assert "World" in result
        assert "Test" in result


class TestBaseEmbeddings:
    """基础嵌入类测试"""

    def test_base_class_is_abstract(self):
        """测试基类是抽象的"""
        with pytest.raises(TypeError):
            BaseEmbeddings()


class TestZhiPuEmbeddingsMocked:
    """智谱嵌入模型测试（模拟）"""

    def test_initialization(self):
        """测试初始化"""
        embeddings = ZhiPuEmbeddings(
            api_key="test_key",
            model_name="embedding-3",
            dimension=1024
        )
        
        assert embeddings.api_key == "test_key"
        assert embeddings.model_name == "embedding-3"
        assert embeddings.dimension == 1024

    def test_max_batch_size(self):
        """测试最大批处理大小"""
        assert ZhiPuEmbeddings.MAX_BATCH_SIZE == 64

    @patch('agentic_rag.vectorstore.embeddings.OpenAI')
    def test_embed_query_mock(self, mock_openai):
        """测试模拟查询嵌入"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1024)]
        mock_openai.return_value.embeddings.create.return_value = mock_response
        
        embeddings = ZhiPuEmbeddings(api_key="test_key")
        result = embeddings.embed_query("测试查询")
        
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)

    @patch('agentic_rag.vectorstore.embeddings.OpenAI')
    def test_embed_query_empty_text(self, mock_openai):
        """测试空文本嵌入"""
        embeddings = ZhiPuEmbeddings(api_key="test_key")
        
        with pytest.raises(ValueError) as exc_info:
            embeddings.embed_query("   \x00   ")
        
        assert "文本清洗后为空" in str(exc_info.value)

    @patch('agentic_rag.vectorstore.embeddings.OpenAI')
    def test_embed_documents_mock(self, mock_openai):
        """测试模拟批量文档嵌入"""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1024),
            MagicMock(embedding=[0.2] * 1024)
        ]
        mock_openai.return_value.embeddings.create.return_value = mock_response
        
        embeddings = ZhiPuEmbeddings(api_key="test_key")
        texts = ["文档1", "文档2"]
        result = embeddings.embed_documents(texts)
        
        assert len(result) == 2
        assert len(result[0]) == 1024

    @patch('agentic_rag.vectorstore.embeddings.OpenAI')
    def test_embed_documents_empty_list(self, mock_openai):
        """测试空文档列表"""
        embeddings = ZhiPuEmbeddings(api_key="test_key")
        
        with pytest.raises(ValueError) as exc_info:
            embeddings.embed_documents(["", "   "])
        
        assert "所有文本清洗后均为空" in str(exc_info.value)


class TestHuggingFaceEmbeddingsMocked:
    """HuggingFace嵌入模型测试（模拟）"""

    @patch('agentic_rag.vectorstore.embeddings.SentenceTransformer')
    def test_initialization(self, mock_transformer):
        """测试初始化"""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_transformer.return_value = mock_model
        
        embeddings = HuggingFaceEmbeddings(
            model_name="test-model",
            device="cpu",
            normalize_embeddings=True
        )
        
        assert embeddings.dimension == 1024
        assert embeddings.normalize_embeddings is True

    @patch('agentic_rag.vectorstore.embeddings.SentenceTransformer')
    def test_embed_query_mock(self, mock_transformer):
        """测试模拟查询嵌入"""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.encode.return_value = MagicMock(tolist=MagicMock(return_value=[0.1] * 1024))
        mock_transformer.return_value = mock_model
        
        embeddings = HuggingFaceEmbeddings()
        result = embeddings.embed_query("测试查询")
        
        assert len(result) == 1024

    @patch('agentic_rag.vectorstore.embeddings.SentenceTransformer')
    def test_embed_documents_mock(self, mock_transformer):
        """测试模拟批量文档嵌入"""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.encode.return_value = MagicMock(tolist=MagicMock(return_value=[[0.1] * 1024, [0.2] * 1024]))
        mock_transformer.return_value = mock_model
        
        embeddings = HuggingFaceEmbeddings()
        texts = ["文档1", "文档2"]
        result = embeddings.embed_documents(texts)
        
        assert len(result) == 2
        assert len(result[0]) == 1024


class TestGetEmbeddings:
    """嵌入工厂函数测试"""

    def test_get_zhipu_embeddings(self):
        """测试获取智谱嵌入"""
        with patch('agentic_rag.vectorstore.embeddings.OpenAI'):
            embeddings = get_embeddings("zhipu", api_key="test")
            
            assert isinstance(embeddings, ZhiPuEmbeddings)

    @patch('agentic_rag.vectorstore.embeddings.SentenceTransformer')
    def test_get_huggingface_embeddings(self, mock_transformer):
        """测试获取HuggingFace嵌入"""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_transformer.return_value = mock_model
        
        embeddings = get_embeddings("huggingface")
        
        assert isinstance(embeddings, HuggingFaceEmbeddings)

    def test_get_unknown_type_raises_error(self):
        """测试未知类型抛出异常"""
        with pytest.raises(ValueError) as exc_info:
            get_embeddings("unknown_type")
        
        assert "未知的嵌入类型" in str(exc_info.value)

    def test_default_type_is_zhipu(self):
        """测试默认类型是智谱"""
        with patch('agentic_rag.vectorstore.embeddings.OpenAI'):
            embeddings = get_embeddings()
            
            assert isinstance(embeddings, ZhiPuEmbeddings)


class TestEmbeddingsBatching:
    """嵌入批处理测试"""

    @patch('agentic_rag.vectorstore.embeddings.OpenAI')
    def test_batch_size_limit(self, mock_openai):
        """测试批处理大小限制"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1024)] * 70
        mock_openai.return_value.embeddings.create.return_value = mock_response
        
        embeddings = ZhiPuEmbeddings(api_key="test")
        
        texts = [f"文档{i}" for i in range(70)]
        result = embeddings.embed_documents(texts)
        
        assert len(result) == 70
        assert mock_openai.return_value.embeddings.create.call_count >= 2

    @patch('agentic_rag.vectorstore.embeddings.OpenAI')
    def test_single_batch(self, mock_openai):
        """测试单批次处理"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1024)] * 10
        mock_openai.return_value.embeddings.create.return_value = mock_response
        
        embeddings = ZhiPuEmbeddings(api_key="test")
        
        texts = [f"文档{i}" for i in range(10)]
        result = embeddings.embed_documents(texts)
        
        assert len(result) == 10
        assert mock_openai.return_value.embeddings.create.call_count == 1
