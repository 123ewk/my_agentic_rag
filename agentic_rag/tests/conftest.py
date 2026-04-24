"""
Pytest 配置文件
提供全局测试夹具和配置
"""
import os
import sys
from typing import Generator
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量（确保测试不依赖真实API）
os.environ.setdefault("ZHIPUAI_API_KEY", "test_key")
os.environ.setdefault("QWEN_API_KEY", "test_key")
os.environ.setdefault("QWEN_API_URL", "http://test.com")
os.environ.setdefault("ZHIPUAI_API_URL", "http://test.com")


@pytest.fixture
def mock_llm():
    """创建模拟LLM"""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=AIMessage(content="测试回答"))
    llm.astream = MagicMock(return_value=AsyncMockGenerator())
    llm.temperature = 0.7
    return llm


@pytest.fixture
def mock_embeddings():
    """创建模拟嵌入模型"""
    embeddings = MagicMock()
    embeddings.embed_query = MagicMock(return_value=[0.1] * 1024)
    embeddings.embed_documents = MagicMock(return_value=[[0.1] * 1024] * 5)
    embeddings.dimension = 1024
    return embeddings


@pytest.fixture
def mock_vectorstore():
    """创建模拟向量存储"""
    vectorstore = MagicMock()
    vectorstore.similarity_search = MagicMock(return_value=[])
    return vectorstore


@pytest.fixture
def mock_reranker():
    """创建模拟重排器"""
    reranker = MagicMock()
    reranker.rerank = MagicMock(return_value=[])
    return reranker


@pytest.fixture
def sample_documents():
    """创建样本文档"""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="这是一个测试文档，用于验证RAG系统的检索功能。",
            metadata={"id": "doc1", "source": "test"}
        ),
        Document(
            page_content="人工智能技术正在快速发展，包括机器学习和深度学习。",
            metadata={"id": "doc2", "source": "test"}
        ),
        Document(
            page_content="Python是一种广泛使用的高级编程语言。",
            metadata={"id": "doc3", "source": "test"}
        ),
    ]


@pytest.fixture
def sample_state():
    """创建示例状态"""
    return {
        "question": "什么是Python？",
        "intent": "",
        "rewritten_queries": [],
        "current_query_index": 0,
        "retrieved_docs": [],
        "reranked_docs": [],
        "generation": "",
        "refined_answer": "",
        "evaluation": {},
        "needs_reflection": False,
        "tool_results": {},
        "tool_calls": [],
        "memory_context": [],
        "conversation_history": [],
        "reflection_count": 0,
        "error": None,
        "metadata": {}
    }


class AsyncMockGenerator:
    """异步生成器模拟"""
    def __init__(self):
        self.content = "测试回答"
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        yield AIMessage(content=self.content)


@pytest.fixture
def tools():
    """创建测试工具"""
    from langchain_core.tools import tool
    
    @tool(description="测试计算器")
    def mock_calculator(expression: str) -> str:
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"错误: {e}"
    
    @tool(description="测试搜索")
    def mock_search(query: str) -> str:
        return f"搜索结果: {query}"
    
    return {
        "calculator": mock_calculator,
        "search": mock_search
    }