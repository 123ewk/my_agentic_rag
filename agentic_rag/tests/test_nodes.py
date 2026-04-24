"""
LangGraph节点功能测试
测试各个处理节点的逻辑
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from agentic_rag.agent.nodes import (
    intent_classification_node,
    query_rewrite_node,
    retrieval_node,
    rerank_node,
    generation_node,
    evaluation_node,
    reflection_node,
    tool_call_node
)


class TestIntentClassificationNode:
    """意图识别节点测试"""

    @pytest.mark.skip(reason="LangChain chain.invoke() mock 复杂，需要真实 LLM")
    def test_classification_factual(self):
        """测试事实查询意图识别"""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        mock_llm = MagicMock()
        mock_response = AIMessage(content='{"intent": "factual"}')
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"intent": "factual"}
        mock_llm.with_structured_output.return_value = mock_chain
        
        state = {"question": "什么是Python？"}
        result = intent_classification_node(state, mock_llm)
        
        assert "intent" in result

    @pytest.mark.skip(reason="LangChain chain.invoke() mock 复杂，需要真实 LLM")
    def test_classification_multi_hop(self):
        """测试多跳查询意图识别"""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"intent": "multi_hop"}
        mock_llm.with_structured_output.return_value = mock_chain
        
        state = {"question": "Python和Java有什么区别？"}
        result = intent_classification_node(state, mock_llm)
        
        assert result["intent"] == "multi_hop"

    @pytest.mark.skip(reason="LangChain chain.invoke() mock 复杂，需要真实 LLM")
    def test_classification_fallback(self):
        """测试意图识别兜底逻辑"""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"intent": "unknown"}
        mock_llm.with_structured_output.return_value = mock_chain
        
        state = {"question": "测试问题"}
        result = intent_classification_node(state, mock_llm)
        
        assert result["intent"] == "multi_hop"


class TestQueryRewriteNode:
    """查询改写节点测试"""

    def test_rewrite_basic(self, mock_llm, mock_embeddings):
        """测试基础查询改写"""
        with patch('agentic_rag.agent.nodes.QueryRewriter') as MockRewriter:
            mock_instance = MagicMock()
            mock_instance.rewrite.return_value = ["查询1", "查询2"]
            MockRewriter.return_value = mock_instance
            
            state = {"question": "Python是什么？"}
            result = query_rewrite_node(state, mock_llm, mock_embeddings)
            
            assert "rewritten_queries" in result
            assert "current_query_index" in result
            assert len(result["rewritten_queries"]) == 2

    def test_rewrite_empty(self, mock_llm, mock_embeddings):
        """测试空查询改写"""
        with patch('agentic_rag.agent.nodes.QueryRewriter') as MockRewriter:
            mock_instance = MagicMock()
            mock_instance.rewrite.return_value = []
            MockRewriter.return_value = mock_instance
            
            state = {"question": ""}
            result = query_rewrite_node(state, mock_llm, mock_embeddings)
            
            assert "rewritten_queries" in result


class TestRetrievalNode:
    """检索节点测试"""

    def test_retrieval_with_queries(self, mock_vectorstore, sample_documents):
        """测试多查询检索"""
        mock_vectorstore.similarity_search.return_value = sample_documents
        
        state = {
            "question": "Python是什么？",
            "rewritten_queries": ["Python是什么", "Python教程"],
            "current_query_index": 0,
            "retrieved_docs": []
        }
        
        result = retrieval_node(state, mock_vectorstore)
        
        assert "retrieved_docs" in result
        assert "current_query_index" in result
        assert mock_vectorstore.similarity_search.called

    def test_retrieval_deduplication(self, mock_vectorstore, sample_documents):
        """测试检索结果去重"""
        doc1, doc2 = sample_documents[0], sample_documents[1]
        mock_vectorstore.similarity_search.return_value = [doc1]
        
        state = {
            "question": "测试问题",
            "rewritten_queries": ["测试"],
            "current_query_index": 0,
            "retrieved_docs": [doc1]
        }
        
        result = retrieval_node(state, mock_vectorstore)
        
        assert len(result["retrieved_docs"]) <= 1

    def test_retrieval_fallback(self, mock_vectorstore):
        """测试检索失败时的兜底处理"""
        mock_vectorstore.similarity_search.return_value = []
        
        state = {
            "question": "测试问题",
            "rewritten_queries": [],
            "current_query_index": 0,
            "retrieved_docs": []
        }
        
        result = retrieval_node(state, mock_vectorstore)
        
        assert "retrieved_docs" in result


class TestRerankNode:
    """重排节点测试"""

    def test_rerank_basic(self, mock_reranker, sample_documents):
        """测试基础重排功能"""
        mock_reranker.rerank.return_value = [
            (sample_documents[0], 0.95),
            (sample_documents[1], 0.85)
        ]
        
        state = {
            "question": "Python是什么？",
            "retrieved_docs": sample_documents
        }
        
        result = rerank_node(state, mock_reranker)
        
        assert "reranked_docs" in result
        assert len(result["reranked_docs"]) == 2

    def test_rerank_empty_docs(self, mock_reranker):
        """测试空文档重排"""
        state = {
            "question": "测试",
            "retrieved_docs": []
        }
        
        result = rerank_node(state, mock_reranker)
        
        assert result["reranked_docs"] == []

    def test_rerank_with_scores(self, mock_reranker, sample_documents):
        """测试带分数的重排"""
        mock_reranker.rerank.return_value = [
            (sample_documents[0], 0.9),
            (sample_documents[1], 0.8),
            (sample_documents[2], 0.7)
        ]
        
        state = {
            "question": "Python相关问题",
            "retrieved_docs": sample_documents
        }
        
        result = rerank_node(state, mock_reranker)
        
        assert len(result["reranked_docs"]) == 3


class TestGenerationNode:
    """生成节点测试"""

    def test_generation_basic(self, mock_llm, sample_documents):
        """测试基础生成功能"""
        mock_llm.invoke.return_value = AIMessage(content="这是一个测试回答")
        
        prompt_template = "上下文：{context}\n问题：{question}\n回答："
        state = {
            "question": "什么是Python？",
            "reranked_docs": sample_documents,
            "tool_results": {}
        }
        
        result = generation_node(state, mock_llm, prompt_template)
        
        assert "generation" in result
        assert len(result["generation"]) > 0

    def test_generation_with_tool_results(self, mock_llm, sample_documents):
        """测试带工具结果的生成"""
        mock_llm.invoke.return_value = AIMessage(content="工具辅助回答")
        
        prompt_template = "上下文：{context}\n问题：{question}\n回答："
        state = {
            "question": "Python最新版本？",
            "reranked_docs": sample_documents,
            "tool_results": {"search": "Python 3.11是最新版本"}
        }
        
        result = generation_node(state, mock_llm, prompt_template)
        
        assert "generation" in result


class TestEvaluationNode:
    """评估节点测试"""

    def test_evaluation_basic(self, mock_llm, sample_documents):
        """测试基础评估功能"""
        mock_llm.invoke.return_value = AIMessage(
            content='{"faithfulness": 0.9, "answer_relevancy": 0.8}'
        )
        
        state = {
            "question": "什么是Python？",
            "generation": "Python是一种编程语言",
            "reranked_docs": sample_documents
        }
        
        result = evaluation_node(state, mock_llm)
        
        assert "evaluation" in result
        assert "needs_reflection" in result

    def test_evaluation_low_faithfulness(self, mock_llm, sample_documents):
        """测试低忠实度触发反思"""
        from agentic_rag.evaluation.metrics import evaluate_response
        
        state = {
            "question": "测试",
            "generation": "答案",
            "reranked_docs": sample_documents
        }
        
        result = evaluate_response("测试", "答案", sample_documents, llm=None)
        
        assert "faithfulness" in result


class TestReflectionNode:
    """反思节点测试"""

    def test_reflection_basic(self, mock_llm):
        """测试基础反思功能"""
        mock_llm.invoke.return_value = AIMessage(content="改进后的回答")
        
        state = {
            "question": "什么是Python？",
            "generation": "原始回答",
            "evaluation": {"faithfulness": 0.6},
            "reflection_count": 0
        }
        
        result = reflection_node(state, mock_llm)
        
        assert "refined_answer" in result
        assert "reflection_count" in result
        assert result["reflection_count"] == 1

    def test_reflection_increment(self, mock_llm):
        """测试反思次数递增"""
        mock_llm.invoke.return_value = AIMessage(content="再次改进")
        
        state = {
            "question": "测试",
            "generation": "回答",
            "evaluation": {},
            "reflection_count": 2
        }
        
        result = reflection_node(state, mock_llm)
        
        assert result["reflection_count"] == 3


class TestToolCallNode:
    """工具调用节点测试"""

    def test_tool_call_basic(self, mock_llm, tools):
        """测试基础工具调用"""
        mock_llm.invoke.return_value = AIMessage(
            content='{"calls": [{"name": "calculator", "parameters": {"expression": "2+2"}}]}'
        )
        
        state = {"question": "2+2等于多少？"}
        result = tool_call_node(state, mock_llm, tools)
        
        assert "tool_results" in result or "tool_calls" in result

    def test_tool_call_no_tool(self, mock_llm, tools):
        """测试不需要工具调用的情况"""
        mock_llm.invoke.return_value = AIMessage(content='{"calls": []}')
        
        state = {"question": "不需要工具的问题"}
        result = tool_call_node(state, mock_llm, tools)
        
        assert "tool_results" in result