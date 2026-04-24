"""
评估指标测试
测试RAG评估功能
"""
import pytest
from unittest.mock import MagicMock, patch

from agentic_rag.evaluation.metrics import (
    RAGEvaluator,
    evaluate_response,
    evaluate_response_async,
    EvaluationMetrics,
    FaithfulnessResult,
    RelevancyResult
)


class TestRAGEvaluator:
    """RAG评估器测试"""

    @pytest.mark.skip(reason="需要真实 LLM 调用进行集成测试")
    @pytest.mark.asyncio
    async def test_evaluate_faithfulness(self, mock_llm):
        """测试忠实度评估"""
        from langchain_core.messages import AIMessage
        
        mock_llm.invoke.return_value = AIMessage(
            content='{"faithfulness_score": 0.9, "reasoning": "评估合理"}'
        )
        
        evaluator = RAGEvaluator(mock_llm)
        
        docs = []
        result = await evaluator.evaluate_faithfulness(
            question="什么是Python？",
            answer="Python是一种编程语言",
            context=docs
        )
        
        assert isinstance(result, FaithfulnessResult)
        assert 0 <= result.faithfulness_score <= 1

    @pytest.mark.skip(reason="需要真实 LLM 调用进行集成测试")
    @pytest.mark.asyncio
    async def test_evaluate_relevancy(self, mock_llm):
        """测试相关性评估"""
        from langchain_core.messages import AIMessage
        
        mock_llm.invoke.return_value = AIMessage(
            content='{"relevancy_score": 0.85, "reasoning": "回答相关"}'
        )
        
        evaluator = RAGEvaluator(mock_llm)
        result = await evaluator.evaluate_answer_relevancy("问题", "回答")
        
        assert isinstance(result, RelevancyResult)

    def test_evaluate_context_precision(self, mock_llm):
        """测试上下文精确度评估"""
        from langchain_core.documents import Document
        
        evaluator = RAGEvaluator(mock_llm)
        
        docs = [
            Document(page_content="Python编程语言", metadata={"id": "1"}),
            Document(page_content="Java编程语言", metadata={"id": "2"}),
        ]
        
        precision = RAGEvaluator.evaluate_context_precision("Python", docs)
        
        assert 0 <= precision <= 1

    def test_evaluate_completeness(self, mock_llm):
        """测试完整性评估"""
        from langchain_core.messages import AIMessage
        
        mock_llm.invoke.return_value = AIMessage(content='{"score": 0.8}')
        
        evaluator = RAGEvaluator(mock_llm)
        result = evaluator.evaluate_completeness("问题", "完整回答")
        
        import asyncio
        loop = asyncio.new_event_loop()
        result_value = loop.run_until_complete(result)
        assert 0 <= result_value <= 1

    @pytest.mark.asyncio
    async def test_evaluate_response(self, mock_llm):
        """测试综合评估响应"""
        from langchain_core.messages import AIMessage
        
        call_count = [0]
        def mock_invoke_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return AIMessage(content='{"faithfulness_score": 0.9, "reasoning": "good"}')
            elif call_count[0] == 2:
                return AIMessage(content='{"relevancy_score": 0.85, "reasoning": "relevant"}')
            else:
                return AIMessage(content='{"score": 0.8}')
        
        mock_llm.invoke.side_effect = mock_invoke_side_effect
        
        evaluator = RAGEvaluator(mock_llm)
        docs = [MagicMock(page_content="测试上下文")]
        result = await evaluator.evaluate_response("问题", "回答", docs)
        
        assert isinstance(result, EvaluationMetrics)
        assert 0 <= result.overall_score <= 1


class TestEvaluateResponse:
    """同步评估接口测试"""

    def test_evaluate_without_llm(self):
        """测试无LLM时的降级处理"""
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="Python", metadata={"id": "1"}),
            Document(page_content="Java", metadata={"id": "2"}),
        ]
        
        result = evaluate_response("Python", "Python是一种语言", docs, llm=None)
        
        assert "faithfulness" in result
        assert "answer_relevancy" in result

    def test_evaluate_response_basic(self, mock_llm):
        """测试基础评估功能"""
        from langchain_core.messages import AIMessage
        
        call_count = [0]
        def mock_invoke_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return AIMessage(content='{"faithfulness_score": 0.9, "reasoning": ""}')
            elif call_count[0] == 2:
                return AIMessage(content='{"relevancy_score": 0.85, "reasoning": ""}')
            else:
                return AIMessage(content='{"score": 0.8}')
        
        mock_llm.invoke.side_effect = mock_invoke_side_effect
        
        docs = [MagicMock(page_content="测试")]
        result = evaluate_response("问题", "回答", docs, llm=mock_llm)
        
        assert isinstance(result, dict)
        assert "faithfulness" in result


class TestEvaluationMetrics:
    """评估指标模型测试"""

    def test_metrics_creation(self):
        """测试评估指标创建"""
        metrics = EvaluationMetrics(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            completeness=0.75,
            overall_score=0.825
        )
        
        assert metrics.faithfulness == 0.9
        assert metrics.overall_score == 0.825

    def test_metrics_to_dict(self):
        """测试指标转字典"""
        metrics = EvaluationMetrics(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            completeness=0.75,
            overall_score=0.825
        )
        
        result = metrics.model_dump()
        
        assert isinstance(result, dict)
        assert result["faithfulness"] == 0.9