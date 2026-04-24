"""
LangGraph边路由逻辑测试
测试节点之间的路由决策
"""
import pytest
from agentic_rag.agent.edges import (
    route_after_intent,
    route_after_rewrite,
    route_after_rerank,
    route_after_tool_call,
    route_after_generation,
    route_after_evaluation,
    route_after_reflection
)


class TestRouteAfterIntent:
    """意图识别后路由测试"""

    def test_multi_hop_routing(self):
        """测试多跳查询路由到查询改写"""
        state = {"intent": "multi_hop", "question": "测试"}
        result = route_after_intent(state)
        assert result == "query_rewrite"

    def test_reasoning_routing(self):
        """测试推理查询路由到查询改写"""
        state = {"intent": "reasoning", "question": "测试"}
        result = route_after_intent(state)
        assert result == "query_rewrite"

    def test_summary_routing(self):
        """测试摘要查询路由到生成"""
        state = {"intent": "summary", "question": "测试"}
        result = route_after_intent(state)
        assert result == "generation"

    def test_factual_routing(self):
        """测试事实查询路由到工具调用"""
        state = {"intent": "factual", "question": "测试"}
        result = route_after_intent(state)
        assert result == "tool_call"

    def test_default_routing(self):
        """测试默认路由（空intent）"""
        state = {"intent": "", "question": "测试"}
        result = route_after_intent(state)
        assert result == "tool_call"


class TestRouteAfterRewrite:
    """查询改写后路由测试"""

    def test_more_queries_to_retrieval(self):
        """测试还有查询时路由到检索"""
        state = {
            "rewritten_queries": ["查询1", "查询2"],
            "current_query_index": 0
        }
        result = route_after_rewrite(state)
        assert result == "retrieval"

    def test_no_more_queries_to_rerank(self):
        """测试没有更多查询时路由到重排"""
        state = {
            "rewritten_queries": ["查询1"],
            "current_query_index": 1
        }
        result = route_after_rewrite(state)
        assert result == "rerank"

    def test_empty_queries_to_rerank(self):
        """测试空查询列表路由到重排"""
        state = {
            "rewritten_queries": [],
            "current_query_index": 0
        }
        result = route_after_rewrite(state)
        assert result == "rerank"


class TestRouteAfterRerank:
    """重排后路由测试"""

    def test_with_tool_calls(self):
        """测试有工具调用时路由到工具调用"""
        state = {
            "tool_calls": ["search", "calculator"],
            "reranked_docs": []
        }
        result = route_after_rerank(state)
        assert result == "tool_call"

    def test_without_tool_calls(self):
        """测试没有工具调用时路由到生成"""
        state = {
            "tool_calls": [],
            "reranked_docs": []
        }
        result = route_after_rerank(state)
        assert result == "generation"

    def test_empty_tool_calls(self):
        """测试空工具调用列表"""
        state = {
            "tool_calls": [],
            "reranked_docs": []
        }
        result = route_after_rerank(state)
        assert result == "generation"


class TestRouteAfterToolCall:
    """工具调用后路由测试"""

    def test_always_route_to_generation(self):
        """测试工具调用后总是路由到生成"""
        state = {"tool_results": {"search": "结果"}}
        result = route_after_tool_call(state)
        assert result == "generation"


class TestRouteAfterGeneration:
    """生成后路由测试"""

    def test_always_route_to_evaluation(self):
        """测试生成后总是路由到评估"""
        state = {"generation": "测试回答"}
        result = route_after_generation(state)
        assert result == "evaluation"


class TestRouteAfterEvaluation:
    """评估后路由测试"""

    def test_needs_reflection_under_limit(self):
        """测试需要反思且未超限"""
        state = {
            "needs_reflection": True,
            "reflection_count": 0,
            "metadata": {"max_reflection_steps": 2}
        }
        result = route_after_evaluation(state)
        assert result == "reflection"

    def test_needs_reflection_at_limit(self):
        """测试需要反思但已超限"""
        state = {
            "needs_reflection": True,
            "reflection_count": 2,
            "metadata": {"max_reflection_steps": 2}
        }
        result = route_after_evaluation(state)
        assert result == "__end__"

    def test_no_reflection_needed(self):
        """测试不需要反思"""
        state = {
            "needs_reflection": False,
            "reflection_count": 0,
            "metadata": {}
        }
        result = route_after_evaluation(state)
        assert result == "__end__"

    def test_default_max_reflection(self):
        """测试默认最大反思次数"""
        state = {
            "needs_reflection": True,
            "reflection_count": 1,
            "metadata": {}
        }
        result = route_after_evaluation(state)
        assert result == "reflection"


class TestRouteAfterReflection:
    """反思后路由测试"""

    def test_under_max_reflection(self):
        """测试未超最大反思次数"""
        state = {
            "reflection_count": 1,
            "metadata": {"max_reflection_steps": 2}
        }
        result = route_after_reflection(state)
        assert result == "evaluation"

    def test_at_max_reflection(self):
        """测试达到最大反思次数"""
        state = {
            "reflection_count": 2,
            "metadata": {"max_reflection_steps": 2}
        }
        result = route_after_reflection(state)
        assert result == "__end__"

    def test_above_max_reflection(self):
        """测试超过最大反思次数"""
        state = {
            "reflection_count": 3,
            "metadata": {"max_reflection_steps": 2}
        }
        result = route_after_reflection(state)
        assert result == "__end__"