"""
AgentState 状态定义测试
验证状态结构和字段类型
"""
import pytest
from agentic_rag.agent.state import AgentState


class TestAgentState:
    """AgentState测试类"""

    def test_state_structure(self):
        """测试状态结构是否符合TypedDict要求"""
        state = AgentState(
            question="测试问题",
            intent="factual",
            rewritten_queries=["查询1", "查询2"],
            current_query_index=1,
            retrieved_docs=[],
            reranked_docs=[],
            generation="测试回答",
            refined_answer="",
            evaluation={"faithfulness": 0.9},
            needs_reflection=False,
            tool_results={"search": "结果"},
            tool_calls=["search"],
            memory_context=["上下文1"],
            conversation_history=[{"role": "user", "content": "问题"}],
            reflection_count=0,
            error=None,
            metadata={"key": "value"}
        )
        
        assert state["question"] == "测试问题"
        assert state["intent"] == "factual"
        assert len(state["rewritten_queries"]) == 2
        assert state["current_query_index"] == 1
        assert state["generation"] == "测试回答"
        assert state["evaluation"]["faithfulness"] == 0.9
        assert state["needs_reflection"] is False

    def test_state_default_values(self):
        """测试状态默认值"""
        state = AgentState(question="测试", intent="", rewritten_queries=[],
                          current_query_index=0, retrieved_docs=[], reranked_docs=[],
                          generation="", refined_answer="", evaluation={},
                          needs_reflection=False, tool_results={}, tool_calls=[],
                          memory_context=[], conversation_history=[], reflection_count=0,
                          error=None, metadata={})
        
        assert state["question"] == "测试"
        assert state["intent"] == ""
        assert state["rewritten_queries"] == []
        assert state["current_query_index"] == 0
        assert state["retrieved_docs"] == []
        assert state["reranked_docs"] == []
        assert state["generation"] == ""
        assert state["refined_answer"] == ""
        assert state["evaluation"] == {}
        assert state["needs_reflection"] is False
        assert state["tool_results"] == {}
        assert state["tool_calls"] == []
        assert state["memory_context"] == []
        assert state["conversation_history"] == []
        assert state["reflection_count"] == 0
        assert state["error"] is None
        assert state["metadata"] == {}

    def test_state_mutability(self):
        """测试状态可变性"""
        state = AgentState(question="原始问题")
        state["question"] = "修改后的问题"
        state["intent"] = "multi_hop"
        state["reflection_count"] = 1
        
        assert state["question"] == "修改后的问题"
        assert state["intent"] == "multi_hop"
        assert state["reflection_count"] == 1

    def test_state_field_types(self):
        """测试字段类型"""
        from langchain_core.documents import Document
        
        docs = [Document(page_content="测试", metadata={"id": "1"})]
        
        state = AgentState(
            question="测试内容",
            intent="factual",
            rewritten_queries=["查询1", "查询2"],
            current_query_index=1,
            retrieved_docs=docs,
            reranked_docs=docs,
            generation="生成内容",
            refined_answer="改进内容",
            evaluation={"score": 0.9},
            needs_reflection=True,
            tool_results={"search": "结果"},
            tool_calls=["search"],
            memory_context=["上下文"],
            conversation_history=[{"role": "user"}],
            reflection_count=1,
            error="错误信息",
            metadata={"key": "value"}
        )
        
        assert isinstance(state["question"], str)
        assert isinstance(state["intent"], str)
        assert isinstance(state["rewritten_queries"], list)
        assert isinstance(state["retrieved_docs"], list)

    def test_state_equality(self):
        """测试状态相等性"""
        state1 = AgentState(
            question="问题",
            intent="factual",
            reflection_count=0
        )
        
        state2 = AgentState(
            question="问题",
            intent="factual",
            reflection_count=0
        )
        
        assert state1["question"] == state2["question"]
        assert state1["intent"] == state2["intent"]
        assert state1["reflection_count"] == state2["reflection_count"]