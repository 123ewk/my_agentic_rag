"""
LangGraph边逻辑
定义节点之间的连接和路由
"""

from typing import Literal
from .state import AgentState

def route_after_intent(state: AgentState) -> Literal["tool_call", "query_rewrite", "generation"]:
    """意图识别后的路由"""
    intent = state.get("intent", "factual")
    
    if intent == "multi_hop" or intent == "reasoning":
        return "query_rewrite"
    elif intent == "summary":
        return "generation"
    else:
        return "tool_call"

def route_after_rewrite(state: AgentState) -> Literal["retrieval", "rerank"]:
    """查询改写后的路由"""
    queries = state.get("rewritten_queries", [])
    current_idx = state.get("current_query_index", 0)
    
    if current_idx < len(queries):
        return "retrieval"
    else:
        return "rerank"

def route_after_rerank(state: AgentState) -> Literal["generation", "tool_call"]:
    """重排后的路由"""
    tool_calls = state.get("tool_calls", [])
    if isinstance(tool_calls, list) and tool_calls:
        return "tool_call"
    else:
        return "generation"

def route_after_tool_call(state: AgentState) -> Literal["generation"]:
    """工具调用后的路由"""
    # 生成回答
    return "generation"

def route_after_generation(state: AgentState) -> Literal["evaluation"]:
    """生成回答后的路由"""
    # 评估回答
    return "evaluation"

def route_after_evaluation(state: AgentState) -> Literal["reflection", "__end__"]:
    """评估后的路由"""
    needs_reflection = state.get("needs_reflection", False)
    reflection_count = state.get("reflection_count", 0)
    max_reflection = state.get("metadata", {}).get("max_reflection_steps", 2)
    
    if needs_reflection and reflection_count < max_reflection:
        return "reflection"
    else:
        return "__end__"

def route_after_reflection(state: AgentState) -> Literal["evaluation", "__end__"]:
    """反思后的路由"""
    reflection_count = state.get("reflection_count", 0)
    max_reflection = state.get("metadata", {}).get("max_reflection_steps", 2)
    
    if reflection_count < max_reflection:
        return "evaluation"
    else:
        return "__end__"