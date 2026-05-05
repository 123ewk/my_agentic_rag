"""
LangGraph边逻辑
定义节点之间的连接和路由
"""

from typing import Literal
from .state import AgentState
from .nodes import should_skip_evaluation

# CRAG最大循环次数，防止 evaluation→web_search→generation→evaluation 无限循环
MAX_CRAG_LOOPS = 1

def route_after_intent(state: AgentState) -> Literal["tool_call", "query_rewrite", "generation", "retrieval"]:
    """意图识别后的路由"""
    intent = state.get("intent", "factual")
    
    if intent == "tool_call":
        return "tool_call"
    elif intent == "multi_hop" or intent == "reasoning":
        return "query_rewrite"
    elif intent == "summary":
        return "generation"
    elif intent == "factual":
        return "retrieval"
    else:
        return "generation"

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

def route_after_generation(state: AgentState) -> Literal["evaluation", "__end__"]:
    """
    生成回答后的路由（优化C：增加快速退出）

    高质量答案（答案够长+有文档/工具支撑）直接结束，
    跳过evaluation和reflection，节省0-5秒延迟。
    """
    if should_skip_evaluation(state):
        return "__end__"
    return "evaluation"

def route_after_evaluation(state: AgentState) -> Literal["reflection", "__end__"]:
    """评估后的路由"""
    needs_reflection = state.get("needs_reflection", False)
    reflection_count = state.get("reflection_count", 0)
    max_reflection = state.get("metadata", {}).get("max_reflection_steps", 0)
    
    if needs_reflection and reflection_count < max_reflection:
        return "reflection"
    else:
        return "__end__"

def route_after_reflection(state: AgentState) -> Literal["__end__"]:
    """
    反思后的路由（优化F：反思后直接结束）

    原逻辑是reflection→evaluation→可能再次reflection，形成循环。
    优化后反思完成直接结束，避免无效循环。
    关键词评估在反思后大概率不变，再次评估无意义。
    """
    return "__end__"


def route_after_web_search(state: AgentState) -> Literal["generation", "__end__"]:
    """
    网络搜索后的路由
    
    有搜索结果 -> generation (使用搜索结果重新生成)
    无搜索结果 -> __end__ (返回已有答案)
    """
    search_results = state.get("search_results", [])
    
    if search_results:
        return "generation"
    else:
        return "__end__"
