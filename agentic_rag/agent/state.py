"""
LangGraph状态定义
定义Agent工作流中的所有状态
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document


class AgentState(TypedDict):
    """Agent状态定义"""
    
    # 用户输入
    question: str                           # 用户问题
    
    # 意图识别
    intent: str                              # 意图类型：factual(事实查询)/多跳查询(multi_hop)/summary(摘要)/reasoning(推理)
    
    # 查询处理
    rewritten_queries: List[str]            # 改写后的查询列表
    current_query_index: int                 # 当前处理的查询索引
    
    # 检索结果
    retrieved_docs: List[Document]          # 检索到的文档
    reranked_docs: List[Document]           # 重排后的文档
    
    # 生成结果
    generation: str                          # 生成的答案
    refined_answer: str                      # 反思修正后的答案
    
    # 评估
    evaluation: Dict[str, float]             # 评估指标结果
    needs_reflection: bool                   # 是否需要反思
    
    # 工具调用
    tool_results: Dict[str, Any]            # 工具调用结果
    tool_calls: List[str]                    # 调用的工具列表
    
    # 记忆
    memory_context: List[str]              # 记忆上下文
    conversation_history: List[Dict]        # 对话历史
    
    # 元数据
    reflection_count: int                   # 反思次数
    error: Optional[str]                   # 错误信息
    metadata: Dict[str, Any]               # 其他元数据