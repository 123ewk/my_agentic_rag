"""
ReAct Agent状态定义

与DAG的AgentState不同，ReActState以行动历史为核心，
而非以节点输出为核心。这使得Agent能基于完整历史做决策，
而不是只能看到当前节点的输入输出。
"""

from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain_core.documents import Document


@dataclass
class Action:
    """
    Agent行动定义

    每一步ReAct循环中，Agent选择一个Action来执行。
    action_type决定了执行逻辑，action_input提供参数，
    reasoning记录Agent为什么选择这个行动（用于可解释性）。

    Attributes:
        type: 行动类型（retrieve/web_search/query_rewrite/tool_call/generate/finish）
        input: 行动参数字典
        reasoning: Agent选择此行动的推理过程
    """
    type: str
    input: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class ActionResult:
    """
    行动执行结果

    每个Action执行后返回ActionResult，包含成功/失败状态和数据。
    失败时error字段供Agent在下一轮Observe中感知，从而调整策略。

    Attributes:
        success: 是否执行成功
        data: 成功时的结果数据
        error: 失败时的错误信息
    """
    success: bool
    data: Any = None
    error: Optional[str] = None


@dataclass
class ActionRecord:
    """
    行动历史记录

    保存在state的actions列表中，供Observe阶段回溯。
    Agent通过行动历史判断"我之前做了什么、效果如何"，
    从而避免重复行动或陷入死循环。

    Attributes:
        step: 步骤编号
        action_type: 行动类型
        action_input: 行动参数
        reasoning: 推理过程
        result_summary: 结果摘要
    """
    step: int
    action_type: str
    action_input: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    result_summary: str = ""


class ReActState(TypedDict, total=False):
    """
    ReAct Agent状态定义

    与DAG的AgentState的核心区别：
    - DAG State是"节点输出管道"：每个节点写入固定字段，下一个节点读取
    - ReAct State是"决策上下文"：Agent基于完整历史自主决策，state是决策依据

    关键新增字段：
    - observations: 观察历史（Observe阶段的输出）
    - actions: 行动历史（Think+Act阶段的输出）
    - current_step: 当前步骤（用于安全策略）
    - errors: 错误信号（供Agent感知失败并调整策略）
    """
    question: str

    observations: List[str]
    actions: List[Dict]
    current_step: int

    retrieved_docs: List[Document]
    search_results: List[Document]
    tool_results: Dict[str, Any]
    rewritten_queries: List[str]

    generation: str
    answer: str

    memory_context: List[str]
    conversation_history: List[Dict]

    errors: List[str]
    warnings: List[str]

    metadata: Dict[str, Any]
