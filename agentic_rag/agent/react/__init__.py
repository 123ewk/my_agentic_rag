"""
ReAct Agent模块 - 基于Observe→Think→Act循环的真正Agent系统

与DAG工作流的根本区别：
- DAG: 预定义路径，LLM只做节点内处理，路由由硬编码规则决定
- ReAct: LLM自主决策每一步行动，根据观察动态调整策略

包含：
- ReActAgent: 主Agent类
- ReActState: Agent状态定义
- SafetyPolicy: 安全策略（防无限循环）
- HallucinationGuard: 幻觉防护
- ToolRegistry: 工具注册表
"""

from .react_agent import ReActAgent
from .state import ReActState, Action, ActionResult, ActionRecord
from .safety import SafetyPolicy, HallucinationGuard
from .tools import ToolRegistry

__all__ = [
    "ReActAgent",
    "ReActState",
    "Action",
    "ActionResult",
    "ActionRecord",
    "SafetyPolicy",
    "HallucinationGuard",
    "ToolRegistry",
]
