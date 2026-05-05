"""
安全策略模块 - 防止Agent失控

Agent拥有自主决策能力后，必须加入安全约束防止：
1. 无限循环：Agent反复执行相同行动
2. 过度调用：Agent无限制地调用工具或检索
3. 幻觉决策：Agent做出不合理的行动选择

SafetyPolicy通过计数器和阈值限制Agent行为，
HallucinationGuard通过规则检测不合理的决策。
"""

from typing import Optional
from ...config.logger_config import logger
from .state import Action


class SafetyPolicy:
    """
    安全策略：防止Agent无限循环和过度调用

    通过为每种行动类型设置计数上限，确保Agent不会：
    - 执行过多步骤（MAX_STEPS）
    - 过度调用工具（MAX_TOOL_CALLS）
    - 反复检索同一知识库（MAX_RETRIEVE_CALLS）
    - 反复生成回答（MAX_GENERATE_CALLS）
    - 连续失败后继续尝试（MAX_CONSECUTIVE_FAILS）

    当计数超过阈值时，check()返回False，Agent被强制转为generate行动。
    """

    def __init__(
        self,
        max_steps: int = 10,
        max_tool_calls: int = 5,
        max_retrieve_calls: int = 3,
        max_generate_calls: int = 2,
        max_consecutive_fails: int = 3,
    ):
        """
        初始化安全策略

        参数:
            max_steps: 最大执行步数
            max_tool_calls: 最大工具调用次数
            max_retrieve_calls: 最大检索次数
            max_generate_calls: 最大生成次数
            max_consecutive_fails: 最大连续失败次数
        """
        self.MAX_STEPS = max_steps
        self.MAX_TOOL_CALLS = max_tool_calls
        self.MAX_RETRIEVE_CALLS = max_retrieve_calls
        self.MAX_GENERATE_CALLS = max_generate_calls
        self.MAX_CONSECUTIVE_FAILS = max_consecutive_fails

        self.step_count = 0
        self.tool_call_count = 0
        self.retrieve_count = 0
        self.generate_count = 0
        self.consecutive_fails = 0

    def check(self, action: Action) -> bool:
        """
        检查行动是否被安全策略允许

        每次调用递增对应计数器，超过阈值则拒绝。
        拒绝后Agent应转为generate行动，避免无限循环。

        参数:
            action: 待检查的行动

        返回:
            True表示允许执行，False表示拒绝
        """
        self.step_count += 1

        if self.step_count > self.MAX_STEPS:
            logger.warning(f"安全策略: 超过最大步数限制({self.MAX_STEPS})，强制终止")
            return False

        if action.type == "tool_call":
            self.tool_call_count += 1
            if self.tool_call_count > self.MAX_TOOL_CALLS:
                logger.warning(f"安全策略: 超过最大工具调用次数({self.MAX_TOOL_CALLS})")
                return False

        if action.type == "retrieve":
            self.retrieve_count += 1
            if self.retrieve_count > self.MAX_RETRIEVE_CALLS:
                logger.warning(f"安全策略: 超过最大检索次数({self.MAX_RETRIEVE_CALLS})")
                return False

        if action.type == "generate":
            self.generate_count += 1
            if self.generate_count > self.MAX_GENERATE_CALLS:
                logger.warning(f"安全策略: 超过最大生成次数({self.MAX_GENERATE_CALLS})")
                return False

        return True

    def record_result(self, success: bool):
        """
        记录行动执行结果

        连续失败计数器用于检测"工具反复失败但Agent仍在尝试"的情况。
        成功一次即重置计数器。

        参数:
            success: 行动是否成功
        """
        if success:
            self.consecutive_fails = 0
        else:
            self.consecutive_fails += 1

    def is_consecutive_fail_exceeded(self) -> bool:
        """检查是否超过连续失败阈值"""
        return self.consecutive_fails >= self.MAX_CONSECUTIVE_FAILS

    def reset(self):
        """重置所有计数器（新请求时必须调用）"""
        self.step_count = 0
        self.tool_call_count = 0
        self.retrieve_count = 0
        self.generate_count = 0
        self.consecutive_fails = 0


class HallucinationGuard:
    """
    幻觉防护：检测Agent决策是否合理

    防止Agent做出以下不合理决策：
    1. 调用不存在的工具（LLM幻觉产生的工具名）
    2. 重复执行相同行动（陷入死循环的信号）
    3. 忽略已有检索结果继续检索（浪费资源）

    检测到不合理决策时，自动修正为更合理的行动。
    """

    def __init__(self, tool_registry=None):
        """
        初始化幻觉防护

        参数:
            tool_registry: 工具注册表实例，用于验证工具是否存在
        """
        self.tool_registry = tool_registry

    def validate_action(self, action: Action, state: dict) -> Action:
        """
        验证Agent的行动是否合理，不合理则修正

        修正策略：
        - 工具不存在 → 改为generate（基于已有信息生成）
        - 重复行动 → 改为generate（避免死循环）
        - 有未用文档仍检索 → 改为generate（先利用已有信息）

        参数:
            action: Agent选择的行动
            state: 当前状态

        返回:
            修正后的行动（可能不变）
        """
        if action.type == "tool_call":
            tool_name = action.input.get("name", "")
            if self.tool_registry and not self.tool_registry.has(tool_name):
                logger.warning(f"幻觉防护: Agent请求调用不存在的工具 '{tool_name}'，修正为generate")
                return Action(
                    type="generate",
                    input={},
                    reasoning=f"工具'{tool_name}'不存在，直接基于已有信息生成",
                )

        if self._is_repeated_action(action, state):
            logger.warning(f"幻觉防护: 检测到重复行动 '{action.type}'，修正为generate")
            return Action(
                type="generate",
                input={},
                reasoning="检测到重复行动，避免无效循环，直接生成回答",
            )

        if action.type == "retrieve" and self._has_unused_docs(state):
            logger.warning("幻觉防护: 已有未充分利用的检索结果，修正为generate")
            return Action(
                type="generate",
                input={},
                reasoning="已有检索结果未充分利用，应先生成回答",
            )

        return action

    def _is_repeated_action(self, action: Action, state: dict) -> bool:
        """
        检测是否重复执行相同行动

        连续2次以上执行相同类型的行动视为重复，
        这通常是Agent陷入局部循环的信号。

        参数:
            action: 当前行动
            state: 当前状态

        返回:
            是否为重复行动
        """
        actions = state.get("actions", [])
        if len(actions) < 2:
            return False

        same_type_count = 0
        for past_action in reversed(actions):
            if past_action.get("action_type") == action.type:
                same_type_count += 1
            else:
                break

        return same_type_count >= 2

    def _has_unused_docs(self, state: dict) -> bool:
        """
        检查是否有未使用的检索结果

        如果已经有3篇以上检索文档但还没生成过回答，
        说明Agent在浪费资源重复检索，应该先利用已有信息。

        参数:
            state: 当前状态

        返回:
            是否有未使用的文档
        """
        retrieved = state.get("retrieved_docs", [])
        has_generation = bool(state.get("generation"))
        return len(retrieved) >= 3 and not has_generation
