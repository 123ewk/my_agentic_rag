"""
记忆价值评估器

核心原则: 一条信息是否值得长期记忆，取决于它是否满足「未来可复用性」测试。
评分维度:
- P(持久性): 信息随时间衰减的速度，越慢衰减越高
- R(可复用性): 信息在多少不同场景下可能被需要，场景越多越高
- U(独特性): 信息能否从外部源重新获取，越难获取越高
- D(决策影响力): 信息如果缺失是否导致错误决策，影响越大越高
"""
import re
from typing import Dict, Any, Optional
from enum import Enum
from loguru import logger


class MemoryType(str, Enum):
    """记忆类型枚举"""
    USER_PROFILE = "user_profile"
    FACT = "fact"
    EXPERIENCE = "experience"
    PREFERENCE = "preference"


MEMORY_VALUE_THRESHOLD = 0.6

# 各类型记忆的半衰期(天)，用于时间衰减计算
MEMORY_HALF_LIFE = {
    "user_profile": 365,
    "fact": 180,
    "experience": 90,
    "preference": 60,
}

# 意图到记忆类型的映射
INTENT_TO_MEMORY_TYPES = {
    "technical": ["fact", "experience"],
    "preference": ["preference", "user_profile"],
    "factual": ["fact", "user_profile"],
    "troubleshooting": ["experience", "fact"],
}


class MemoryCandidate:
    """候选记忆，评估后决定是否写入"""

    def __init__(
        self,
        content: str,
        memory_type: MemoryType,
        summary: str = "",
        value_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.memory_type = memory_type
        self.summary = summary
        self.value_score = value_score
        self.metadata = metadata or {}


# 硬性丢弃规则: 闲聊/确认/问候等无价值内容
NOISE_PATTERNS = [
    re.compile(r"^(好的|嗯|是|对|谢谢|感谢|不客气|没问题|明白|了解|收到|OK|ok|是的|对呀|没错)", re.IGNORECASE),
    re.compile(r"^(请稍等|稍等|等一下|我来帮你|让我看看|我来查一下)", re.IGNORECASE),
    re.compile(r"^(你好|您好|hi|hello|hey)", re.IGNORECASE),
    re.compile(r"^\W*$", re.IGNORECASE),
]


def should_discard_immediately(content: str) -> bool:
    """
    硬性丢弃判断: 满足任一条件直接丢弃，无需评分

    Args:
        content: 候选记忆内容

    Returns:
        True表示应丢弃
    """
    if not content or len(content.strip()) < 10:
        return True

    stripped = content.strip()
    for pattern in NOISE_PATTERNS:
        if pattern.match(stripped):
            return True

    # 纯数字/日期
    if re.match(r"^[\d\s\-/:.年月日号周星期]+$", stripped):
        return True

    return False


def evaluate_memory_value(
    content: str,
    memory_type: MemoryType,
    context: Optional[Dict[str, Any]] = None,
) -> float:
    """
    计算信息价值分数 (0~1)

    评分公式: score = 0.2*P + 0.25*R + 0.25*U + 0.3*D

    Args:
        content: 记忆内容
        memory_type: 记忆类型
        context: 额外上下文(如对话历史、工具使用等)

    Returns:
        价值分数 0~1
    """
    P = _calc_persistence(content, memory_type)
    R = _calc_reusability(content, memory_type)
    U = _calc_uniqueness(content, memory_type, context)
    D = _calc_decision_impact(content, memory_type, context)

    score = 0.2 * P + 0.25 * R + 0.25 * U + 0.3 * D
    return round(min(1.0, max(0.0, score)), 3)


def _calc_persistence(content: str, memory_type: MemoryType) -> float:
    """
    持久性评分: 信息有效期越长越高

    - user_profile: 用户画像信息通常长期稳定 → 高
    - fact: 事实知识通常长期有效 → 中高
    - experience: 经验可能随环境变化 → 中
    - preference: 偏好可能改变 → 中低
    """
    base_scores = {
        MemoryType.USER_PROFILE: 0.85,
        MemoryType.FACT: 0.75,
        MemoryType.EXPERIENCE: 0.55,
        MemoryType.PREFERENCE: 0.45,
    }

    score = base_scores.get(memory_type, 0.5)

    # 包含时间相关词的内容衰减更快
    temporal_words = ["今天", "昨天", "刚才", "目前", "现在", "暂时", "临时", "当前"]
    for word in temporal_words:
        if word in content:
            score -= 0.15
            break

    # 包含稳定属性词的内容衰减更慢
    stable_words = ["总是", "一直", "习惯", "偏好", "默认", "通常", "永远", "始终"]
    for word in stable_words:
        if word in content:
            score += 0.1
            break

    return min(1.0, max(0.0, score))


def _calc_reusability(content: str, memory_type: MemoryType) -> float:
    """
    可复用性评分: 跨场景适用性越广越高

    - 通用方法论/规则 → 高
    - 特定问题的具体细节 → 低
    """
    base_scores = {
        MemoryType.USER_PROFILE: 0.8,
        MemoryType.FACT: 0.7,
        MemoryType.EXPERIENCE: 0.65,
        MemoryType.PREFERENCE: 0.75,
    }

    score = base_scores.get(memory_type, 0.5)

    # 通用性关键词提升分数
    general_keywords = ["方法", "规则", "原则", "模式", "策略", "步骤", "流程", "习惯", "偏好"]
    for kw in general_keywords:
        if kw in content:
            score += 0.1
            break

    # 过于具体的细节降低分数
    specific_indicators = ["这个", "那次", "刚才那个", "上次那个"]
    specific_count = sum(1 for ind in specific_indicators if ind in content)
    if specific_count >= 2:
        score -= 0.15

    return min(1.0, max(0.0, score))


def _calc_uniqueness(
    content: str,
    memory_type: MemoryType,
    context: Optional[Dict[str, Any]] = None,
) -> float:
    """
    独特性评分: 外部不可获取性越高越高

    - 用户私有信息 → 高
    - 公开文档可查 → 低
    """
    base_scores = {
        MemoryType.USER_PROFILE: 0.9,
        MemoryType.PREFERENCE: 0.85,
        MemoryType.EXPERIENCE: 0.7,
        MemoryType.FACT: 0.5,
    }

    score = base_scores.get(memory_type, 0.5)

    # 用户私有信息特征词
    private_keywords = ["我喜欢", "我习惯", "我的", "我们公司", "我们项目", "我们团队", "我偏好"]
    for kw in private_keywords:
        if kw in content:
            score += 0.15
            break

    # 可从外部获取的特征词
    external_keywords = ["文档说", "官方", "标准", "规范", "定义", "百科"]
    for kw in external_keywords:
        if kw in content:
            score -= 0.2
            break

    # 如果有context且标记了来源是检索文档，降低独特性
    if context and context.get("from_retrieval"):
        score -= 0.2

    return min(1.0, max(0.0, score))


def _calc_decision_impact(
    content: str,
    memory_type: MemoryType,
    context: Optional[Dict[str, Any]] = None,
) -> float:
    """
    决策影响力评分: 缺失后决策偏差越大越高

    - 影响工具选择/策略方向 → 高
    - 不影响决策的补充说明 → 低
    """
    base_scores = {
        MemoryType.EXPERIENCE: 0.8,
        MemoryType.PREFERENCE: 0.75,
        MemoryType.USER_PROFILE: 0.7,
        MemoryType.FACT: 0.6,
    }

    score = base_scores.get(memory_type, 0.5)

    # 高决策影响力关键词
    impact_keywords = ["必须", "不要", "避免", "注意", "关键", "重要", "务必", "禁止", "推荐"]
    for kw in impact_keywords:
        if kw in content:
            score += 0.15
            break

    # 否定性信息通常决策影响力高(缺失会导致错误操作)
    negative_keywords = ["不要用", "不能用", "不支持", "不兼容", "会失败", "会报错", "有问题"]
    for kw in negative_keywords:
        if kw in content:
            score += 0.1
            break

    # 如果context中有工具使用记录，说明该对话影响了工具选择
    if context and context.get("tools_used"):
        score += 0.05

    return min(1.0, max(0.0, score))


def calculate_decay_score(
    memory_type: str,
    created_at,
    access_count: int = 0,
    current_time=None,
) -> float:
    """
    时间衰减分数计算

    不同类型记忆的衰减速率不同，被多次检索的记忆衰减更慢

    Args:
        memory_type: 记忆类型
        created_at: 创建时间
        access_count: 被检索命中次数
        current_time: 当前时间(可注入用于测试)

    Returns:
        衰减分数 0~1
    """
    from datetime import datetime, timezone

    if current_time is None:
        current_time = datetime.now(timezone.utc)

    if created_at is None:
        return 1.0

    if hasattr(created_at, 'tzinfo') and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    days_since = (current_time - created_at).days
    half_life = MEMORY_HALF_LIFE.get(memory_type, 90)

    if days_since <= 0:
        decay = 1.0
    else:
        decay = 0.5 ** (days_since / half_life)

    # 访问频率加成: 被多次检索的记忆衰减更慢
    access_boost = min(0.3, access_count * 0.05)

    return min(1.0, decay + access_boost)
