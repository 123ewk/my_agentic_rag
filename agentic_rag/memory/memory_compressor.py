"""
记忆压缩器

核心能力:
1. 从对话中提取高价值记忆候选
2. 将原始文本压缩为结构化内容
3. 从具体事件提炼通用经验
4. 防止过度压缩导致信息丢失
"""
import json
from typing import Dict, Any, Optional, List
from loguru import logger

from .memory_evaluator import (
    MemoryType,
    MemoryCandidate,
    MEMORY_VALUE_THRESHOLD,
    should_discard_immediately,
    evaluate_memory_value,
)

EXTRACTION_PROMPT = """从以下对话中提取值得长期记住的信息。

规则:
1. 只提取长期有效、未来可复用的信息
2. 丢弃临时性、一次性、可从外部获取的信息
3. 丢弃问候、确认、闲聊等无决策价值的内容
4. 保留否定性信息(如"不要用X")，缺失会导致错误决策
5. 压缩后内容不超过100字，摘要不超过30字

对话内容:
用户: {question}
助手: {answer}

请严格以JSON格式输出:
{{
    "should_remember": true或false,
    "memory_type": "user_profile或fact或experience或preference",
    "content": "压缩后的核心信息(不超过100字)",
    "summary": "一句话摘要(不超过30字)",
    "reasoning": "为什么值得/不值得记住"
}}

记忆类型说明:
- user_profile: 用户的职业、技术栈、角色等稳定属性
- fact: 项目配置、系统架构、业务规则等长期有效事实
- experience: 成功/失败案例、排查步骤、最佳实践、踩坑记录
- preference: 工具偏好、输出格式偏好、交互风格偏好"""


EXPERIENCE_GENERALIZATION_PROMPT = """将以下具体事件提炼为通用经验规则。

具体事件: {specific_event}

要求:
1. 去除具体的时间、人名等临时细节
2. 保留可复用的模式、方法、结论
3. 保留否定性信息(不要做什么)
4. 输出格式: "当[条件]时，[行动]，因为[原因]"
5. 不超过100字

通用经验:"""


async def extract_memory_candidate(
    llm,
    question: str,
    answer: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[MemoryCandidate]:
    """
    从对话中提取并评估记忆候选

    流程: LLM提取 → 硬性过滤 → 价值评分 → 返回候选或None

    Args:
        llm: 大语言模型实例
        question: 用户问题
        answer: AI回答
        context: 额外上下文

    Returns:
        MemoryCandidate 或 None(不值得记忆)
    """
    if not answer or len(answer.strip()) < 10:
        return None

    if not question or len(question.strip()) < 5:
        return None

    try:
        prompt = EXTRACTION_PROMPT.format(
            question=question[:500],
            answer=answer[:800],
        )

        response = await llm.ainvoke(prompt)
        raw_text = response.content if hasattr(response, 'content') else str(response)

        result = _parse_extraction_result(raw_text)
        if result is None:
            logger.debug(f"记忆提取JSON解析失败，跳过: {raw_text[:100]}")
            return None

        if not result.get("should_remember", False):
            logger.debug(f"LLM判断不值得记忆: {result.get('reasoning', '')}")
            return None

        content = result.get("content", "").strip()
        summary = result.get("summary", "").strip()
        memory_type_str = result.get("memory_type", "fact")

        # 硬性丢弃检查
        if should_discard_immediately(content):
            logger.debug(f"硬性丢弃: {content[:50]}")
            return None

        # 解析记忆类型
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            memory_type = MemoryType.FACT

        # 价值评分
        value_score = evaluate_memory_value(content, memory_type, context)

        if value_score < MEMORY_VALUE_THRESHOLD:
            logger.debug(f"价值评分过低({value_score:.2f})，丢弃: {content[:50]}")
            return None

        if not summary:
            summary = content[:30]

        return MemoryCandidate(
            content=content,
            memory_type=memory_type,
            summary=summary,
            value_score=value_score,
            metadata={
                "reasoning": result.get("reasoning", ""),
                "extraction_source": "llm",
            },
        )

    except Exception as e:
        logger.warning(f"记忆提取失败: {e}")
        return None


def _parse_extraction_result(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    从LLM输出中解析JSON结果

    支持多种格式: 纯JSON、markdown代码块包裹的JSON等

    Args:
        raw_text: LLM原始输出

    Returns:
        解析后的字典，失败返回None
    """
    # 尝试直接解析
    text = raw_text.strip()

    # 去除markdown代码块标记
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # 提取JSON对象
    json_str = _extract_json(text)
    if json_str is None:
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def _extract_json(text: str) -> Optional[str]:
    """
    从文本中提取最外层JSON对象(支持嵌套)

    Args:
        text: 包含JSON的文本

    Returns:
        JSON字符串，未找到返回None
    """
    stack = []
    start = None
    in_string = False
    escape = False

    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            if not stack:
                start = i
            stack.append(c)
        elif c == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    return text[start:i + 1]

    return None


async def generalize_experience(llm, specific_event: str) -> Optional[str]:
    """
    将具体事件提炼为通用经验

    Args:
        llm: 大语言模型实例
        specific_event: 具体事件描述

    Returns:
        通用经验描述，失败返回None
    """
    try:
        prompt = EXPERIENCE_GENERALIZATION_PROMPT.format(
            specific_event=specific_event[:500]
        )
        response = await llm.ainvoke(prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        result = result.strip()

        # 过度压缩检测: 结果太短可能丢失了关键信息
        if len(result) < 15:
            return None

        return result
    except Exception as e:
        logger.warning(f"经验泛化失败: {e}")
        return None


def check_over_compression(original: str, compressed: str) -> bool:
    """
    检测是否过度压缩

    规则:
    1. 信息密度不应超过原始内容的3倍
    2. 否定性信息不应被丢失

    Args:
        original: 原始内容
        compressed: 压缩后内容

    Returns:
        True表示过度压缩，需要补充
    """
    if not original or not compressed:
        return True

    # 检查否定性信息是否丢失
    negative_patterns = ["不要", "不能", "不可", "避免", "禁止", "不支持", "不兼容"]
    for pattern in negative_patterns:
        if pattern in original and pattern not in compressed:
            return True

    # 信息密度检查: 压缩比过高说明可能丢失信息
    original_facts = len(original) / 20  # 粗略估计原始事实数
    compressed_facts = len(compressed) / 15  # 压缩后信息密度更高
    if original_facts > 0 and compressed_facts / original_facts > 3:
        return True

    return False
