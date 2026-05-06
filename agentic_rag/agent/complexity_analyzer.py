"""
问题复杂度分析器
用于自动判断用户问题应该使用 DAG 模式还是 ReAct 模式

设计原则：
- DAG 模式（默认）：快速、高效，适合简单问题
- ReAct 模式：复杂推理、调试、多步决策，适合复杂问题
"""
import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
from loguru import logger


class ComplexityLevel(Enum):
    """复杂度等级"""
    SIMPLE = "simple"           # 简单问题，DAG模式
    MODERATE = "moderate"       # 中等问题，DAG模式但可升级
    COMPLEX = "complex"         # 复杂问题，ReAct模式


# ReAct 触发关键词（复杂场景）
REACT_TRIGGERS = {
    # 多步骤/顺序执行
    "multi_step": [
        r"首先", r"然后", r"接着", r"最后", r"第一步", r"第二步", r"第三步",
        r"依次", r"逐步", r"顺序", r"流程",
        r"先做.*再做了", r"做了.*再做.*", r".*之后.*之前",
    ],
    # 原因分析/解释
    "reasoning": [
        r"为什么", r"为何", r"怎么会", r"原因是什么",
        r"解释.*为什么", r"说明.*原因", r"分析.*原因",
        r"是什么导致", r"导致.*的原因是",
    ],
    # 调试/排错
    "debugging": [
        r"报错", r"错误", r"异常", r"失败", r"问题",
        r"调试", r"排查", r"解决.*问题", r"修复",
        r"出了什么问题", r"哪里错了", r"如何避免",
    ],
    # 设计与规划
    "design": [
        r"设计", r"规划", r"架构", r"方案",
        r"应该如何设计", r"怎么架构", r"最佳实践",
        r"应该用.*还是.*", r"选择.*还是.*",
    ],
    # 算法与实现
    "algorithm": [
        r"算法", r"实现.*功能", r"怎么实现",
        r"代码", r"函数", r"类", r"模块",
        r"写.*代码", r"编写.*程序",
    ],
    # 深度分析
    "analysis": [
        r"分析", r"深度", r"详细",
        r"深入.*分析", r"全面.*分析",
        r"对比.*分析", r"优缺点.*分析",
    ],
}

# DAG 优化关键词（简单场景）
DAG_OPTIMIZERS = {
    # 事实查询
    "factual": [
        r"什么是", r"什么叫", r"哪个是", r"是什么",
        r"定义", r"概念", r"的意思",
        r"简单.*介绍", r"简要.*说明",
    ],
    # 总结/摘要
    "summary": [
        r"总结", r"概括", r"归纳", r"摘要",
        r"梳理", r"整理", r"汇总",
        r"主要.*是", r"核心.*是",
    ],
    # 简单比较
    "comparison": [
        r"比较.*和.*", r".*vs.*", r".*VS.*",
        r"区别", r"差异", r"不同",
        r"哪个好", r"有什么.*区别",
    ],
}


class ComplexityAnalyzer:
    """
    问题复杂度分析器

    判断逻辑：
    1. 先检查是否匹配复杂问题特征（ReAct触发词）
    2. 再检查是否匹配简单问题特征（DAG优化词）
    3. 综合评分，输出决策
    """

    def __init__(self):
        self.react_triggers = REACT_TRIGGERS
        self.dag_optimizers = DAG_OPTIMIZERS

    def analyze(self, question: str) -> Tuple[ComplexityLevel, Dict[str, any]]:
        """
        分析问题复杂度

        参数:
            question: 用户问题

        返回:
            (ComplexityLevel, 分析详情字典)
        """
        question = question.strip()

        # 1. 检查是否匹配复杂问题特征
        react_matches = self._check_patterns(question, self.react_triggers)

        # 2. 检查是否匹配简单问题特征
        dag_matches = self._check_patterns(question, self.dag_optimizers)

        # 3. 计算复杂度分数
        score = self._calculate_score(react_matches, dag_matches, question)

        # 4. 长度加权（长问题更可能复杂）
        length_weight = min(1.0, len(question) / 100)  # 100字符以上满分

        # 5. 综合判断
        final_score = score + (length_weight * 0.1)

        if final_score >= 0.6:
            level = ComplexityLevel.COMPLEX
        elif final_score >= 0.3:
            level = ComplexityLevel.MODERATE
        else:
            level = ComplexityLevel.SIMPLE

        detail = {
            "question": question,
            "react_matches": react_matches,
            "dag_matches": dag_matches,
            "base_score": score,
            "length_weight": length_weight,
            "final_score": final_score,
            "level": level,
            "recommended_mode": "react" if level == ComplexityLevel.COMPLEX else "dag",
        }

        logger.debug(
            f"复杂度分析: level={level.value}, score={final_score:.2f}, "
            f"react_matches={list(react_matches.keys())}, dag_matches={list(dag_matches.keys())}"
        )

        return level, detail

    def _check_patterns(self, question: str, patterns: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """检查问题匹配了哪些模式"""
        matches = {}
        for category, regex_list in patterns.items():
            matched = []
            for pattern in regex_list:
                if re.search(pattern, question):
                    matched.append(pattern)
            if matched:
                matches[category] = matched
        return matches

    def _calculate_score(
        self,
        react_matches: Dict[str, List[str]],
        dag_matches: Dict[str, List[str]],
        question: str
    ) -> float:
        """
        计算复杂度分数

        分数规则：
        - 每个ReAct匹配类别: +0.15
        - 每个DAG匹配类别: -0.2
        - 分数范围: [0, 1]
        """
        score = 0.0

        # ReAct 匹配加分
        react_categories = len(react_matches)
        score += min(0.6, react_categories * 0.15)  # 最多0.6

        # DAG 匹配减分
        dag_categories = len(dag_matches)
        score -= min(0.4, dag_categories * 0.2)  # 最多减0.4

        # 限制在 [0, 1] 范围内
        return max(0.0, min(1.0, score))

    def should_use_react(self, question: str) -> Tuple[bool, Optional[str]]:
        """
        判断是否应该使用 ReAct 模式

        参数:
            question: 用户问题

        返回:
            (是否使用ReAct, 原因说明)
        """
        level, detail = self.analyze(question)

        if level == ComplexityLevel.COMPLEX:
            # 构建原因说明
            reasons = []
            for category in detail["react_matches"]:
                reasons.append(f"涉及{category}特征")

            reason = " + ".join(reasons) if reasons else "问题复杂，需要多步推理"
            return True, reason

        return False, None


# 全局单例
_analyzer: Optional[ComplexityAnalyzer] = None


def get_complexity_analyzer() -> ComplexityAnalyzer:
    """获取复杂度分析器单例"""
    global _analyzer
    if _analyzer is None:
        _analyzer = ComplexityAnalyzer()
    return _analyzer


def analyze_question_complexity(question: str) -> Tuple[str, Dict[str, any]]:
    """
    快速分析问题复杂度（便捷函数）

    参数:
        question: 用户问题

    返回:
        (推荐模式, 分析详情)
    """
    analyzer = get_complexity_analyzer()
    level, detail = analyzer.analyze(question)
    mode = "react" if level == ComplexityLevel.COMPLEX else "dag"
    return mode, detail


if __name__ == "__main__":
    # 简单测试
    analyzer = ComplexityAnalyzer()

    test_cases = [
        "什么是Python？",
        "帮我总结一下RAG系统的核心组件",
        "比较一下Python和Java的区别",
        "解释一下为什么会出现这个错误",
        "首先连接数据库，然后查询用户表，最后关闭连接，这个流程有什么问题？",
        "设计一个高并发的分布式系统架构",
        "帮我调试这段代码，报错信息是IndexError",
    ]

    for q in test_cases:
        level, detail = analyzer.analyze(q)
        print(f"\n问题: {q}")
        print(f"  复杂度: {level.value} (score={detail['final_score']:.2f})")
        print(f"  推荐模式: {detail['recommended_mode']}")
        print(f"  React匹配: {detail['react_matches']}")
        print(f"  DAG匹配: {detail['dag_matches']}")