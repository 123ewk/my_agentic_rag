"""
LangGraph节点实现
定义工作流中的各个处理节点
"""

import re
import json
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional, AsyncIterator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..config.logger_config import logger
from ..config.settings import get_settings
from ..tools.tool_calls import tool_call
from .state import AgentState
from ..retrieval.query_rewrite import QueryRewriter, _clean_think_tags
from ..tools.search import duckduckgo_search


def _rule_based_intent(question: str) -> Optional[str]:
    """
    规则预过滤意图：用正则匹配常见模式，命中则跳过LLM调用

    参数:
        question: 用户问题

    返回:
        匹配到的意图字符串，未匹配返回None
    """
    q = question.strip()

    if re.search(r'(今天|现在|最新|当前|实时|最近|近期|这几天|这周)', q):
        return "tool_call"

    if re.search(r'(总结|概括|梳理|归纳|摘要|提炼|整理一下|汇总)', q):
        return "summary"

    if len(q) < 20 and re.search(r'(什么是|什么叫|哪个|多少|谁|在哪|何时)', q):
        return "factual"

    if re.search(r'(对比|比较|区别|优缺点|异同|vs|VS|还是)', q):
        return "multi_hop"

    if re.search(r'(为什么|怎么会|原因|如何实现|怎么解决|怎么办|为什么)', q):
        return "reasoning"

    return None


def intent_classification_node(
    state: AgentState,
    llm: BaseChatModel,
    intent_cache: Optional[object] = None
) -> AgentState:
    """
    意图识别节点（带规则预过滤 + 缓存优化）

    优化层级：
    1. 规则预过滤：常见模式直接匹配，跳过LLM调用
    2. 内存LRU缓存：快速访问，减少重复LLM调用
    3. LLM调用：仅在规则和缓存都未命中时调用
    """
    question = state["question"]
    settings = get_settings()

    # 优化D：规则预过滤——常见模式直接匹配，避免LLM调用
    rule_intent = _rule_based_intent(question)
    if rule_intent:
        logger.info(f"规则预过滤命中: '{rule_intent}' <- {question[:50]}...")
        if settings.intent_cache_enabled and intent_cache is not None:
            intent_cache.set(question, rule_intent)
        return {"intent": rule_intent}

    cached_intent = None
    if settings.intent_cache_enabled and intent_cache is not None:
        cached_intent = intent_cache.get(question)
        if cached_intent:
            logger.info(f"意图缓存命中: '{cached_intent}' <- {question[:50]}...")
            return {"intent": cached_intent}

    intents = ["factual", "multi_hop", "summary", "reasoning", "tool_call"]

    intent_prompt = ChatPromptTemplate.from_template("""
        你是一个用户意图分类助手,需要根据用户的问题,将其归类为以下5种类型之一:

        1. factual:事实查询类问题,比如"什么是Python?""哪个是正确的?",只需要直接给出客观事实答案
        2. reasoning:推理/原因分析类问题,比如"为什么会报错?""如何实现这个功能?",需要解释原因或给出步骤
        3. summary:总结/概括类问题,比如"帮我总结一下这段内容""梳理要点",需要提炼核心信息
        4. multi_hop:多步推理/复杂查询类问题,比如"对比A和B的区别并说明优缺点""先做X再做Y会怎样?",需要多轮信息或多步逻辑才能回答
        5. tool_call:需要实时信息/最新数据/联网搜索的问题,比如"今天天气怎么样?""最新的人工智能新闻""帮我查一下这个概念""现在几点"等,涉及当前时间、实时数据、网络搜索等

        用户问题:{question}

        请严格按照JSON格式返回,key为"intent",值为上面5个类型之一:
        {{"intent": "xxx"}}
        """
    )

    chain = intent_prompt | llm

    try:
        response = chain.invoke({"question": question})
        raw_text = response.content if hasattr(response, 'content') else str(response)

        cleaned = _clean_think_tags(raw_text)
        # 提取JSON里的"intent"键对应的字符串值,大模型的输出，经常不是 “干净的纯 JSON”。
        json_match = re.search(r'\{[^{}]*"intent"\s*:\s*"[^"]+?"[^{}]*\}', cleaned)
        if json_match:
            result = json.loads(json_match.group()) # .group() 方法返回匹配的字符串,.loads() 方法将字符串转换为Python对象
            intent = result.get("intent", "multi_hop") # 从Python对象中获取"intent"键对应的值,如果不存在则返回"multi_hop"
        else:
            logger.warning(f"意图识别未找到有效JSON,原始输出: {raw_text[:200]}")
            intent = "multi_hop"
    except Exception as e:
        logger.warning(f"意图识别解析失败,使用默认意图: {e}")
        intent = "multi_hop"

    if intent not in intents:
        intent = "multi_hop"

    if settings.intent_cache_enabled and intent_cache is not None:
        intent_cache.set(question, intent)

    return {"intent": intent}

def query_rewrite_node(state: AgentState, llm, embeddings) -> AgentState:
    """查询改写节点"""
    question = state["question"]
    
    rewriter = QueryRewriter(llm, embeddings)
    rewritten_queries = rewriter.rewrite(question, strategy="all")
    
    return {
        "rewritten_queries": rewritten_queries,
        "current_query_index": 0
    }


def intent_and_rewrite_node(
    state: AgentState,
    llm: BaseChatModel,
    intent_cache: Optional[object] = None
) -> AgentState:
    """
    合并意图识别+查询改写节点（优化A：减少2-3次LLM调用为1次）

    将原本的intent_classification(1次LLM) + query_rewrite(2-3次LLM)合并为单次LLM调用，
    同时输出意图分类和改写查询，节省3-5秒延迟。

    参数:
        state: 当前Agent状态
        llm: 大语言模型实例
        intent_cache: 意图缓存实例

    返回:
        包含intent和rewritten_queries的状态更新
    """
    question = state["question"]
    settings = get_settings()

    # 规则预过滤优先
    rule_intent = _rule_based_intent(question)
    if rule_intent:
        logger.info(f"规则预过滤命中(合并节点): '{rule_intent}' <- {question[:50]}...")
        if settings.intent_cache_enabled and intent_cache is not None:
            intent_cache.set(question, rule_intent)
        # 规则命中时，对需要改写的意图生成简单改写
        if rule_intent in ("multi_hop", "reasoning"):
            return {"intent": rule_intent, "rewritten_queries": [question], "current_query_index": 0}
        return {"intent": rule_intent}

    # 缓存检查
    if settings.intent_cache_enabled and intent_cache is not None:
        cached_intent = intent_cache.get(question)
        if cached_intent:
            logger.info(f"意图缓存命中(合并节点): '{cached_intent}' <- {question[:50]}...")
            if cached_intent in ("multi_hop", "reasoning"):
                return {"intent": cached_intent, "rewritten_queries": [question], "current_query_index": 0}
            return {"intent": cached_intent}

    combined_prompt = ChatPromptTemplate.from_template("""
        你是一个查询分析助手，需要同时完成两个任务：

        任务1-意图分类：将问题归类为以下5种类型之一：
        1. factual: 事实查询，只需直接给出客观事实答案
        2. reasoning: 推理/原因分析，需要解释原因或给出步骤
        3. summary: 总结/概括，需要提炼核心信息
        4. multi_hop: 多步推理/复杂查询，需要多轮信息或多步逻辑
        5. tool_call: 需要实时信息/最新数据/联网搜索

        任务2-查询改写：仅对multi_hop和reasoning类型，生成2-3个不同表述的查询；
        其他类型返回空列表。

        用户问题：{question}

        请严格按照JSON格式返回：
        {{"intent": "xxx", "rewrites": ["query1", "query2"]}}
        """
    )

    chain = combined_prompt | llm
    intents = ["factual", "multi_hop", "summary", "reasoning", "tool_call"]

    try:
        response = chain.invoke({"question": question})
        raw_text = response.content if hasattr(response, 'content') else str(response)
        cleaned = _clean_think_tags(raw_text)

        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            intent = result.get("intent", "multi_hop")
            rewrites = result.get("rewrites", [])
        else:
            logger.warning(f"合并节点未找到有效JSON,原始输出: {raw_text[:200]}")
            intent = "multi_hop"
            rewrites = []
    except Exception as e:
        logger.warning(f"合并节点解析失败,使用默认值: {e}")
        intent = "multi_hop"
        rewrites = []

    if intent not in intents:
        intent = "multi_hop"

    # 仅对需要改写的意图保留rewrites
    if intent in ("multi_hop", "reasoning") and rewrites:
        rewritten_queries = [question] + [q for q in rewrites if q and q != question]
    else:
        rewritten_queries = [question]

    if settings.intent_cache_enabled and intent_cache is not None:
        intent_cache.set(question, intent)

    logger.info(f"合并节点结果: intent={intent}, rewrites={len(rewritten_queries)}")

    return {
        "intent": intent,
        "rewritten_queries": rewritten_queries,
        "current_query_index": 0
    }


def parallel_retrieval_node(state: AgentState, vectorstore) -> AgentState:
    """
    并行检索节点 - 优化版本
    
    同时检索所有改写后的查询,然后合并结果
    大幅减少检索时间(从串行O(n)降低到并行O(1))
    """
    queries = state.get("rewritten_queries", [state["question"]])
    
    # 检查向量库是否已初始化
    if not hasattr(vectorstore, 'vectorstore') or vectorstore.vectorstore is None:
        return {
            "retrieved_docs": [],
            "vectorstore_uninitialized": True
        }
    
    # 并行检索所有查询
    all_docs = []
    seen_ids = set()
    
    def retrieve_single(query):
        try:
            return vectorstore.similarity_search(query, k=5)
        except Exception as e:
            logger.warning(f"检索失败: {query}, 错误: {e}")
            return []
    
    # 使用线程池并行检索（max_workers至少为1，防止空查询列表导致崩溃）
    worker_count = max(1, min(len(queries), 5)) if queries else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(retrieve_single, queries))
    
    # 合并所有结果(去重)
    for docs in results:
        for doc in docs:
            doc_id = doc.metadata.get("id", doc.page_content[:100])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
    
    return {
        "retrieved_docs": all_docs
    }


def rerank_node(state: AgentState, reranker) -> AgentState:
    """重排节点"""
    query = state["question"]
    docs = state.get("retrieved_docs", [])
    
    if not docs:
        return {"reranked_docs": []}
    
    # 重排
    reranked = reranker.rerank(query, docs, top_k=3)
    reranked_docs = [doc for doc, score in reranked]
    
    return {"reranked_docs": reranked_docs}

def tool_call_node(state: AgentState, llm: BaseChatModel, tools: Dict[str, BaseTool]) -> AgentState:
    """工具调用节点"""
    return tool_call(state, llm, tools)


def _estimate_tokens(text: str) -> int:
    """
    估算token数量（简单估算：中文约2字符/token，英文约4字符/token）

    参数:
        text: 输入文本

    返回:
        估算的token数量
    """
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 2 + other_chars / 4)


def _truncate_context(
    context_parts: List[str],
    max_tokens: int,
    max_docs: int
) -> List[str]:
    """
    截断上下文以符合token限制

    参数:
        context_parts: 上下文部分列表
        max_tokens: 最大token数
        max_docs: 最多使用的文档数量

    返回:
        截断后的上下文列表
    """
    result = []
    total_tokens = 0

    for part in context_parts:
        part_tokens = _estimate_tokens(part)

        if "[检索到的文档]" in part and max_docs > 0:
            import re
            # 提取检索到的文档
            # 匹配【检索到的文档】\n后面的所有内容，直到遇到\n\n或字符串结束,re.DOTALL: 匹配换行符,作用是让 . 可以匹配任意字符，包括换行符 \n
            docs_match = re.search(r'【检索到的文档】\n(.*)', part, re.DOTALL)
            if docs_match:
                docs_text = docs_match.group(1) # .group(1) 表示取出正则表达式中第 1 个捕获组（也就是 (.*) 匹配到的内容）
                doc_blocks = re.split(r'\n\n+', docs_text)

                kept_docs = []
                kept_docs_tokens = 0
                for doc in doc_blocks[:max_docs]:
                    doc_tokens = _estimate_tokens(doc)
                    if total_tokens + kept_docs_tokens + doc_tokens <= max_tokens:
                        kept_docs.append(doc)
                        kept_docs_tokens += doc_tokens
                    else:
                        break

                if kept_docs:
                    part = "【检索到的文档】\n" + "\n\n".join(kept_docs)
                    part_tokens = kept_docs_tokens
        else:
            if total_tokens + part_tokens > max_tokens:
                continue

        total_tokens += part_tokens
        result.append(part)

    return result


def generation_node(state: AgentState, llm, prompt_template: str) -> AgentState:
    """
    生成节点，支持记忆上下文和上下文截断

    功能：
    - 支持记忆上下文和对话历史
    - 上下文截断：避免过长上下文导致LLM输入超限
    - 按优先级保留：记忆 > 对话历史 > 检索文档
    - CRAG支持：融合网络搜索结果
    """
    question = state["question"]
    context_docs = state.get("reranked_docs", [])
    tool_results = state.get("tool_results", {})
    search_results = state.get("search_results", [])  # CRAG网络搜索结果
    conversation_history = state.get("conversation_history", [])
    memory_context = state.get("memory_context", [])
    settings = get_settings()

    if not isinstance(tool_results, dict):
        tool_results = {}

    if state.get("vectorstore_uninitialized", False):
        generation = f"您好!我目前还没有加载知识库内容,无法基于文档回答您的问题。\n\n请先使用文档上传接口(POST /api/v1/upload)上传您的文档,我会自动建立索引后再为您服务。\n\n上传文档后,我就能基于您的知识库回答问题了!"
        return {"generation": generation}

    context_parts = _build_context_parts(
        memory_context, conversation_history, context_docs,
        search_results, tool_results, state.get("tool_call_failed", False), settings
    )

    context = "\n".join(context_parts) if context_parts else "(无相关上下文)"

    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)

    generation = response.content if hasattr(response, 'content') else str(response)
    generation = _clean_think_tags(generation)

    return {"generation": generation}


async def generation_node_stream(
    state: AgentState,
    llm,
    prompt_template: str
) -> AsyncIterator[str]:
    """
    流式生成节点（优化E：真正的token级流式输出）

    与generation_node的区别：
    - generation_node使用llm.invoke()同步等待完整生成，是伪流式
    - 本节点使用llm.astream()逐token产出，用户可实时看到输出

    参数:
        state: 当前Agent状态
        llm: 大语言模型实例
        prompt_template: 提示词模板

    产出:
        逐token的文本片段
    """
    question = state["question"]
    context_docs = state.get("reranked_docs", [])
    tool_results = state.get("tool_results", {})
    search_results = state.get("search_results", [])
    conversation_history = state.get("conversation_history", [])
    memory_context = state.get("memory_context", [])
    settings = get_settings()

    if not isinstance(tool_results, dict):
        tool_results = {}

    if state.get("vectorstore_uninitialized", False):
        yield "您好!我目前还没有加载知识库内容，请先上传文档。"
        return

    context_parts = _build_context_parts(
        memory_context, conversation_history, context_docs,
        search_results, tool_results, state.get("tool_call_failed", False), settings
    )

    context = "\n".join(context_parts) if context_parts else "(无相关上下文)"
    prompt = prompt_template.format(context=context, question=question)

    full_text = []
    async for chunk in llm.astream(prompt):
        token = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if token:
            full_text.append(token)
            yield token


def _build_context_parts(
    memory_context,
    conversation_history,
    context_docs,
    search_results,
    tool_results,
    tool_call_failed: bool,
    settings
) -> List[str]:
    """
    构建上下文部分列表（从generation_node和generation_node_stream中提取的公共逻辑）

    参数:
        memory_context: 记忆上下文
        conversation_history: 对话历史
        context_docs: 检索到的文档
        search_results: 网络搜索结果
        tool_results: 工具调用结果
        tool_call_failed: 工具调用是否失败
        settings: 配置实例

    返回:
        上下文部分列表
    """
    context_parts = []

    if memory_context:
        if isinstance(memory_context, list):
            memory_text = "\n".join(memory_context)
        else:
            memory_text = str(memory_context)
        # 限制记忆上下文长度，防止上下文污染
        max_memory_chars = 2000
        if len(memory_text) > max_memory_chars:
            memory_text = memory_text[-max_memory_chars:]
        context_parts.append(f"【相关记忆】\n{memory_text}")

    if conversation_history:
        history_lines = []
        # 只保留最近5轮对话，防止长对话上下文污染
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        for msg in recent_history:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            # 截断单条消息，防止单条过长
            if len(content) > 500:
                content = content[:500] + "..."
            history_lines.append(f"{role}: {content}")
        if history_lines:
            context_parts.append(f"【对话历史】\n" + "\n".join(history_lines))

    if context_docs:
        docs_content = "\n\n".join([doc.page_content for doc in context_docs])
        context_parts.append(f"【检索到的文档】\n{docs_content}")

    # CRAG: 添加网络搜索结果到上下文
    if search_results:
        search_content = "\n\n".join([doc.page_content for doc in search_results])
        context_parts.append(f"【网络搜索结果】\n{search_content}")

    if tool_results and isinstance(tool_results, dict):
        tool_context = "\n\n【工具调用结果】\n"
        for tool_name, result in tool_results.items():
            tool_context += f"- {tool_name}: {result}\n"
        context_parts.append(tool_context)

    # 工具调用失败提示：让LLM知道工具调用出了问题
    if tool_call_failed and not tool_results:
        context_parts.append("【注意】工具调用未能成功执行，请基于已有知识回答，不要引用工具结果。")

    if settings.context_truncation_enabled:
        context_parts = _truncate_context(
            context_parts,
            max_tokens=settings.max_context_tokens,
            max_docs=settings.max_docs_for_context
        )

    return context_parts


def should_skip_evaluation(state: AgentState) -> bool:
    """
    快速退出判断（优化C）：高质量答案跳过evaluation+reflection

    判断逻辑：
    - 答案够长（>100字）且有检索文档支撑（≥2篇）→ 跳过评估直接结束
    - 答案够长且来自工具调用结果 → 跳过评估直接结束

    参数:
        state: 当前Agent状态

    返回:
        True表示可以跳过evaluation直接结束
    """
    answer = state.get("generation", "")
    docs = state.get("reranked_docs", [])
    tool_results = state.get("tool_results", {})

    if len(answer) > 100 and len(docs) >= 2:
        return True

    if len(answer) > 100 and tool_results and isinstance(tool_results, dict) and len(tool_results) > 0:
        return True

    return False

def evaluation_node(state: AgentState, llm) -> AgentState:
    """
    评估节点(轻量级版本,不调用LLM)
    
    评估维度：
    - faithfulness: 答案与上下文的一致性（基于关键词重叠度而非纯长度）
    - answer_relevancy: 答案与问题的相关性
    - context_precision: 上下文精度
    - completeness: 答案完整性
    - overall_score: 加权综合得分
    """
    question = state["question"]
    answer = state["generation"]
    context_docs = state.get("reranked_docs", [])
    settings = get_settings()
    
    answer_len = len(answer) if answer else 0
    context_len = sum(len(doc.page_content) for doc in context_docs) if context_docs else 0

    # 关键词重叠度：检查答案中是否包含上下文的关键词
    context_text = " ".join([doc.page_content for doc in context_docs]) if context_docs else ""
    question_keywords = set(question.replace("？", "").replace("?", "").replace("，", "").replace(",", "").split())
    answer_keywords = set(answer.replace("？", "").replace("?", "").replace("，", "").replace(",", "").split()) if answer else set()
    context_keywords = set(context_text.replace("，", "").replace(",", "").split()) if context_text else set()

    # 问题-答案关键词重叠率（答案是否回应了问题中的关键概念）
    question_answer_overlap = len(question_keywords & answer_keywords) / max(len(question_keywords), 1)
    # 答案-上下文关键词重叠率（答案是否基于上下文）
    answer_context_overlap = len(answer_keywords & context_keywords) / max(len(answer_keywords), 1) if answer_keywords else 0.0

    # faithfulness: 基于答案与上下文的关键词重叠度，而非纯长度
    faithfulness = min(1.0, answer_context_overlap * 1.5) if answer_len > 0 else 0.0
    # answer_relevancy: 基于问题与答案的关键词重叠度
    answer_relevancy = min(1.0, question_answer_overlap * 2.0) if context_docs else 0.3
    context_precision = min(1.0, len(context_docs) / 5.0) if context_docs else 0.0
    completeness = min(1.0, answer_len / 500)
    overall_score = faithfulness * 0.35 + answer_relevancy * 0.35 + context_precision * 0.15 + completeness * 0.15
    
    # 反思触发条件：答案过短、无上下文、或答案与问题几乎无关
    needs_reflection = answer_len < 50 or (not context_docs and context_len == 0) or question_answer_overlap < 0.1
    
    confidence_level = "high"
    confidence_score = overall_score
    if overall_score < settings.crag_confidence_threshold_low:
        confidence_level = "low"
    elif overall_score < settings.crag_confidence_threshold_high:
        confidence_level = "medium"
    
    # 保存上一次评估结果，供反思节点对比
    previous_evaluation = state.get("evaluation")

    return {
        "evaluation": {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "completeness": completeness,
            "overall_score": overall_score
        },
        "needs_reflection": needs_reflection,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "previous_evaluation": previous_evaluation
    }


def reflection_node(state: AgentState, llm) -> AgentState:
    """
    反思修正节点
    
    改进点：
    - 传入具体评估指标，让LLM知道哪里不好
    - 传入上一次评估结果用于对比
    - 反思后如果答案变差，保留原答案
    """
    question = state["question"]
    original_answer = state["generation"]
    evaluation = state["evaluation"]
    previous_evaluation = state.get("previous_evaluation")
    reflection_count = state.get("reflection_count", 0)
    
    # 构建具体的改进指导
    weak_points = []
    if evaluation.get("faithfulness", 1.0) < 0.5:
        weak_points.append("答案与检索到的上下文不够一致，请更紧密地基于上下文内容回答")
    if evaluation.get("answer_relevancy", 1.0) < 0.5:
        weak_points.append("答案与用户问题的相关性不足，请更直接地回应问题核心")
    if evaluation.get("completeness", 1.0) < 0.3:
        weak_points.append("答案不够完整，请补充更多细节")
    
    improvement_guidance = "\n".join(f"- {point}" for point in weak_points) if weak_points else "- 请提升答案的整体质量"
    
    # 构建反思prompt（包含具体评估指标和改进方向）
    reflection_prompt = f"""
        请反思以下回答的质量,并根据评估反馈改进。

        原始问题：{question}
        原始回答：{original_answer}

        评估结果：
        - 忠实度（答案与上下文一致性）：{evaluation.get('faithfulness', 'N/A')}
        - 相关性（答案与问题相关性）：{evaluation.get('answer_relevancy', 'N/A')}
        - 上下文精度：{evaluation.get('context_precision', 'N/A')}
        - 完整性：{evaluation.get('completeness', 'N/A')}
        - 综合得分：{evaluation.get('overall_score', 'N/A')}

        需要改进的方面：
        {improvement_guidance}

        请生成一个改进后的回答，确保：
        1. 更紧密地基于上下文信息
        2. 更直接地回答用户问题
        3. 提供更完整的解释
        """
    
    response = llm.invoke(reflection_prompt)
    refined = response.content if hasattr(response, 'content') else str(response)
    refined = _clean_think_tags(refined)
    
    # 质量守卫：如果反思后答案明显变差（如变短很多），保留原答案
    if len(refined) < len(original_answer) * 0.3 and len(original_answer) > 50:
        logger.warning(f"反思后答案明显变短({len(refined)} vs {len(original_answer)})，保留原答案")
        refined = original_answer
    
    return {
        "generation": refined,
        "refined_answer": refined,
        "reflection_count": reflection_count + 1
    }


def web_search_node(state: AgentState, llm) -> AgentState:
    """
    网络搜索节点(CRAG低置信度触发)
    
    当本地检索置信度不足时,通过网络搜索获取最新/更准确的信息
    使用项目自带的duckduckgo_search工具
    
    改进：递增crag_loop_count，配合路由守卫防止无限循环
    """
    from langchain_core.documents import Document
    question = state["question"]
    crag_loop_count = state.get("crag_loop_count", 0)
    logger.info(f"CRAG触发网络搜索(第{crag_loop_count + 1}次): {question[:50]}...")
    
    try:
        # 调用duckduckgo_search工具
        search_text = duckduckgo_search.invoke({"query": question})
        
        if not search_text or search_text == "未找到相关结果":
            logger.warning("网络搜索未返回结果")
            return {
                "search_results": [],
                "needs_reflection": False,
                "crag_loop_count": crag_loop_count + 1
            }
        
        # 将搜索结果转换为Document格式
        # duckduckgo_search返回的是格式化文本，需要解析
        docs = []
        lines = search_text.split("\n")
        current_result = {"title": "", "body": "", "url": ""}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是结果编号行 (如 "1. 标题")
            if line and line[0].isdigit() and ". " in line:
                # 保存前一个结果
                if current_result["title"] or current_result["body"]:
                    content = f"标题: {current_result['title']}\n内容: {current_result['body']}\n来源: {current_result['url']}"
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": current_result["url"],
                            "title": current_result["title"],
                            "type": "web_search",
                            "score": 1.0
                        }
                    )
                    docs.append(doc)
                
                # 开始新的结果
                title = line.split(". ", 1)[1] if ". " in line else line
                current_result = {"title": title, "body": "", "url": ""}
            
            # 检查是否是来源行
            elif line.startswith("来源:"):
                current_result["url"] = line.replace("来源:", "").strip()
            
            # 否则是内容行
            elif "   " in line:
                current_result["body"] += line.strip() + " "
        
        # 保存最后一个结果
        if current_result["title"] or current_result["body"]:
            content = f"标题: {current_result['title']}\n内容: {current_result['body']}\n来源: {current_result['url']}"
            doc = Document(
                page_content=content,
                metadata={
                    "source": current_result["url"],
                    "title": current_result["title"],
                    "type": "web_search",
                    "score": 1.0
                }
            )
            docs.append(doc)
        
        logger.info(f"网络搜索完成,获取到{len(docs)}条结果")
        
        return {
            "search_results": docs,
            "needs_reflection": True,  # 网络搜索后可能需要反思
            "crag_loop_count": crag_loop_count + 1
        }
        
    except Exception as e:
        logger.error(f"网络搜索失败: {e}")
        return {
            "search_results": [],
            "needs_reflection": False,
            "crag_loop_count": crag_loop_count + 1
        }
