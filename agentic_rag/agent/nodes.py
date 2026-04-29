"""
LangGraph节点实现
定义工作流中的各个处理节点
"""

import re
import json
import concurrent.futures
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..config.logger_config import logger
from ..config.settings import get_settings
from ..tools.tool_calls import tool_call
from .state import AgentState
from ..retrieval.query_rewrite import QueryRewriter, _clean_think_tags
from ..retrieval.hybrid_search import HybridSearchRetriever
from ..retrieval.reranker import get_reranker
from ..tools.search import duckduckgo_search
from ..evaluation.metrics import evaluate_response
from ..memory.intent_cache import get_intent_cache


def intent_classification_node(
    state: AgentState,
    llm: BaseChatModel,
    intent_cache: Optional[object] = None
) -> AgentState:
    """
    意图识别节点（带缓存优化）

    支持：
    - 内存LRU缓存：快速访问，减少重复LLM调用
    - 可选Redis持久化：支持多实例共享缓存
    """
    question = state["question"]
    settings = get_settings()

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

        请严格按照JSON格式返回,key为"intent",值为上面4个类型之一:
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


def retrieval_node(state: AgentState, vectorstore) -> AgentState:
    """检索节点"""
    queries = state.get("rewritten_queries", [state["question"]])
    current_idx = state.get("current_query_index", 0)
    
    # 获取当前查询
    query = queries[current_idx] if current_idx < len(queries) else state["question"]
    
    # 检查向量库是否已初始化
    if not hasattr(vectorstore, 'vectorstore') or vectorstore.vectorstore is None:
        # 向量库未初始化,返回空结果
        return {
            "retrieved_docs": [],
            "current_query_index": current_idx + 1,
            "vectorstore_uninitialized": True
        }
    
    # 检索文档
    docs = vectorstore.similarity_search(query, k=5)
    
    # 合并到已有结果(去重),集合推导式
    existing_ids = {
        doc.metadata.get("id", doc.page_content[:100]) 
        for doc in state.get("retrieved_docs", [])
    }
    
    new_docs = [
        doc for doc in docs 
        if doc.metadata.get("id", doc.page_content[:100]) not in existing_ids
    ]
    
    all_docs = state.get("retrieved_docs", []) + new_docs
    
    # 这是一个 循环检索模式 。检索节点被设计为每次执行只处理一个查询,通过 current_query_index 追踪进度。
    return {
        "retrieved_docs": all_docs,
        "current_query_index": current_idx + 1
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
    
    # 使用线程池并行检索
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
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

    context_parts = []

    if memory_context:
        if isinstance(memory_context, list):
            memory_text = "\n".join(memory_context)
        else:
            memory_text = str(memory_context)
        context_parts.append(f"【相关记忆】\n{memory_text}")

    # MiniMax模型会回显prompt中的对话历史，因此跳过对话历史传入
    model_name = getattr(llm, 'model_name', '') or getattr(llm, 'model', '')
    is_minimax = "minimax" in model_name.lower()
    
    if not is_minimax and conversation_history:
        history_lines = []
        for msg in conversation_history:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        if history_lines:
            context_parts.append(f"【对话历史】\n" + "\n".join(history_lines))
    elif is_minimax and conversation_history:
        logger.debug(f"MiniMax模型在generation_node中跳过对话历史，避免预览回显问题")

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

    if settings.context_truncation_enabled:
        context_parts = _truncate_context(
            context_parts,
            max_tokens=settings.max_context_tokens,
            max_docs=settings.max_docs_for_context
        )

    context = "\n".join(context_parts) if context_parts else "(无相关上下文)"

    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)

    generation = response.content if hasattr(response, 'content') else str(response)
    generation = _clean_think_tags(generation)

    return {"generation": generation}

def evaluation_node(state: AgentState, llm) -> AgentState:
    """
    评估节点(轻量级版本,不调用LLM)
    
    评估维度：
    - faithfulness: 回答忠实度（基于回答长度和上下文覆盖率）
    - answer_relevancy: 回答相关性（基于文档与问题的关键词重叠度）
    - context_precision: 上下文精确度（基于文档数量和相关性得分）
    - completeness: 回答完整性
    
    CRAG依赖overall_score判断是否触发网络搜索
    """
    question = state["question"]
    answer = state["generation"]
    context_docs = state.get("reranked_docs", [])
    
    answer_len = len(answer) if answer else 0
    context_len = sum(len(doc.page_content) for doc in context_docs) if context_docs else 0
    
    # 1. 忠实度：回答越长且上下文越丰富，越可能忠实
    faithfulness = min(1.0, answer_len / 200) if answer_len > 0 else 0.0
    
    # 2. 回答相关性：基于问题关键词在文档中的出现率
    # 提取问题中的关键词（简单分词：按空格和标点分割，过滤短词）
    question_keywords = set(
        w.lower() for w in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{2,}', question)
    )
    
    if context_docs and question_keywords:
        # 计算关键词在文档中的覆盖率
        all_doc_text = " ".join(doc.page_content for doc in context_docs).lower()
        matched = sum(1 for kw in question_keywords if kw in all_doc_text)
        keyword_coverage = matched / len(question_keywords) if question_keywords else 0.0
        answer_relevancy = min(1.0, keyword_coverage * 1.2)  # 略微放大，0.8以上算好
    elif context_docs:
        # 有文档但无法提取关键词，给中等分数
        answer_relevancy = 0.5
    else:
        # 无文档，相关性低
        answer_relevancy = 0.1
    
    # 3. 上下文精确度：基于文档的reranker得分
    if context_docs:
        relevance_scores = [doc.metadata.get("score", 0.5) for doc in context_docs]
        avg_score = sum(relevance_scores) / len(relevance_scores)
        context_precision = min(1.0, avg_score * len(context_docs) / 3.0)
    else:
        context_precision = 0.0
    
    # 4. 完整性
    completeness = min(1.0, answer_len / 500)
    
    # 综合评分
    overall_score = (
        faithfulness * 0.25 +
        answer_relevancy * 0.35 +
        context_precision * 0.25 +
        completeness * 0.15
    )
    
    # 判断是否需要反思
    needs_reflection = answer_len < 50 or overall_score < 0.3
    
    # CRAG置信度等级（基于overall_score）
    settings = get_settings()
    if settings.crag_enabled:
        if overall_score >= settings.crag_confidence_threshold_high:
            confidence_level = "high"
        elif overall_score >= settings.crag_confidence_threshold_low:
            confidence_level = "medium"
        else:
            confidence_level = "low"
    else:
        confidence_level = "high"
    
    logger.info(
        f"评估结果: overall={overall_score:.3f}, "
        f"faith={faithfulness:.2f}, relevancy={answer_relevancy:.2f}, "
        f"precision={context_precision:.2f}, completeness={completeness:.2f}, "
        f"confidence={confidence_level}, docs={len(context_docs)}"
    )
    
    return {
        "evaluation": {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "completeness": completeness,
            "overall_score": overall_score
        },
        "needs_reflection": needs_reflection,
        "confidence_score": overall_score,
        "confidence_level": confidence_level,
    }


def reflection_node(state: AgentState, llm) -> AgentState:
    """反思修正节点"""
    question = state["question"]
    original_answer = state["generation"]
    evaluation = state["evaluation"]
    reflection_count = state.get("reflection_count", 0)
    
    # 构建反思prompt
    reflection_prompt = f"""
        请反思以下回答的质量,并根据反馈改进。

        原始问题：{question}
        原始回答：{original_answer}

        评估结果：
        - 忠实度：{evaluation.get('faithfulness', 'N/A')}
        - 相关性：{evaluation.get('answer_relevancy', 'N/A')}

        请生成一个改进后的回答：
        """
    
    response = llm.invoke(reflection_prompt)
    refined = response.content if hasattr(response, 'content') else str(response)
    refined = _clean_think_tags(refined)
    
    return {
        "refined_answer": refined,
        "reflection_count": reflection_count + 1
    }


def web_search_node(state: AgentState, llm) -> AgentState:
    """
    网络搜索节点(CRAG低置信度触发)
    
    当本地检索置信度不足时,通过网络搜索获取最新/更准确的信息
    使用项目自带的duckduckgo_search工具
    """
    from langchain_core.documents import Document
    question = state["question"]
    logger.info(f"CRAG触发网络搜索: {question[:50]}...")
    
    try:
        # 调用duckduckgo_search工具
        search_text = duckduckgo_search.invoke({"query": question})
        
        if not search_text or search_text == "未找到相关结果":
            logger.warning("网络搜索未返回结果")
            return {
                "search_results": [],
                "needs_reflection": False
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
            "needs_reflection": True  # 网络搜索后可能需要反思
        }
        
    except Exception as e:
        logger.error(f"网络搜索失败: {e}")
        return {
            "search_results": [],
            "needs_reflection": False
        }
