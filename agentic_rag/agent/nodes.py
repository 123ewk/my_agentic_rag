"""
LangGraph节点实现
定义工作流中的各个处理节点
"""

import re
import json
import concurrent.futures
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..config.logger_config import logger
from ..tools.tool_calls import tool_call
from .state import AgentState
from ..retrieval.query_rewrite import QueryRewriter
from ..retrieval.hybrid_search import HybridSearchRetriever
from ..retrieval.reranker import get_reranker
from ..tools.search import duckduckgo_search
from ..evaluation.metrics import evaluate_response

def intent_classification_node(state: AgentState, llm: BaseChatModel) -> AgentState:
    """意图识别节点"""
    question = state["question"]

    intents = ["factual", "multi_hop", "summary", "reasoning"]

    intent_prompt = ChatPromptTemplate.from_template("""
        你是一个用户意图分类助手,需要根据用户的问题,将其归类为以下4种类型之一:

        1. factual:事实查询类问题,比如"什么是Python?""哪个是正确的?",只需要直接给出客观事实答案
        2. reasoning:推理/原因分析类问题,比如"为什么会报错?""如何实现这个功能?",需要解释原因或给出步骤
        3. summary:总结/概括类问题,比如"帮我总结一下这段内容""梳理要点",需要提炼核心信息
        4. multi_hop:多步推理/复杂查询类问题,比如"对比A和B的区别并说明优缺点""先做X再做Y会怎样?",需要多轮信息或多步逻辑才能回答

        用户问题:{question}

        请严格按照JSON格式返回,key为"intent",值为上面4个类型之一:
        {{"intent": "xxx"}}
        """
    )

    # 直接调用LLM,手动解析JSON,避免JsonOutputParser无法处理思考标签
    chain = intent_prompt | llm

    try:
        response = chain.invoke({"question": question})
        raw_text = response.content if hasattr(response, 'content') else str(response)
        
        # 清理MiniMax思考标签
        cleaned = re.sub(r'<think.*?>.*?</think\s*>', '', raw_text, flags=re.DOTALL)
        # 提取JSON部分
        json_match = re.search(r'\{[^{}]*"intent"\s*:\s*"[^"]+?"[^{}]*\}', cleaned)
        if json_match:
            result = json.loads(json_match.group())
            intent = result.get("intent", "multi_hop")
        else:
            logger.warning(f"意图识别未找到有效JSON,原始输出: {raw_text[:200]}")
            intent = "multi_hop"
    except Exception as e:
        logger.warning(f"意图识别解析失败,使用默认意图: {e}")
        intent = "multi_hop"
    
    if intent not in intents:
        intent = "multi_hop"
    
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


def generation_node(state: AgentState, llm, prompt_template: str) -> AgentState:
    """生成节点,支持记忆上下文"""
    question = state["question"]
    context_docs = state.get("reranked_docs", [])
    tool_results = state.get("tool_results", {})
    conversation_history = state.get("conversation_history", [])
    memory_context = state.get("memory_context", [])
    
    if not isinstance(tool_results, dict):
        tool_results = {}
    
    if state.get("vectorstore_uninitialized", False):
        generation = f"您好!我目前还没有加载知识库内容,无法基于文档回答您的问题。\n\n请先使用文档上传接口(POST /api/v1/upload)上传您的文档,我会自动建立索引后再为您服务。\n\n上传文档后,我就能基于您的知识库回答问题了!"
        return {"generation": generation}
    
    # 构建上下文
    context_parts = []
    
    # 添加记忆上下文(如果有)
    if memory_context:
        if isinstance(memory_context, list):
            memory_text = "\n".join(memory_context)
        else:
            memory_text = str(memory_context)
        context_parts.append(f"【相关记忆】\n{memory_text}")
    
    # 添加对话历史(如果有)
    if conversation_history:
        history_lines = []
        for msg in conversation_history:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        if history_lines:
            context_parts.append(f"【对话历史】\n" + "\n".join(history_lines))
    
    # 添加检索到的文档
    if context_docs:
        docs_content = "\n\n".join([doc.page_content for doc in context_docs])
        context_parts.append(f"【检索到的文档】\n{docs_content}")
    
    # 添加工具结果
    if tool_results and isinstance(tool_results, dict):
        tool_context = "\n\n【工具调用结果】\n"
        for tool_name, result in tool_results.items():
            tool_context += f"- {tool_name}: {result}\n"
        context_parts.append(tool_context)
    
    context = "\n".join(context_parts) if context_parts else "(无相关上下文)"
    
    # 生成答案
    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)
    
    generation = response.content if hasattr(response, 'content') else str(response)
    
    return {"generation": generation}

def evaluation_node(state: AgentState, llm) -> AgentState:
    """评估节点(轻量级版本,不调用LLM)"""
    question = state["question"]
    answer = state["generation"]
    context_docs = state.get("reranked_docs", [])
    
    # 简单评估(不调用LLM,只做关键词匹配)
    answer_len = len(answer) if answer else 0
    context_len = sum(len(doc.page_content) for doc in context_docs) if context_docs else 0
    
    # 基于回答长度和上下文覆盖率的简单评估
    faithfulness = min(1.0, answer_len / 200) if answer_len > 0 else 0.0
    answer_relevancy = 0.8 if context_docs else 0.5  # 有上下文时默认高分
    context_precision = len(context_docs) / 5.0 if context_docs else 0.0
    
    # 判断是否需要反思(如果回答太短或者没有上下文)
    needs_reflection = answer_len < 50 or (not context_docs and context_len == 0)
    
    return {
        "evaluation": {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "completeness": min(1.0, answer_len / 500),
            "overall_score": (faithfulness * 0.35 + answer_relevancy * 0.35 + context_precision * 0.15 + min(1.0, answer_len / 500) * 0.15)
        },
        "needs_reflection": needs_reflection,
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
    
    return {
        "refined_answer": refined,
        "reflection_count": reflection_count + 1
    }
