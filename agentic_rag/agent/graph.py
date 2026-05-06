"""
LangGraph状态机图构建

支持两种模式：
1. DAG模式（use_react=False）: 传统LangGraph StateGraph工作流
2. ReAct模式（use_react=True）: 基于Observe→Think→Act循环的真正Agent系统

性能优化记录：
- 优化A：合并intent+rewrite为单次LLM调用（节省3-5s）
- 优化C：generation快速退出（高质量答案跳过evaluation）
- 优化E：DAG模式真流式generation（llm.astream替代伪流式）
- 优化F：reflection默认关闭，反思后直接END
- 优化G：记忆并行加载（asyncio.gather）
"""
import re
from typing import Dict, Any, AsyncIterator, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from loguru import logger
import asyncio
import time

from .state import AgentState
from .react import ReActAgent
from ..memory.intent_cache import get_intent_cache
from ..memory.gen_cache import get_generation_cache
from ..config.settings import get_settings
from ..retrieval.query_rewrite import _clean_think_tags
from .nodes import (
    intent_classification_node,
    intent_and_rewrite_node,
    query_rewrite_node,
    parallel_retrieval_node,
    rerank_node,
    tool_call_node,
    generation_node,
    generation_node_stream,
    evaluation_node,
    reflection_node,
    web_search_node,
    should_skip_evaluation,
    _build_context_parts,
)
from ..execution.parallel_executor import ParallelExecutor
from .edges import (
    route_after_intent,
    route_after_evaluation,
    route_after_reflection,
    route_after_generation,
    route_after_tool_call,
    route_after_rerank,
    route_after_web_search
)


class AgenticRAGGraph:
    """
    Agentic RAG状态机
    
    支持两种执行模式：
    - DAG模式（use_react=False）: 传统LangGraph StateGraph工作流，
      路径由硬编码route_after_*决定
    - ReAct模式（use_react=True）: 基于Observe→Think→Act循环的真正Agent，
      LLM自主决策每一步行动
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        embeddings,
        vectorstore,
        reranker,
        tools: Dict[str, Any],
        prompt_template: str,
        short_term_memory=None,
        long_term_memory=None,
        use_react: bool = True,
    ):
        """
        初始化Agent

        参数:
            llm: 大语言模型实例
            embeddings: 嵌入模型实例
            vectorstore: 向量存储实例
            reranker: 重排模型实例
            tools: 外部工具字典
            prompt_template: 生成回答的提示词模板
            short_term_memory: 短期记忆管理器
            long_term_memory: 长期记忆管理器
            use_react: 是否使用ReAct模式（默认True）
        """
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.tools = tools
        self.prompt_template = prompt_template
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.use_react = use_react

        settings = get_settings()
        self.intent_cache = get_intent_cache(
            max_size=settings.intent_cache_max_size,
            ttl_seconds=settings.intent_cache_ttl
        )
        self.gen_cache = get_generation_cache(
            max_size=settings.generation_cache_max_size,
            ttl_seconds=settings.generation_cache_ttl
        )

        if use_react:
            self._react_agent = ReActAgent(
                llm=llm,
                embeddings=embeddings,
                vectorstore=vectorstore,
                reranker=reranker,
                tools=tools,
                prompt_template=prompt_template,
                short_term_memory=short_term_memory,
                long_term_memory=long_term_memory,
            )
            logger.info("已启用ReAct Agent模式")
        else:
            self._react_agent = None
            self.graph = self._build_graph()
            logger.info("已启用DAG工作流模式")

        # 初始化并行执行器（优化H：最大化并行执行）
        self._parallel_executor = ParallelExecutor(
            vectorstore=vectorstore,
            reranker=reranker,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
            web_search_tool=tools.get("duckduckgo"),
        )
        logger.info("并行执行器初始化完成")

    def _build_graph(self) -> StateGraph[AgentState]:
        """
        构建状态机图（优化A：使用合并节点替代分离的intent+rewrite）

        原流程：intent_classification → route → query_rewrite → retrieval
        优化后：intent_and_rewrite → route → retrieval（减少2-3次LLM调用）
        """
        graph = StateGraph(AgentState)
        
        # 优化A：使用合并节点替代分离的intent_classification + query_rewrite
        graph.add_node("intent_and_rewrite",
                      lambda s: intent_and_rewrite_node(s, self.llm, self.intent_cache))
        # 保留原节点供兼容
        graph.add_node("intent_classification",
                      lambda s: intent_classification_node(s, self.llm, self.intent_cache))
        graph.add_node("query_rewrite", 
                      lambda s: query_rewrite_node(s, self.llm, self.embeddings))
        graph.add_node("retrieval", 
                      lambda s: parallel_retrieval_node(s, self.vectorstore))
        graph.add_node("rerank", 
                      lambda s: rerank_node(s, self.reranker))
        graph.add_node("tool_call", 
                      lambda s: tool_call_node(s, self.llm, self.tools))
        graph.add_node("generation", 
                      lambda s: generation_node(s, self.llm, self.prompt_template))
        graph.add_node("evaluation", 
                      lambda s: evaluation_node(s, self.llm))
        graph.add_node("reflection", 
                      lambda s: reflection_node(s, self.llm))
        graph.add_node("web_search",
                      lambda s: web_search_node(s, self.llm))
        
        # 优化A：入口改为合并节点
        graph.set_entry_point("intent_and_rewrite")

        # 合并节点后的路由（同时输出intent和rewritten_queries）
        graph.add_conditional_edges(
            "intent_and_rewrite",
            self._route_after_intent_and_rewrite,
            {
                "tool_call": "tool_call", 
                "retrieval": "retrieval",
                "generation": "generation",
            }
        )

        # 查询改写后直接进入检索（保留原路径兼容）
        graph.add_edge("query_rewrite", "retrieval")

        # 并行检索完成后直接进入重排
        graph.add_edge("retrieval", "rerank")

        # 重排后的路由
        graph.add_conditional_edges(
            "rerank",
            route_after_rerank,
            {
                "tool_call": "tool_call",
                "generation": "generation"
            }
        )

        # 工具调用后的路由
        graph.add_conditional_edges(
            "tool_call",
            route_after_tool_call,
            {
                "generation": "generation"
            }
        )

        # 优化C：生成后的路由（高质量答案快速退出）
        graph.add_conditional_edges(
            "generation",
            route_after_generation,
            {
                "evaluation": "evaluation",
                "__end__": END
            }
        )
        
        # 评估后的路由（包含CRAG）
        graph.add_conditional_edges(
            "evaluation",
            self._route_after_evaluation_with_crag,
            {
                "reflection": "reflection",
                "web_search": "web_search",
                "__end__": END
            }
        )
        
        # 优化F：反思后直接END（不再回evaluation循环）
        graph.add_conditional_edges(
            "reflection",
            route_after_reflection,
            {
                "__end__": END
            }
        )
        
        # 网络搜索后的路由
        graph.add_conditional_edges(
            "web_search",
            route_after_web_search,
            {
                "generation": "generation",
                "__end__": END
            }
        )
        
        return graph.compile()

    def _route_after_intent_and_rewrite(self, state: AgentState):
        """
        合并节点后的路由（优化A）

        合并节点已同时输出intent和rewritten_queries，
        根据intent直接路由到对应节点，无需再经过query_rewrite。
        """
        intent = state.get("intent", "factual")
        
        if intent == "tool_call":
            return "tool_call"
        elif intent in ("multi_hop", "reasoning", "factual"):
            # 这些意图都需要检索，且rewritten_queries已在合并节点中生成
            return "retrieval"
        elif intent == "summary":
            return "generation"
        else:
            return "generation"

    def _route_after_evaluation_with_crag(self, state: AgentState):
        """
        评估后的路由（支持CRAG，带循环守卫）
        
        防止 evaluation→web_search→generation→evaluation 无限循环：
        - crag_loop_count 记录CRAG循环次数
        - 超过 MAX_CRAG_LOOPS 后不再触发web_search
        """
        from .edges import MAX_CRAG_LOOPS
        settings = get_settings()
        confidence_level = state.get("confidence_level", "high")
        needs_reflection = state.get("needs_reflection", False)
        reflection_count = state.get("reflection_count", 0)
        max_reflection = state.get("metadata", {}).get("max_reflection_steps", 0)
        crag_loop_count = state.get("crag_loop_count", 0)
        
        # CRAG: 低置信度触发网络搜索（带循环守卫）
        if settings.crag_enabled and confidence_level == "low" and crag_loop_count < MAX_CRAG_LOOPS:
            return "web_search"
        
        if needs_reflection and reflection_count < max_reflection:
            return "reflection"
        else:
            return "__end__"

    def invoke(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        执行Agent，支持短期记忆和长期记忆
        
        ReAct模式下委托给ReActAgent，DAG模式下使用LangGraph图执行
        """
        if self.use_react and self._react_agent:
            return self._react_agent.invoke(question, **kwargs)
        
        session_id = kwargs.get("session_id")
        user_id = kwargs.get("user_id")
        
        initial_state = self._create_initial_state(question, kwargs)
        
        # 优化G：记忆并行加载
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            memory_tasks = []
            if self.short_term_memory and session_id:
                memory_tasks.append(self._load_short_term_memory(initial_state, session_id))
            if self.long_term_memory and user_id:
                memory_tasks.append(self._search_long_term_memory(initial_state, user_id, question))
            if memory_tasks:
                loop.run_until_complete(asyncio.gather(*memory_tasks))
        finally:
            loop.close()
        
        result = self.graph.invoke(initial_state)
        
        if self.short_term_memory and session_id:
            answer = result.get("refined_answer") or result.get("generation", "")
            loop2 = asyncio.new_event_loop()
            asyncio.set_event_loop(loop2)
            try:
                loop2.run_until_complete(
                    self.short_term_memory.add_message(
                        session_id=session_id,
                        question=question,
                        answer=answer,
                        metadata={"intent": result.get("intent")}
                    )
                )
            finally:
                loop2.close()
        
        return result
    
    def _create_initial_state(self, question: str, kwargs: Dict) -> Dict[str, Any]:
        """创建初始状态"""
        return {
            "question": question,
            "intent": "",
            "rewritten_queries": [],
            "current_query_index": 0,
            "retrieved_docs": [],
            "reranked_docs": [],
            "generation": "",
            "refined_answer": "",
            "evaluation": {},
            "needs_reflection": False,
            "tool_results": {},
            "tool_calls": [],
            "tool_call_failed": False,
            "memory_context": [],
            "conversation_history": [],
            "reflection_count": 0,
            "previous_evaluation": None,
            "error": None,
            "metadata": kwargs,
            "confidence_score": None,
            "confidence_level": None,
            "needs_web_search": False,
            "search_results": [],
            "crag_loop_count": 0
        }
    
    async def _load_short_term_memory(self, state: Dict, session_id: str):
        """加载短期记忆"""
        try:
            messages = await self.short_term_memory.get_message(session_id)
            history = []
            for msg in messages:
                role = "user" if hasattr(msg, "type") and msg.type == "human" else "assistant"
                history.append({"role": role, "content": msg.content})
            state["conversation_history"] = history
            
            context = await self.short_term_memory.get_context(session_id)
            state["memory_context"] = context.split("\n") if context else []
        except Exception as e:
            logger.warning(f"加载短期记忆失败: {e}")
    
    async def _search_long_term_memory(self, state: Dict, user_id: str, query: str):
        """
        搜索长期记忆

        V2改进: 使用分层检索(search_with_decay)
        """
        try:
            # V2: 使用分层检索(带时间衰减和类型预筛)
            if hasattr(self.long_term_memory, 'search_with_decay'):
                memories = await self.long_term_memory.search_with_decay(
                    user_id=user_id,
                    query=query,
                )
            else:
                memories = await self.long_term_memory.search(user_id, query)

            if memories:
                memory_contents = []
                for m in memories:
                    summary = m.get("summary", "")
                    content = m.get("content", "")
                    mem_type = m.get("memory_type", "fact")
                    if summary:
                        memory_contents.append(f"[{mem_type}] {summary}: {content}")
                    else:
                        memory_contents.append(content)
                state["memory_context"] = memory_contents
            else:
                state["memory_context"] = []
        except Exception as e:
            logger.warning(f"搜索长期记忆失败: {e}")
    
    def _build_memory_content(self, state: Dict, question: str) -> Optional[str]:
        """构建可用于长期记忆的内容"""
        intent = state.get("intent", "")
        generation = state.get("generation", "")
        
        if not generation:
            return None
        
        memory_parts = []
        memory_parts.append(f"用户问题: {question}")
        memory_parts.append(f"意图: {intent}")
        
        reflection_count = state.get("reflection_count", 0)
        if reflection_count > 0:
            memory_parts.append(f"经过{reflection_count}轮反思优化")
        
        retrieved_docs = state.get("retrieved_docs", [])
        if retrieved_docs:
            doc_sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]
            memory_parts.append(f"参考文档: {', '.join(set(doc_sources))}")
        
        tool_results = state.get("tool_results", {})
        if tool_results:
            tool_names = list(tool_results.keys())
            memory_parts.append(f"使用工具: {', '.join(tool_names)}")
        
        memory_parts.append(f"最终回答: {generation[:500]}...")
        
        return "\n".join(memory_parts)

    async def _save_memories(self, state: Dict, question: str, session_id: str = None, user_id: str = None):
        """
        保存对话到短期和长期记忆

        V2改进: 长期记忆走价值评估+压缩+去重流程
        """
        if self.short_term_memory and session_id:
            try:
                await self.short_term_memory.add_message(
                    session_id=session_id,
                    question=question,
                    answer=state.get("generation", ""),
                    metadata={"intent": state.get("intent")}
                )
            except Exception as e:
                logger.warning(f"保存对话到短期记忆失败: {e}")

        if self.long_term_memory and user_id:
            try:
                answer = state.get("generation", "") or state.get("refined_answer", "")
                if answer:
                    # V2: 使用save_from_conversation走完整提取流程
                    if hasattr(self.long_term_memory, 'save_from_conversation'):
                        await self.long_term_memory.save_from_conversation(
                            user_id=user_id,
                            question=question,
                            answer=answer,
                            session_id=session_id,
                            context={
                                "intent": state.get("intent"),
                                "reflection_count": state.get("reflection_count", 0),
                                "tools_used": list(state.get("tool_results", {}).keys()),
                            },
                        )
                    else:
                        # 兼容V1
                        memory_content = self._build_memory_content(state, question)
                        if memory_content:
                            await self.long_term_memory.save_memory(
                                user_id=user_id,
                                content=memory_content,
                                session_id=session_id,
                                metadata={
                                    "intent": state.get("intent"),
                                    "reflection_count": state.get("reflection_count", 0),
                                    "tools_used": list(state.get("tool_results", {}).keys()),
                                },
                            )
                    logger.info(f"已保存对话到长期记忆 (user_id: {user_id})")
            except Exception as e:
                logger.warning(f"保存对话到长期记忆失败: {e}")

    async def stream_invoke(
        self,
        question: str,
        session_id: str = None,
        user_id: str = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式执行Agent（优化E：DAG模式支持真流式generation）

        架构说明：
        - 使用 graph.astream() 执行前置节点（intent、retrieval、rerank等）
        - 到达generation时，手动用llm.astream()进行真token级流式输出
        - evaluation/reflection等后续节点通过直接调用执行

        产出：
            Dict[str, Any]: 流式事件，包含type和content字段
                - type="status": 状态更新
                - type="token": token级真流式输出
                - type="sources": 检索到的文档
                - type="metrics": 评估指标
                - type="done": 完成信号
        """
        if self.use_react and self._react_agent:
            async for event in self._react_agent.stream_run(
                question=question,
                session_id=session_id,
                user_id=user_id,
                **kwargs,
            ):
                yield event
            return

        settings = get_settings()
        initial_state = self._create_initial_state(question, kwargs)
        state = initial_state.copy()

        # ===== 阶段1: 并行加载记忆 =====
        memory_tasks = []
        if self.short_term_memory and session_id:
            memory_tasks.append(self._load_short_term_memory_stream(state, session_id))
        if self.long_term_memory and user_id:
            memory_tasks.append(self._search_long_term_memory_stream(state, user_id, question))
        if memory_tasks:
            await asyncio.gather(*memory_tasks)

        # ===== 阶段2: 意图识别 + 查询改写（合并节点）=====
        yield {
            "type": "status",
            "content": "正在分析问题意图...",
            "data": {"stage": "intent_classification"}
        }
        intent_result = intent_and_rewrite_node(state, self.llm, self.intent_cache)
        state.update(intent_result)
        intent = state.get("intent", "factual")

        # 生成缓存检查
        if settings.generation_cache_enabled:
            cached_gen = self.gen_cache.get(question, intent) or self.gen_cache.get(question, None)
            if cached_gen:
                cached_response = cached_gen.get("response", "")
                logger.info(f"生成缓存命中: {question[:50]}...")
                for i in range(0, len(cached_response), 50):
                    chunk = cached_response[i:i+50]
                    if chunk:
                        yield {
                            "type": "token",
                            "content": chunk,
                            "data": {"partial_response": cached_response[:i+len(chunk)], "cached": True}
                        }
                        await asyncio.sleep(0.05)
                await self._save_memories(
                    {"generation": cached_response, "intent": cached_gen.get("intent", "unknown")},
                    question, session_id, user_id
                )
                yield {
                    "type": "done",
                    "content": "回答生成完成(缓存)",
                    "data": {
                        "session_id": session_id,
                        "user_id": user_id,
                        "cached": True,
                        "intent": cached_gen.get("intent", "unknown"),
                        "reflection_count": 0,
                        "gen_cache_hit": True
                    }
                }
                return

        # ===== 阶段3: 根据意图执行不同路径 =====
        if intent == "tool_call":
            yield {
                "type": "status",
                "content": "正在调用工具...",
                "data": {"stage": "tool_call"}
            }
            tool_result = tool_call_node(state, self.llm, self.tools)
            state.update(tool_result)

        elif intent in ("factual", "multi_hop", "reasoning"):
            # 检索路径
            yield {
                "type": "status",
                "content": "正在检索相关文档...",
                "data": {"stage": "retrieval"}
            }
            retrieval_result = parallel_retrieval_node(state, self.vectorstore)
            state.update(retrieval_result)

            yield {
                "type": "status",
                "content": "正在优化文档排序...",
                "data": {"stage": "rerank"}
            }
            rerank_result = rerank_node(state, self.reranker)
            state.update(rerank_result)

            if state.get("reranked_docs"):
                docs_info = [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                        "score": doc.metadata.get("score")
                    }
                    for doc in state["reranked_docs"][:3]
                ]
                yield {
                    "type": "sources",
                    "content": "检索到相关文档",
                    "data": {"documents": docs_info}
                }

        # summary意图直接跳到生成

        # ===== 阶段4: 真流式生成（核心优化！使用llm.astream替代伪流式）=====
        yield {
            "type": "status",
            "content": "正在生成回答...",
            "data": {"stage": "generation"}
        }

        context_parts = _build_context_parts(
            state.get("memory_context", []),
            state.get("conversation_history", []),
            state.get("reranked_docs", []),
            state.get("search_results", []),
            state.get("tool_results", {}),
            state.get("tool_call_failed", False),
            settings
        )
        context = "\n".join(context_parts) if context_parts else "(无相关上下文)"
        prompt = self.prompt_template.format(context=context, question=question)

        full_text = []
        async for chunk in self.llm.astream(prompt):
            token = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if token:
                full_text.append(token)
                yield {
                    "type": "token",
                    "content": token,
                    "data": {"partial_response": "".join(full_text)}
                }

        generation = "".join(full_text)
        generation = _clean_think_tags(generation)
        state["generation"] = generation

        # 写入生成缓存
        if settings.generation_cache_enabled:
            self.gen_cache.set(
                question,
                generation,
                intent=None,
                metadata={"cached_at": time.time(), "actual_intent": state.get("intent")}
            )

        # ===== 阶段5: 评估（轻量级，不阻塞）=====
        yield {
            "type": "status",
            "content": "正在评估回答质量...",
            "data": {"stage": "evaluation"}
        }
        eval_result = evaluation_node(state, self.llm)
        state.update(eval_result)

        if state.get("evaluation"):
            yield {
                "type": "metrics",
                "content": "评估完成",
                "data": state["evaluation"]
            }

        confidence_level = state.get("confidence_level", "high")
        overall_score = state.get("evaluation", {}).get("overall_score", 0.5)

        # CRAG低置信度处理
        crag_triggered = False
        if settings.crag_enabled and confidence_level == "low":
            crag_loop_count = state.get("crag_loop_count", 0)
            if crag_loop_count < 3:  # MAX_CRAG_LOOPS
                crag_triggered = True
                yield {
                    "type": "status",
                    "content": "置信度较低，正在搜索网络信息...",
                    "data": {"stage": "crag_web_search"}
                }
                web_search_result = web_search_node(state, self.llm)
                state.update(web_search_result)

                # 重新生成
                yield {
                    "type": "status",
                    "content": "正在基于搜索结果重新生成...",
                    "data": {"stage": "regeneration"}
                }
                context_parts = _build_context_parts(
                    state.get("memory_context", []),
                    state.get("conversation_history", []),
                    state.get("reranked_docs", []),
                    state.get("search_results", []),
                    state.get("tool_results", {}),
                    state.get("tool_call_failed", False),
                    settings
                )
                context = "\n".join(context_parts) if context_parts else "(无相关上下文)"
                prompt = self.prompt_template.format(context=context, question=question)

                full_text = []
                async for chunk in self.llm.astream(prompt):
                    token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    if token:
                        full_text.append(token)
                        yield {
                            "type": "token",
                            "content": token,
                            "data": {"partial_response": "".join(full_text), "regenerated": True}
                        }

                generation = "".join(full_text)
                generation = _clean_think_tags(generation)
                state["generation"] = generation

        # ===== 阶段6: 反思（如需要）=====
        reflection_count = state.get("reflection_count", 0)
        max_reflection = kwargs.get("max_reflection", 2)
        if state.get("needs_reflection", False) and reflection_count < max_reflection:
            yield {
                "type": "status",
                "content": "正在进行反思优化...",
                "data": {"stage": "reflection", "reflection_count": reflection_count}
            }
            reflection_result = reflection_node(state, self.llm)
            state.update(reflection_result)

            # 反思后重新生成
            yield {
                "type": "status",
                "content": "正在基于反思重新生成...",
                "data": {"stage": "regeneration"}
            }
            context_parts = _build_context_parts(
                state.get("memory_context", []),
                state.get("conversation_history", []),
                state.get("reranked_docs", []),
                state.get("search_results", []),
                state.get("tool_results", {}),
                state.get("tool_call_failed", False),
                settings
            )
            context = "\n".join(context_parts) if context_parts else "(无相关上下文)"
            prompt = self.prompt_template.format(context=context, question=question)

            full_text = []
            async for chunk in self.llm.astream(prompt):
                token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if token:
                    full_text.append(token)
                    yield {
                        "type": "token",
                        "content": token,
                        "data": {"partial_response": "".join(full_text), "regenerated": True}
                    }

            generation = "".join(full_text)
            generation = _clean_think_tags(generation)
            state["generation"] = generation

        # ===== 阶段7: 保存记忆 + 完成 =====
        await self._save_memories(state, question, session_id, user_id)

        think_pattern = r'<think\b[^>]*>(.*?)</think\s*>'
        think_matches = re.findall(think_pattern, "".join(full_text), re.DOTALL)
        think_content = [match.strip() for match in think_matches if match.strip()]

        yield {
            "type": "done",
            "content": "回答生成完成",
            "data": {
                "session_id": session_id,
                "user_id": user_id,
                "intent": state.get("intent"),
                "reflection_count": state.get("reflection_count", 0),
                "tools_used": list(state.get("tool_results", {}).keys()),
                "think_content": think_content,
                "has_think": bool(think_content),
                "confidence_score": state.get("confidence_score"),
                "confidence_level": state.get("confidence_level"),
                "overall_score": state.get("evaluation", {}).get("overall_score"),
                "crag_triggered": crag_triggered
            }
        }

    async def _load_short_term_memory_stream(self, state: Dict, session_id: str):
        """流式模式下加载短期记忆"""
        try:
            messages = await self.short_term_memory.get_message(session_id)
            history = []
            for msg in messages:
                role = "user" if hasattr(msg, "type") and msg.type == "human" else "assistant"
                history.append({"role": role, "content": msg.content})
            state["conversation_history"] = history

            context = await self.short_term_memory.get_context(session_id)
            state["memory_context"] = context.split("\n") if context else []
        except Exception as e:
            logger.warning(f"加载短期记忆失败: {e}")

    async def _search_long_term_memory_stream(self, state: Dict, user_id: str, question: str):
        """流式模式下搜索长期记忆(V2: 分层检索)"""
        try:
            if hasattr(self.long_term_memory, 'search_with_decay'):
                memories = await self.long_term_memory.search_with_decay(
                    user_id=user_id,
                    query=question,
                )
            else:
                memories = await self.long_term_memory.search(user_id, question)

            if memories:
                existing_context = state.get("memory_context", [])
                memory_contents = []
                for m in memories:
                    summary = m.get("summary", "")
                    content = m.get("content", "")
                    mem_type = m.get("memory_type", "fact")
                    if summary:
                        memory_contents.append(f"[{mem_type}] {summary}: {content}")
                    else:
                        memory_contents.append(content)
                state["memory_context"] = existing_context + memory_contents
        except Exception as e:
            logger.warning(f"搜索长期记忆失败: {e}")

    async def fast_stream_invoke(
        self,
        question: str,
        session_id: str = None,
        user_id: str = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        快速流式执行Agent（核心优化：首token时间<2s）

        与stream_invoke的根本区别：
        - stream_invoke: graph.astream()节点级流式，generation节点内部llm.invoke()阻塞
          → 用户必须等generation节点完成才能看到第一个token
        - fast_stream_invoke: 手动执行前置节点，到达generation时直接用llm.astream()
          → 用户在LLM开始生成时立即看到第一个token

        架构变化：
        1. 前置阶段（intent→retrieval→rerank）手动执行，不走graph
        2. generation阶段用llm.astream()真流式，不走graph的generation_node
        3. evaluation/CRAG/reflection移到后台异步执行，不阻塞用户
        4. 记忆保存移到后台异步执行

        参数:
            question: 用户问题
            session_id: 会话ID
            user_id: 用户ID

        产出:
            Dict[str, Any]: 流式事件
                - type="status": 状态更新
                - type="token": 生成内容的token级流式（真正的逐token输出）
                - type="sources": 检索到的文档
                - type="done": 完成信号
        """
        settings = get_settings()
        state = self._create_initial_state(question, kwargs)

        # ===== 阶段1: 并行加载记忆（与原stream_invoke相同）=====
        memory_tasks = []
        if self.short_term_memory and session_id:
            memory_tasks.append(self._load_short_term_memory_stream(state, session_id))
        if self.long_term_memory and user_id:
            memory_tasks.append(self._search_long_term_memory_stream(state, user_id, question))
        if memory_tasks:
            await asyncio.gather(*memory_tasks)

        # ===== 快速路径: 生成缓存检查 =====
        if settings.generation_cache_enabled:
            cached_gen = self.gen_cache.get(question, None)
            if cached_gen:
                cached_response = cached_gen.get("response", "")
                logger.info(f"快速流式-生成缓存命中: {question[:50]}...")
                for i in range(0, len(cached_response), 50):
                    chunk = cached_response[i:i+50]
                    if chunk:
                        yield {
                            "type": "token",
                            "content": chunk,
                            "data": {"partial_response": cached_response[:i+len(chunk)], "cached": True}
                        }
                        await asyncio.sleep(0.05)
                asyncio.create_task(self._save_memories(
                    {"generation": cached_response, "intent": cached_gen.get("intent", "unknown")},
                    question, session_id, user_id
                ))
                yield {
                    "type": "done",
                    "content": "回答生成完成(缓存)",
                    "data": {
                        "session_id": session_id,
                        "user_id": user_id,
                        "cached": True,
                        "intent": cached_gen.get("intent", "unknown"),
                        "reflection_count": 0,
                        "gen_cache_hit": True,
                    }
                }
                return

        # ===== 阶段2: 意图识别+查询改写（1次LLM调用）=====
        yield {
            "type": "status",
            "content": "正在分析问题...",
            "data": {"stage": "intent_classification"}
        }
        intent_result = intent_and_rewrite_node(state, self.llm, self.intent_cache)
        state.update(intent_result)
        intent = state.get("intent", "factual")
        yield {
            "type": "status",
            "content": f"意图识别完成: {intent}",
            "data": {"stage": "intent_classification", "intent": intent}
        }

        # 优化缓存检查：意图识别完成后，用 (question, intent) 精确查找缓存
        if settings.generation_cache_enabled:
            # 先尝试精确匹配 (question + intent)
            cached_gen = self.gen_cache.get(question, intent)
            if not cached_gen:
                # 回退到 (question, None) 兼容老缓存
                cached_gen = self.gen_cache.get(question, None)
            if cached_gen:
                cached_response = cached_gen.get("response", "")
                logger.info(f"快速流式-生成缓存命中(意图过滤): {question[:50]}...")
                for i in range(0, len(cached_response), 50):
                    chunk = cached_response[i:i+50]
                    if chunk:
                        yield {
                            "type": "token",
                            "content": chunk,
                            "data": {"partial_response": cached_response[:i+len(chunk)], "cached": True}
                        }
                        await asyncio.sleep(0.05)
                asyncio.create_task(self._save_memories(
                    {"generation": cached_response, "intent": intent},
                    question, session_id, user_id
                ))
                yield {
                    "type": "done",
                    "content": "回答生成完成(缓存)",
                    "data": {
                        "session_id": session_id,
                        "user_id": user_id,
                        "cached": True,
                        "intent": intent,
                        "reflection_count": 0,
                        "gen_cache_hit": True,
                    }
                }
                return

        # ===== 阶段3: 根据意图执行不同路径 =====
        if intent == "tool_call":
            yield {
                "type": "status",
                "content": "正在调用工具...",
                "data": {"stage": "tool_call"}
            }
            tool_result = tool_call_node(state, self.llm, self.tools)
            state.update(tool_result)

        elif intent in ("factual", "multi_hop", "reasoning"):
            # 检索路径
            yield {
                "type": "status",
                "content": "正在检索相关文档...",
                "data": {"stage": "retrieval"}
            }
            retrieval_result = parallel_retrieval_node(state, self.vectorstore)
            state.update(retrieval_result)

            yield {
                "type": "status",
                "content": "正在优化文档排序...",
                "data": {"stage": "rerank"}
            }
            rerank_result = rerank_node(state, self.reranker)
            state.update(rerank_result)

            if state.get("reranked_docs"):
                docs_info = [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                        "score": doc.metadata.get("score")
                    }
                    for doc in state["reranked_docs"][:3]
                ]
                yield {
                    "type": "sources",
                    "content": "检索到相关文档",
                    "data": {"documents": docs_info}
                }

        # summary意图直接跳到生成

        # ===== 阶段4: 真流式生成（核心优化！）=====
        yield {
            "type": "status",
            "content": "正在生成回答...",
            "data": {"stage": "generation"}
        }

        context_parts = _build_context_parts(
            state.get("memory_context", []),
            state.get("conversation_history", []),
            state.get("reranked_docs", []),
            state.get("search_results", []),
            state.get("tool_results", {}),
            state.get("tool_call_failed", False),
            settings
        )
        context = "\n".join(context_parts) if context_parts else "(无相关上下文)"
        prompt = self.prompt_template.format(context=context, question=question)

        full_text = []
        async for chunk in self.llm.astream(prompt):
            token = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if token:
                full_text.append(token)
                yield {
                    "type": "token",
                    "content": token,
                    "data": {}
                }

        generation = "".join(full_text)
        generation = _clean_think_tags(generation)
        state["generation"] = generation

        # 写入生成缓存
        if settings.generation_cache_enabled:
            self.gen_cache.set(
                question,
                generation,
                intent=None,
                metadata={"cached_at": time.time(), "actual_intent": state.get("intent")}
            )

        # ===== 阶段5: 后台异步执行评估+记忆保存（不阻塞用户）=====
        asyncio.create_task(
            self._background_post_processing(state, question, session_id, user_id)
        )

        # ===== 完成 =====
        think_pattern = r'<think\b[^>]*>(.*?)</think\s*>'
        think_matches = re.findall(think_pattern, "".join(full_text), re.DOTALL)
        think_content = [match.strip() for match in think_matches if match.strip()]

        yield {
            "type": "done",
            "content": "回答生成完成",
            "data": {
                "session_id": session_id,
                "user_id": user_id,
                "intent": state.get("intent"),
                "reflection_count": 0,
                "tools_used": list(state.get("tool_results", {}).keys()),
                "think_content": think_content,
                "has_think": bool(think_content),
                "fast_stream": True,
            }
        }

    async def _background_post_processing(
        self,
        state: Dict,
        question: str,
        session_id: str = None,
        user_id: str = None
    ):
        """
        后台异步后处理（不阻塞用户感知的响应时间）

        包含：
        1. 轻量级评估（用于日志/监控，不影响用户）
        2. 记忆保存（短期+长期）
        3. CRAG判断（低置信度时记录日志，但不重新生成）

        为什么可以后台执行：
        - 评估结果不影响已输出的答案
        - 记忆保存是写操作，与读操作无关
        - CRAG的重新生成在快速流式模式下被禁用，
          因为"先给用户一个答案"比"给用户一个完美答案"更重要
        """
        try:
            # 轻量级评估（无LLM调用，纯规则计算）
            evaluation_result = evaluation_node(state, self.llm)
            state.update(evaluation_result)

            confidence_level = state.get("confidence_level", "high")
            overall_score = state.get("evaluation", {}).get("overall_score", 0.5)
            logger.info(
                f"后台评估完成: confidence={confidence_level}, score={overall_score:.3f}, "
                f"intent={state.get('intent')}"
            )

            # 保存记忆
            await self._save_memories(state, question, session_id, user_id)

        except Exception as e:
            logger.warning(f"后台后处理失败: {e}")

    async def ultra_fast_stream_invoke(
        self,
        question: str,
        session_id: str = None,
        user_id: str = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        超快速流式执行（终极并行优化）
        
        与 fast_stream_invoke 的区别：
        - fast_stream_invoke: 手动顺序执行前置节点
        - ultra_fast_stream_invoke: 使用 ParallelExecutor 并行 Memory + Retrieval
        
        优化点：
        1. Memory 加载与 Retrieval 完全并行（无等待）
        2. Vector + BM25 混合检索并行
        3. Multi-query 检索异步并发
        4. 后台 evaluation + memory save
        
        参数:
            question: 用户问题
            session_id: 会话 ID
            user_id: 用户 ID
            
        产出:
            Dict[str, Any]: 流式事件
        """
        settings = get_settings()
        
        # ===== 阶段1: 意图识别 + 查询改写 =====
        initial_state = self._create_initial_state(question, kwargs)
        
        yield {
            "type": "status",
            "content": "正在分析问题...",
            "data": {"stage": "intent_classification"}
        }
        intent_result = intent_and_rewrite_node(initial_state, self.llm, self.intent_cache)
        initial_state.update(intent_result)
        intent = initial_state.get("intent", "factual")
        rewritten_queries = initial_state.get("rewritten_queries", [question])
        
        yield {
            "type": "status",
            "content": f"意图识别完成: {intent}",
            "data": {"stage": "intent_classification", "intent": intent}
        }
        
        # ===== 快速路径：生成缓存检查 =====
        if settings.generation_cache_enabled:
            cached_gen = self.gen_cache.get(question, intent)
            if cached_gen:
                cached_response = cached_gen.get("response", "")
                logger.info(f"超快速流式-缓存命中: {question[:50]}...")
                
                for i in range(0, len(cached_response), 50):
                    chunk = cached_response[i:i+50]
                    if chunk:
                        yield {
                            "type": "token",
                            "content": chunk,
                            "data": {"partial_response": cached_response[:i+len(chunk)], "cached": True}
                        }
                        await asyncio.sleep(0.05)
                
                # 后台保存记忆
                asyncio.create_task(
                    self._save_memories(
                        {"generation": cached_response, "intent": intent},
                        question, session_id, user_id
                    )
                )
                
                yield {
                    "type": "done",
                    "content": "回答生成完成(缓存)",
                    "data": {
                        "session_id": session_id,
                        "user_id": user_id,
                        "cached": True,
                        "intent": intent,
                        "gen_cache_hit": True,
                        "ultra_fast": True,
                    }
                }
                return
        
        # ===== 阶段2: 并行执行 Memory + Retrieval + WebSearch =====
        parallel_result = None
        if intent in ("tool_call",):
            # 工具调用路径：直接执行工具
            yield {
                "type": "status",
                "content": "正在调用工具...",
                "data": {"stage": "tool_call"}
            }
            tool_result = tool_call_node(initial_state, self.llm, self.tools)
            initial_state.update(tool_result)
            
            state_for_gen = initial_state
        else:
            # 检索路径：并行加载所有数据
            yield {
                "type": "status",
                "content": "并行加载记忆和检索文档...",
                "data": {"stage": "parallel_loading"}
            }
            
            parallel_result = await self._parallel_executor.execute_full_pipeline(
                question=question,
                intent=intent,
                rewritten_queries=rewritten_queries,
                session_id=session_id,
                user_id=user_id,
            )
            
            # 合并结果到状态
            state_for_gen = initial_state.copy()
            state_for_gen["retrieved_docs"] = parallel_result.retrieved_docs
            state_for_gen["reranked_docs"] = parallel_result.retrieved_docs  # 已经是 reranked
            state_for_gen["memory_context"] = parallel_result.memory_context
            state_for_gen["conversation_history"] = parallel_result.conversation_history
            state_for_gen["search_results"] = parallel_result.search_results
            
            yield {
                "type": "status",
                "content": f"检索完成: {len(state_for_gen['reranked_docs'])} 条文档",
                "data": {
                    "stage": "retrieval_done",
                    "doc_count": len(state_for_gen["reranked_docs"]),
                    "parallel_times": parallel_result.execution_times,
                }
            }
        
        # ===== 阶段3: 真流式生成 =====
        yield {
            "type": "status",
            "content": "正在生成回答...",
            "data": {"stage": "generation"}
        }
        
        context_parts = _build_context_parts(
            state_for_gen.get("memory_context", []),
            state_for_gen.get("conversation_history", []),
            state_for_gen.get("reranked_docs", []),
            state_for_gen.get("search_results", []),
            state_for_gen.get("tool_results", {}),
            state_for_gen.get("tool_call_failed", False),
            settings
        )
        context = "\n".join(context_parts) if context_parts else "(无相关上下文)"
        prompt = self.prompt_template.format(context=context, question=question)
        
        full_text = []
        async for chunk in self.llm.astream(prompt):
            token = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if token:
                full_text.append(token)
                yield {
                    "type": "token",
                    "content": token,
                    "data": {}
                }
        
        generation = "".join(full_text)
        generation = _clean_think_tags(generation)
        state_for_gen["generation"] = generation
        
        # 写入缓存
        if settings.generation_cache_enabled:
            self.gen_cache.set(
                question,
                generation,
                intent=intent,
                metadata={"cached_at": time.time(), "actual_intent": intent}
            )
        
        # ===== 阶段4: 后台异步后处理 =====
        asyncio.create_task(
            self._background_post_processing(state_for_gen, question, session_id, user_id)
        )
        
        # ===== 完成 =====
        think_pattern = r'<think\b[^>]*>(.*?)</think\s*>'
        think_matches = re.findall(think_pattern, "".join(full_text), re.DOTALL)
        think_content = [match.strip() for match in think_matches if match.strip()]
        
        yield {
            "type": "done",
            "content": "回答生成完成",
            "data": {
                "session_id": session_id,
                "user_id": user_id,
                "intent": state_for_gen.get("intent"),
                "tools_used": list(state_for_gen.get("tool_results", {}).keys()),
                "think_content": think_content,
                "has_think": bool(think_content),
                "ultra_fast": True,
                "parallel_execution": True,
                "execution_times": parallel_result.execution_times if parallel_result else {},
            }
        }
