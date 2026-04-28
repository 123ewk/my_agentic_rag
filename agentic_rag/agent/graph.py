"""
LangGraph状态机图构建
"""
import re
from typing import Dict, Any, AsyncIterator, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
import asyncio
import json
import time

from .state import AgentState
from ..memory.intent_cache import get_intent_cache
from ..memory.gen_cache import get_generation_cache
from ..config.settings import get_settings
from ..retrieval.query_rewrite import _clean_think_tags
from .nodes import (
    intent_classification_node,
    query_rewrite_node,
    retrieval_node,
    parallel_retrieval_node,
    rerank_node,
    tool_call_node,
    generation_node,
    evaluation_node,
    reflection_node
)
from .edges import (
    route_after_intent,
    route_after_rewrite,
    route_after_evaluation,
    route_after_reflection,
    route_after_generation,
    route_after_tool_call,
    route_after_rerank
)

class AgenticRAGGraph:
    """Agentic RAG状态机"""

    def __init__(
        self,
        llm: ChatOpenAI,
        embeddings,
        vectorstore,
        reranker,
        tools: Dict[str, Any],
        prompt_template: str,
        short_term_memory=None,
        long_term_memory=None
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.tools = tools
        self.prompt_template = prompt_template
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory

        settings = get_settings()
        self.intent_cache = get_intent_cache(
            max_size=settings.intent_cache_max_size,
            ttl_seconds=settings.intent_cache_ttl
        )
        self.gen_cache = get_generation_cache(
            max_size=settings.generation_cache_max_size,
            ttl_seconds=settings.generation_cache_ttl
        )

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph[AgentState]:
        """构建状态机图"""
        graph = StateGraph(AgentState)
        

        # LangGraph 对节点处理函数有一个硬性要求：节点函数只能接收一个参数，也就是 state。
        # 所以 lambda 就是用来解决这个问题的！这个s就是state。

        # 添加节点
        graph.add_node("intent_classification",
                      lambda s: intent_classification_node(s, self.llm, self.intent_cache))
        graph.add_node("query_rewrite", 
                      lambda s: query_rewrite_node(s, self.llm, self.embeddings))
        graph.add_node("retrieval", 
                      lambda s: retrieval_node(s, self.vectorstore))
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
        
        # 设置入口点
        graph.set_entry_point("intent_classification")

        # 添加边
        graph.add_conditional_edges(
            "intent_classification", # 起点节点
            route_after_intent,      # 路由函数
            {                        # 路由映射表:  "返回值" -> "目标节点名称"
                "tool_call": "tool_call", 
                "query_rewrite": "query_rewrite",
                "generation": "generation"
            }
        )

        graph.add_conditional_edges(
            "query_rewrite",
            route_after_rewrite,
            {
                "retrieval": "retrieval",
                "rerank": "rerank"
            }
        )

        graph.add_conditional_edges(
            "rerank",
            route_after_rerank,
            {
                "tool_call": "tool_call",
                "generation": "generation"
            }
        )

        graph.add_conditional_edges(
            "tool_call",
            route_after_tool_call,
            {
                "generation": "generation"
            }
        )

        graph.add_conditional_edges(
            "generation",
            route_after_generation,
            {
                "evaluation": "evaluation",
            }
        )
        
        graph.add_conditional_edges(
            "evaluation",
            route_after_evaluation,
            {
                "reflection": "reflection",
                "__end__": END
            }
        )
        
        graph.add_conditional_edges(
            "reflection",
            route_after_reflection,
            {
                "evaluation": "evaluation",
                "__end__": END
            }
        )
        # 编译图
        return graph.compile()

    def invoke(self, question: str, **kwargs) -> Dict[str, Any]:
        """执行Agent，支持短期记忆和长期记忆"""
        import asyncio
        
        session_id = kwargs.get("session_id")
        user_id = kwargs.get("user_id")
        
        initial_state = {
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
            "memory_context": [],
            "conversation_history": [],
            "reflection_count": 0,
            "error": None,
            "metadata": kwargs
        }
        
        loop = asyncio.new_event_loop() # 创建一个全新的、独立的 asyncio 事件循环对象。
        asyncio.set_event_loop(loop) # 把刚创建的 loop 设置为当前线程的默认事件循环。
        try:
            # 同步代码会暂停，等异步的短期记忆加载完成，才继续往下走。
            if self.short_term_memory and session_id:
                loop.run_until_complete(self._load_short_term_memory(initial_state, session_id))
            # 同步代码会暂停，等异步的长期记忆搜索完成，才继续往下走。
            if self.long_term_memory and user_id:
                loop.run_until_complete(self._search_long_term_memory(initial_state, user_id, question))
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
            from loguru import logger
            logger.warning(f"加载短期记忆失败: {e}")
    
    async def _search_long_term_memory(self, state: Dict, user_id: str, query: str):
        """搜索长期记忆"""
        try:
            memories = await self.long_term_memory.search(user_id, query)
            state["memory_context"] = [m["content"] for m in memories]
        except Exception as e:
            from loguru import logger
            logger.warning(f"搜索长期记忆失败: {e}")
    
    def _build_memory_content(self, state: Dict, question: str) -> Optional[str]:
        """
        构建可用于长期记忆的内容
        
        Args:
            state: Agent状态字典
            question: 用户问题
            
        Returns:
            格式化后的记忆内容，如果无需保存则返回None
        """
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
    
    async def stream_invoke(
        self,
        question: str,
        session_id: str = None,
        user_id: str = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式执行Agent,支持实时返回生成内容和记忆

        参数：
            question: 用户问题
            session_id: 会话ID
            user_id: 用户ID
            **kwargs: 其他参数（use_tools, temperature等）

        产出：
            Dict[str, Any]: 流式事件，包含type和content字段
                - type="status": 状态更新（如"意图识别"、"检索中"等）
                - type="chunk": 生成的内容块
                - type="sources": 检索到的文档
                - type="metrics": 评估指标
                - type="done": 完成信号
        """
        initial_state = {
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
            "memory_context": [],
            "conversation_history": [],
            "reflection_count": 0,
            "error": None,
            "metadata": kwargs
        }

        # 加载短期记忆
        if self.short_term_memory and session_id:
            try:
                messages = await self.short_term_memory.get_message(session_id)
                history = []
                for msg in messages:
                    role = "user" if hasattr(msg, "type") and msg.type == "human" else "assistant"
                    history.append({"role": role, "content": msg.content})
                initial_state["conversation_history"] = history

                context = await self.short_term_memory.get_context(session_id)
                initial_state["memory_context"] = context.split("\n") if context else []
            except Exception as e:
                logger.warning(f"加载短期记忆失败: {e}")
        
        # 搜索长期记忆
        if self.long_term_memory and user_id:
            try:
                memories = await self.long_term_memory.search(user_id, question)
                if memories:
                    existing_context = initial_state.get("memory_context", [])
                    initial_state["memory_context"] = existing_context + [m["content"] for m in memories]
            except Exception as e:
                logger.warning(f"搜索长期记忆失败: {e}")

        # 0. 生成缓存快速路径检查（在意图分类之前，避免不必要的LLM调用）
        settings = get_settings()
        if settings.generation_cache_enabled:
            cached_gen = self.gen_cache.get(question, None)
            if cached_gen:
                cached_response = cached_gen.get("response", "")
                logger.info(f"生成缓存命中（快速路径）: {question[:50]}...")

                for i in range(0, len(cached_response), 50):
                    chunk = cached_response[i:i+50]
                    if chunk:
                        yield {
                            "type": "chunk",
                            "content": chunk,
                            "data": {"partial_response": cached_response[:i+len(chunk)], "cached": True}
                        }
                        # 模拟LLM生成时间
                        await asyncio.sleep(0.05)

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

        # 1. 意图分类阶段
        yield {
            "type": "status",
            "content": "正在分析问题意图...",
            "data": {"stage": "intent_classification"}
        }
        state = intent_classification_node(initial_state, self.llm, self.intent_cache)
        initial_state.update(state)

        # 1.5 生成缓存检查（基于意图的二次检查）
        if settings.generation_cache_enabled:
            cached_gen = self.gen_cache.get(question, initial_state.get("intent"))
            if cached_gen:
                cached_response = cached_gen.get("response", "")
                logger.info(f"生成缓存命中（意图过滤）: {question[:50]}...")

                for i in range(0, len(cached_response), 50):
                    chunk = cached_response[i:i+50]
                    if chunk:
                        yield {
                            "type": "chunk",
                            "content": chunk,
                            "data": {"partial_response": cached_response[:i+len(chunk)], "cached": True}
                        }
                        await asyncio.sleep(0.05)

                yield {
                    "type": "done",
                    "content": "回答生成完成(缓存)",
                    "data": {
                        "session_id": session_id,
                        "user_id": user_id,
                        "cached": True,
                        "intent": initial_state.get("intent"),
                        "reflection_count": 0,
                        "gen_cache_hit": True
                    }
                }
                return

        # 2. 查询改写阶段（对于factual类型跳过，因为事实查询不需要复杂改写）
        intent = initial_state.get("intent", "")
        if intent == "factual":
            # factual类型直接使用原始问题作为查询
            initial_state["rewritten_queries"] = [question]
            logger.info(f"factual类型问题，跳过查询改写")
        else:
            yield {
                "type": "status",
                "content": "正在改写查询...",
                "data": {"stage": "query_rewrite"}
            }
            state = query_rewrite_node(initial_state, self.llm, self.embeddings)
            initial_state.update(state)

        # 3. 检索阶段(使用并行检索优化)
        yield {
            "type": "status",
            "content": "正在检索相关文档...",
            "data": {"stage": "retrieval"}
        }
        state = parallel_retrieval_node(initial_state, self.vectorstore)
        initial_state.update(state)
        
        # 4. 重排阶段（factual类型跳过重排API调用，直接使用检索结果）
        if intent == "factual":
            # factual类型直接使用检索结果（按相关性排序即可）
            initial_state["reranked_docs"] = initial_state.get("retrieved_docs", [])[:3]
            logger.info(f"factual类型问题，跳过重排")
        else:
            yield {
                "type": "status",
                "content": "正在优化文档排序...",
                "data": {"stage": "rerank"}
            }
            state = rerank_node(initial_state, self.reranker)
            initial_state.update(state)
        
        # 发送检索到的文档
        if initial_state.get("reranked_docs"):
            docs_info = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score")
                }
                for doc in initial_state["reranked_docs"][:3]
            ]
            yield {
                "type": "sources",
                "content": "检索到相关文档",
                "data": {"documents": docs_info}
            }
        
        # 5. 工具调用阶段（如果需要）
        # 当意图为 tool_call 时，自动调用工具
        if initial_state.get("intent") == "tool_call":
            yield {
                "type": "status",
                "content": "正在调用工具...",
                "data": {"stage": "tool_call"}
            }
            state = tool_call_node(initial_state, self.llm, self.tools)
            initial_state.update(state)
        
        # 6. 生成阶段（流式）
        yield {
            "type": "status",
            "content": "正在生成回答...",
            "data": {"stage": "generation"}
        }
        
        # 构建上下文（包含记忆）
        context_parts = []
        memory_context = initial_state.get("memory_context", [])
        conversation_history = initial_state.get("conversation_history", [])
        context_docs = initial_state.get("reranked_docs", [])
        tool_results = initial_state.get("tool_results", {})
        
        if not isinstance(tool_results, dict):
            tool_results = {}
        
        # 添加记忆上下文
        if memory_context:
            if isinstance(memory_context, list):
                memory_text = "\n".join(memory_context)
            else:
                memory_text = str(memory_context)
            context_parts.append(f"【相关记忆】\n{memory_text}")
        
        # 添加对话历史
        if conversation_history:
            history_lines = []
            for msg in conversation_history:
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            if history_lines:
                context_parts.append(f"【对话历史】\n" + "\n".join(history_lines))
        
        # 添加检索文档
        if context_docs:
            docs_content = "\n\n".join([doc.page_content for doc in context_docs])
            context_parts.append(f"【检索到的文档】\n{docs_content}")
        
        # 添加工具结果
        if tool_results:
            tool_context = "\n\n【工具调用结果】\n"
            for tool_name, result in tool_results.items():
                tool_context += f"- {tool_name}: {result}\n"
            context_parts.append(tool_context)

        # 上下文截断
        settings = get_settings()
        if settings.context_truncation_enabled:
            from .nodes import _truncate_context, _estimate_tokens
            context_parts = _truncate_context(
                context_parts,
                max_tokens=settings.max_context_tokens,
                max_docs=settings.max_docs_for_context
            )

        context = "\n".join(context_parts) if context_parts else "（无相关上下文）"

        prompt = self.prompt_template.format(context=context, question=question)

        temperature = kwargs.get("temperature", 0.7)
        self.llm.temperature = temperature

        full_response = ""
        buffer = ""

        async for chunk in self.llm.astream(prompt):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            buffer += content

            if len(buffer) < 50 and not content.endswith('</think'):
                continue

            content = buffer
            buffer = ""

            if not content.strip():
                continue

            full_response += content

            think_pattern = r'<think\b[^>]*>.*?</think\s*>'
            has_think = bool(re.search(think_pattern, full_response, re.DOTALL))
            cleaned_preview = re.sub(think_pattern, '', content, flags=re.DOTALL) if has_think else content

            yield {
                "type": "chunk",
                "content": cleaned_preview,
                "data": {
                    "partial_response": full_response,
                    "has_think": has_think
                }
            }

        if buffer:
            if buffer.strip():
                full_response += buffer
                think_pattern = r'<think\b[^>]*>.*?</think\s*>'
                has_think = bool(re.search(think_pattern, full_response, re.DOTALL))
                cleaned_buffer = re.sub(think_pattern, '', buffer, flags=re.DOTALL) if has_think else buffer

                yield {
                    "type": "chunk",
                    "content": cleaned_buffer,
                    "data": {
                        "partial_response": full_response,
                        "has_think": has_think
                    }
                }

        think_pattern = r'<think\b[^>]*>(.*?)</think\s*>'
        think_matches = re.findall(think_pattern, full_response, re.DOTALL)
        think_content = [match.strip() for match in think_matches if match.strip()]

        logger.info(f"流式生成完成 - has_think: {bool(think_content)}, think_content数量: {len(think_content)}")
        if think_content:
            logger.info(f"思考内容预览: {think_content[:2] if think_content else []}")

        cleaned_for_cache = _clean_think_tags(full_response)
        self.gen_cache.set(
            question,
            cleaned_for_cache,
            intent=initial_state.get("intent"),
            metadata={"cached_at": time.time()}
        )

        initial_state["generation"] = cleaned_for_cache
        
        # 7. 评估阶段
        yield {
            "type": "status",
            "content": "正在评估回答质量...",
            "data": {"stage": "evaluation"}
        }
        state = evaluation_node(initial_state, self.llm)
        initial_state.update(state)
        
        if initial_state.get("evaluation"):
            yield {
                "type": "metrics",
                "content": "评估完成",
                "data": initial_state["evaluation"]
            }
        
        # 8. 反思阶段（如果需要）
        if initial_state.get("needs_reflection", False):
            max_reflection = kwargs.get("max_reflection_steps", 2)
            if initial_state.get("reflection_count", 0) < max_reflection:
                yield {
                    "type": "status",
                    "content": "正在进行反思优化...",
                    "data": {"stage": "reflection"}
                }
                state = reflection_node(initial_state, self.llm)
                initial_state.update(state)
        
        # 9. 保存对话到短期记忆
        if self.short_term_memory and session_id:
            try:
                await self.short_term_memory.add_message(
                    session_id=session_id,
                    question=question,
                    answer=initial_state.get("generation", ""),
                    metadata={"intent": initial_state.get("intent")}
                )
            except Exception as e:
                logger.warning(f"保存对话到短期记忆失败: {e}")
        
        # 9.5 保存重要内容到长期记忆
        if self.long_term_memory and user_id:
            try:
                memory_content = self._build_memory_content(initial_state, question)
                if memory_content:
                    await self.long_term_memory.save_memory(
                        user_id=user_id,
                        content=memory_content,
                        session_id=session_id,
                        metadata={
                            "intent": initial_state.get("intent"),
                            "reflection_count": initial_state.get("reflection_count", 0),
                            "tools_used": list(initial_state.get("tool_results", {}).keys())
                        }
                    )
                    logger.info(f"已保存对话到长期记忆 (user_id: {user_id})")
            except Exception as e:
                logger.warning(f"保存对话到长期记忆失败: {e}")
        
        # 10. 完成
        yield {
            "type": "done",
            "content": "回答生成完成",
            "data": {
                "session_id": session_id,
                "user_id": user_id,
                "intent": initial_state.get("intent"),
                "reflection_count": initial_state.get("reflection_count", 0),
                "tools_used": list(initial_state.get("tool_results", {}).keys()),
                "think_content": think_content,
                "has_think": bool(think_content)
            }
        }
