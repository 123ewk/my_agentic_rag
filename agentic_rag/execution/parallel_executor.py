"""
并行执行优化模块

实现 Agentic RAG 系统的最大化并行执行策略，包括：
1. Memory + Retrieval 并行加载
2. Multi-Query 检索异步并发（跨查询去重）
3. Web Search 与 Memory/Retrieval 三路并行
4. 后台后处理（evaluation/memory save）

设计原则：
- 不增加额外 LLM 调用
- 适合单机环境（使用 asyncio.to_thread 包装同步调用）
- 保持代码可读性和可维护性
- 异常隔离：单个任务失败不影响其他任务
"""
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from loguru import logger
import time


@dataclass
class ParallelExecutionResult:
    """并行执行结果容器"""
    retrieved_docs: List[Any] = field(default_factory=list)
    memory_context: List[str] = field(default_factory=list)
    conversation_history: List[Dict] = field(default_factory=list)
    search_results: List[Any] = field(default_factory=list)
    execution_times: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class ParallelExecutor:
    """
    并行执行器 - 负责协调所有可并行的任务

    核心优化：
    1. Memory 加载与 Retrieval 并行（无数据依赖）
    2. 多查询检索使用 asyncio.gather 并发，完成后统一去重
    3. Web Search 与 Memory/Retrieval 三路并行
    4. Rerank 串行依赖 retrieval 完成

    并行策略图：
    ┌──────────────────────────────────────────────┐
    │  Memory 加载    │   Retrieval    │  WebSearch│
    │  (short+long)   │  (multi-query) │ (optional)│
    └────────┬────────┴────────┬───────┴─────┬─────┘
             │                 │             │
             └─────────────────┼─────────────┘
                               ↓
                          Rerank（串行依赖）
    """

    def __init__(
        self,
        vectorstore=None,
        reranker=None,
        short_term_memory=None,
        long_term_memory=None,
        web_search_tool=None,
    ):
        """
        初始化并行执行器

        参数:
            vectorstore: 向量存储实例（MilvusClient）
            reranker: 重排模型实例（BGEReranker）
            short_term_memory: 短期记忆管理器
            long_term_memory: 长期记忆管理器
            web_search_tool: 网络搜索工具（可选）
        """
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.web_search_tool = web_search_tool

    async def execute_full_pipeline(
        self,
        question: str,
        intent: str,
        rewritten_queries: List[str] = None,
        session_id: str = None,
        user_id: str = None,
        enable_web_search: bool = False
    ) -> ParallelExecutionResult:
        """
        执行完整的并行管线（核心优化入口）

        参数:
            question: 用户原始问题
            intent: 意图分类结果
            rewritten_queries: 改写后的查询列表
            session_id: 会话 ID
            user_id: 用户 ID
            enable_web_search: 是否启用网络搜索

        返回:
            ParallelExecutionResult 包含所有并行任务结果
        """
        start_time = time.time()
        result = ParallelExecutionResult()

        # 构建并行任务列表（根据意图决定需要哪些任务）
        tasks = []

        # 1. Memory 加载任务（与 Retrieval 无依赖，可并行）
        if session_id or user_id:
            tasks.append(self._execute_memory_tasks(
                question, session_id, user_id, result
            ))

        # 2. Retrieval 任务（与 Memory 无依赖，可并行）
        if intent in ("factual", "multi_hop", "reasoning", "summary"):
            queries = rewritten_queries or [question]
            tasks.append(self._execute_retrieval_tasks(
                question, queries, result
            ))

        # 3. Web Search 任务（与 Memory/Retrieval 无依赖，可并行）
        if enable_web_search and self.web_search_tool:
            tasks.append(self._execute_web_search(
                question, result
            ))

        if tasks:
            # 关键优化：所有无依赖任务并行执行
            await asyncio.gather(*tasks, return_exceptions=True)

        # 4. 跨查询去重（并行检索完成后统一去重）
        if result.retrieved_docs:
            result.retrieved_docs = self._deduplicate_docs(result.retrieved_docs)

        # 5. Rerank 任务（依赖 retrieval 完成，必须串行）
        if result.retrieved_docs and self.reranker:
            rerank_start = time.time()
            try:
                reranked = await asyncio.to_thread(
                    self.reranker.rerank,
                    question, result.retrieved_docs, top_k=3
                )
                result.retrieved_docs = [doc for doc, score in reranked]
                result.execution_times["rerank"] = time.time() - rerank_start
            except Exception as e:
                result.errors.append(f"Rerank 失败: {e}")
                logger.warning(f"Rerank 失败，使用原始排序: {e}")

        # 记录总执行时间
        result.execution_times["total_parallel"] = time.time() - start_time

        if result.errors:
            logger.warning(f"并行执行完成，但有 {len(result.errors)} 个错误: {result.errors}")

        logger.info(
            f"并行执行完成: 总耗时={result.execution_times.get('total_parallel', 0):.2f}s, "
            f"文档数={len(result.retrieved_docs)}, "
            f"记忆条数={len(result.memory_context)}"
        )

        return result

    @staticmethod
    def _deduplicate_docs(docs: List[Any]) -> List[Any]:
        """跨查询去重（基于文档 ID 或内容前缀）"""
        seen_ids = set()
        unique_docs = []
        for doc in docs:
            doc_id = doc.metadata.get("id", doc.page_content[:100])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        return unique_docs

    async def _execute_memory_tasks(
        self,
        question: str,
        session_id: Optional[str],
        user_id: Optional[str],
        result: ParallelExecutionResult
    ):
        """并行加载短期记忆和长期记忆（两者无依赖）"""
        memory_tasks = []

        if self.short_term_memory and session_id:
            memory_tasks.append(
                self._load_short_term_memory(question, session_id, result)
            )

        if self.long_term_memory and user_id:
            memory_tasks.append(
                self._load_long_term_memory(question, user_id, result)
            )

        if memory_tasks:
            try:
                await asyncio.gather(*memory_tasks)
            except Exception as e:
                result.errors.append(f"Memory 加载失败: {e}")
                logger.warning(f"Memory 并行加载失败: {e}")

    async def _execute_retrieval_tasks(
        self,
        question: str,
        queries: List[str],
        result: ParallelExecutionResult
    ):
        """并行执行多查询检索（每个查询独立检索，最后统一去重）"""
        if not self.vectorstore or not hasattr(self.vectorstore, 'vectorstore'):
            logger.warning("向量库未初始化，跳过检索")
            return

        retrieval_tasks = []
        for query in queries:
            retrieval_tasks.append(
                self._retrieve_single_query(query, result)
            )

        try:
            # 多查询并行检索（核心优化点）
            await asyncio.gather(*retrieval_tasks)
        except Exception as e:
            result.errors.append(f"Retrieval 失败: {e}")
            logger.warning(f"并行检索失败: {e}")

    async def _execute_web_search(
        self,
        question: str,
        result: ParallelExecutionResult
    ):
        """异步执行网络搜索（不阻塞主流程）"""
        if not self.web_search_tool:
            return

        try:
            search_start = time.time()
            search_results = await asyncio.to_thread(
                self.web_search_tool.invoke, {"query": question}
            )
            result.search_results = search_results
            result.execution_times["web_search"] = time.time() - search_start
        except Exception as e:
            result.errors.append(f"Web Search 失败: {e}")
            logger.warning(f"网络搜索失败: {e}")

    async def _load_short_term_memory(
        self,
        question: str,
        session_id: str,
        result: ParallelExecutionResult
    ):
        """加载短期记忆"""
        try:
            start = time.time()
            messages = await self.short_term_memory.get_message(session_id)
            history = []
            for msg in messages:
                role = "user" if hasattr(msg, "type") and msg.type == "human" else "assistant"
                history.append({"role": role, "content": msg.content})
            result.conversation_history = history

            context = await self.short_term_memory.get_context(session_id)
            # 短期记忆上下文放在前面，长期记忆 extend 追加在后面
            if context:
                result.memory_context = context.split("\n")
            result.execution_times["short_term_memory"] = time.time() - start
        except Exception as e:
            result.errors.append(f"短期记忆加载失败: {e}")

    async def _load_long_term_memory(
        self,
        question: str,
        user_id: str,
        result: ParallelExecutionResult
    ):
        """加载长期记忆（支持 V2 分层检索）"""
        try:
            start = time.time()
            # V2: 优先使用分层检索（带时间衰减）
            if hasattr(self.long_term_memory, 'search_with_decay'):
                memories = await self.long_term_memory.search_with_decay(
                    user_id=user_id,
                    query=question,
                )
            else:
                memories = await self.long_term_memory.search(user_id, question)

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
                # 用 extend 追加，不覆盖短期记忆
                result.memory_context.extend(memory_contents)

            result.execution_times["long_term_memory"] = time.time() - start
        except Exception as e:
            result.errors.append(f"长期记忆搜索失败: {e}")

    async def _retrieve_single_query(
        self,
        query: str,
        result: ParallelExecutionResult
    ):
        """单个查询的并行检索（同步 similarity_search 用 to_thread 包装）"""
        try:
            start = time.time()

            if hasattr(self.vectorstore, 'vectorstore'):
                docs = await asyncio.to_thread(
                    self.vectorstore.similarity_search, query, k=5
                )
                # 直接追加，跨查询去重在 execute_full_pipeline 中统一处理
                result.retrieved_docs.extend(docs)
                result.execution_times[f"retrieval_{query[:20]}"] = time.time() - start
        except Exception as e:
            result.errors.append(f"查询 '{query[:20]}...' 检索失败: {e}")
