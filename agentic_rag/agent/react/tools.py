"""
工具注册表 - 统一管理所有工具的注册、发现和执行

DAG模式中，工具调用由intent_classification节点触发，
工具选择由硬编码的route_after_*决定。

ReAct模式中，所有工具统一注册到ToolRegistry，
Agent在Think阶段自主决定调用哪个工具。

同时将DAG中的retrieval+rerank、web_search、query_rewrite
封装为独立工具，Agent可以像调用calculator一样调用它们。
"""

from typing import Dict, Any, List, Optional
from langchain_core.documents import Document

from ...config.logger_config import logger


class ToolNotFoundError(Exception):
    """工具不存在异常"""
    pass


class ToolExecutionError(Exception):
    """工具执行失败异常"""
    pass


class ToolRegistry:
    """
    工具注册表

    统一管理所有工具的注册、描述生成和执行。
    支持动态注册和分类管理（external/rag/general）。

    Agent在Think阶段通过get_description()获取可用工具列表，
    在Act阶段通过execute()调用具体工具。
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, tool: Any, category: str = "general"):
        """
        注册工具

        参数:
            name: 工具名称（Agent通过此名称调用工具）
            tool: 工具实例（BaseTool或自定义可调用对象）
            category: 工具分类（external=外部工具, rag=RAG内置工具, general=通用）
        """
        description = ""
        if hasattr(tool, 'description'):
            description = tool.description
        elif hasattr(tool, '__doc__') and tool.__doc__:
            description = tool.__doc__.strip().split('\n')[0]

        self._tools[name] = {
            "tool": tool,
            "category": category,
            "description": description,
        }
        logger.debug(f"注册工具: {name} (分类: {category})")

    def has(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools

    def get(self, name: str) -> Optional[Any]:
        """获取工具实例"""
        entry = self._tools.get(name)
        return entry["tool"] if entry else None

    def get_description(self) -> str:
        """
        获取所有工具的描述文本

        供ReAct prompt使用，让LLM知道有哪些工具可用。
        格式为每行一个工具：- 工具名: 描述
        """
        lines = []
        for name, entry in self._tools.items():
            lines.append(f"- {name}: {entry['description']}")
        return "\n".join(lines)

    def get_tool_names(self) -> List[str]:
        """获取所有工具名称列表"""
        return list(self._tools.keys())

    async def execute(self, name: str, args: Dict[str, Any]) -> Any:
        """
        执行工具调用

        优先使用ainvoke（异步），回退到invoke（同步），最后尝试直接调用。
        这样既支持LangChain BaseTool，也支持普通可调用对象。

        参数:
            name: 工具名称
            args: 调用参数字典

        返回:
            工具执行结果

        异常:
            ToolNotFoundError: 工具不存在
            ToolExecutionError: 工具执行失败
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"工具不存在: {name}")

        tool = self._tools[name]["tool"]

        try:
            if hasattr(tool, 'ainvoke'):
                return await tool.ainvoke(args)
            elif hasattr(tool, 'invoke'):
                return tool.invoke(args)
            elif callable(tool):
                return tool(**args)
            else:
                raise ToolExecutionError(f"工具'{name}'不可调用")
        except (ToolNotFoundError, ToolExecutionError):
            raise
        except Exception as e:
            raise ToolExecutionError(f"工具'{name}'执行失败: {str(e)}") from e


class VectorSearchTool:
    """
    向量检索工具（含重排）

    将DAG中的retrieval+rerank两个节点封装为单一工具。
    Agent只需调用一次即可获得重排后的高质量文档，
    无需关心检索和重排的内部细节。
    """

    def __init__(self, vectorstore, reranker):
        """
        参数:
            vectorstore: 向量存储实例
            reranker: 重排模型实例
        """
        self.vectorstore = vectorstore
        self.reranker = reranker

    @property
    def description(self) -> str:
        return "从知识库中检索相关文档，自动重排序后返回最相关的结果。参数: query(检索查询), top_k(返回文档数，默认3)"

    def invoke(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行向量检索+重排

        先检索top_k*2篇文档，再重排取top_k篇，
        确保返回的文档既相关又高质量。

        参数:
            args: {"query": str, "top_k": int}

        返回:
            {"documents": List[Document], "count": int}
        """
        query = args.get("query", "")
        top_k = args.get("top_k", 3)

        if not query:
            return {"documents": [], "count": 0}

        if not hasattr(self.vectorstore, 'vectorstore') or self.vectorstore.vectorstore is None:
            return {"documents": [], "count": 0, "error": "向量库未初始化"}

        try:
            docs = self.vectorstore.similarity_search(query, k=top_k * 2)
        except Exception as e:
            logger.warning(f"向量检索失败: {e}")
            return {"documents": [], "count": 0, "error": str(e)}

        if not docs:
            return {"documents": [], "count": 0}

        try:
            reranked = self.reranker.rerank(query, docs, top_k=top_k)
            reranked_docs = [doc for doc, _score in reranked]
        except Exception as e:
            logger.warning(f"重排失败，使用原始排序: {e}")
            reranked_docs = docs[:top_k]

        return {"documents": reranked_docs, "count": len(reranked_docs)}

    async def ainvoke(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行向量检索+重排"""
        return self.invoke(args)


class WebSearchTool:
    """
    网络搜索工具

    封装DuckDuckGo搜索，返回格式化的Document列表。
    将DAG中的web_search_node封装为工具，
    Agent可以在任何步骤自主决定是否需要搜索互联网。
    """

    def __init__(self):
        from ...tools.search import duckduckgo_search
        self._search = duckduckgo_search

    @property
    def description(self) -> str:
        return "搜索互联网获取最新信息。参数: query(搜索查询)"

    def invoke(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行网络搜索

        参数:
            args: {"query": str}

        返回:
            {"documents": List[Document], "count": int}
        """
        query = args.get("query", "")
        if not query:
            return {"documents": [], "count": 0}

        try:
            search_text = self._search.invoke({"query": query})

            if not search_text or search_text == "未找到相关结果":
                return {"documents": [], "count": 0}

            docs = self._parse_search_results(search_text)
            return {"documents": docs, "count": len(docs)}
        except Exception as e:
            logger.warning(f"网络搜索失败: {e}")
            return {"documents": [], "count": 0, "error": str(e)}

    async def ainvoke(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行网络搜索"""
        return self.invoke(args)

    def _parse_search_results(self, search_text: str) -> List[Document]:
        """
        将搜索结果文本解析为Document列表

        DuckDuckGo返回的是格式化文本，需要解析为结构化的Document。
        解析逻辑与原web_search_node保持一致，确保兼容性。

        参数:
            search_text: 搜索结果的原始文本

        返回:
            Document列表
        """
        docs = []
        lines = search_text.split("\n")
        current_result = {"title": "", "body": "", "url": ""}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line and line[0].isdigit() and ". " in line:
                if current_result["title"] or current_result["body"]:
                    content = f"标题: {current_result['title']}\n内容: {current_result['body']}\n来源: {current_result['url']}"
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": current_result["url"], "title": current_result["title"], "type": "web_search"},
                    ))
                title = line.split(". ", 1)[1] if ". " in line else line
                current_result = {"title": title, "body": "", "url": ""}
            elif line.startswith("来源:"):
                current_result["url"] = line.replace("来源:", "").strip()
            elif "   " in line:
                current_result["body"] += line.strip() + " "

        if current_result["title"] or current_result["body"]:
            content = f"标题: {current_result['title']}\n内容: {current_result['body']}\n来源: {current_result['url']}"
            docs.append(Document(
                page_content=content,
                metadata={"source": current_result["url"], "title": current_result["title"], "type": "web_search"},
            ))

        return docs


class QueryRewriteTool:
    """
    查询改写工具

    将DAG中的query_rewrite_node封装为工具。
    Agent可以在检索结果不理想时自主决定改写查询，
    而不是每次都强制改写（DAG模式的做法）。
    """

    def __init__(self, llm, embeddings):
        """
        参数:
            llm: 大语言模型实例
            embeddings: 嵌入模型实例
        """
        self.llm = llm
        self.embeddings = embeddings

    @property
    def description(self) -> str:
        return "改写查询以获得更好的检索结果。参数: query(原始查询), strategy(策略: expansion/decomposition/all，默认all)"

    def invoke(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行查询改写

        参数:
            args: {"query": str, "strategy": str}

        返回:
            {"rewritten_queries": List[str], "count": int}
        """
        from ...retrieval.query_rewrite import QueryRewriter

        query = args.get("query", "")
        strategy = args.get("strategy", "all")

        if not query:
            return {"rewritten_queries": [query], "count": 1}

        try:
            rewriter = QueryRewriter(self.llm, self.embeddings)
            rewritten = rewriter.rewrite(query, strategy=strategy)
            return {"rewritten_queries": rewritten, "count": len(rewritten)}
        except Exception as e:
            logger.warning(f"查询改写失败: {e}")
            return {"rewritten_queries": [query], "count": 1}

    async def ainvoke(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行查询改写"""
        return self.invoke(args)
