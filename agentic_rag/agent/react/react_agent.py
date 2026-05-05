"""
ReAct Agent - 基于Observe→Think→Act循环的真正Agent系统

与DAG工作流的根本区别：
- DAG: 预定义路径(intent→rewrite→retrieval→rerank→generation→evaluation)，
  LLM只做节点内处理，路由由硬编码route_after_*决定
- ReAct: LLM自主决策每一步行动，根据观察动态调整策略

核心循环：Observe(观察) → Think(思考) → Act(行动) → Observe(观察) → ...

每一步的含义：
- Observe: 提取当前state快照，生成结构化观察摘要
- Think: LLM基于观察自主选择下一步行动（而非硬编码路由）
- Act: 执行选定行动，更新state

流式支持：
- Token级流式：generate行动使用llm.astream()逐token输出
- 中间状态流式：每个Observe/Think/Act阶段都yield事件
"""

import re
import json
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .state import Action, ActionResult
from .safety import SafetyPolicy, HallucinationGuard
from .tools import ToolRegistry, VectorSearchTool, WebSearchTool, QueryRewriteTool, ToolExecutionError
from ...retrieval.query_rewrite import _clean_think_tags
from ...config.logger_config import logger
from ...config.settings import get_settings
from ...memory.gen_cache import get_generation_cache

REACT_SYSTEM_PROMPT = """你是一个智能AI助手，通过观察-思考-行动的循环来回答用户问题。

## 可用工具

{tools_description}

## 行动类型

你可以选择以下行动之一：
1. **retrieve** - 从知识库检索相关文档（需要查找信息时使用）
   参数: {{"query": "检索查询"}}
2. **web_search** - 搜索互联网获取最新信息（知识库信息不足时使用）
   参数: {{"query": "搜索查询"}}
3. **query_rewrite** - 改写查询以获得更好的检索结果（检索结果不理想时使用）
   参数: {{"query": "原始查询", "strategy": "all"}}
4. **tool_call** - 调用特定工具（如计算器等）
   参数: {{"name": "工具名", "args": {{...}}}}
5. **generate** - 基于已有信息生成回答（信息足够时使用）
   参数: {{}}
6. **finish** - 输出最终答案并结束（回答已经足够好时使用）
   参数: {{}}

## 决策规则

- 如果还没有检索过，优先检索知识库
- 如果检索结果不够好，可以改写查询重新检索
- 如果知识库信息不足，可以搜索互联网
- 如果信息足够，生成回答
- 如果回答已经完整且准确，结束

## 输出格式

请严格以JSON格式输出你的决策：
{{"thought": "你的思考过程", "action": "行动类型", "action_input": {{行动参数}}}}

## 当前状态

- 问题: {question}
- 已执行步骤: {step_count}
- 已有信息: {information_summary}
- 行动历史: {action_history}"""

# 优化H：压缩版ReAct System Prompt，减少每步~500 tokens输入
REACT_SYSTEM_PROMPT_COMPACT = """AI助手。工具:{tools_description}
行动:retrieve(query)/web_search(query)/query_rewrite(query,strategy)/tool_call(name,args)/generate/finish
规则:先检索→不够则搜索→足够则生成→完成则结束
状态:问题={question} 步骤={step_count} 信息={information_summary} 历史={action_history}
输出JSON:{{"thought":"...","action":"...","action_input":{{...}}}}"""


# ========== Function Calling 工具定义 ==========

class _RetrieveInput(BaseModel):
    """检索行动的输入参数"""
    query: str = Field(description="检索查询关键词")

class _WebSearchInput(BaseModel):
    """网络搜索行动的输入参数"""
    query: str = Field(description="网络搜索关键词")

class _QueryRewriteInput(BaseModel):
    """查询改写行动的输入参数"""
    query: str = Field(description="原始查询")
    strategy: str = Field(default="all", description="改写策略: expansion/decomposition/all")

class _ToolCallInput(BaseModel):
    """外部工具调用行动的输入参数"""
    name: str = Field(description="工具名称")
    args: dict = Field(default_factory=dict, description="工具调用参数")

class _GenerateInput(BaseModel):
    """生成回答行动的输入参数"""
    reasoning: str = Field(default="", description="选择直接生成回答的原因")

class _FinishInput(BaseModel):
    """结束循环行动的输入参数"""
    reasoning: str = Field(default="", description="认为回答已完成的原因")


def _action_noop(**kwargs):
    """Agent决策占位函数，实际执行由_act方法处理"""
    pass


ACTION_TOOLS = [
    StructuredTool.from_function(
        func=_action_noop,
        name="retrieve",
        description="从知识库检索相关文档。当需要查找信息、回答事实性问题时优先使用。",
        args_schema=_RetrieveInput,
    ),
    StructuredTool.from_function(
        func=_action_noop,
        name="web_search",
        description="搜索互联网获取最新信息。当知识库信息不足或需要实时数据时使用。",
        args_schema=_WebSearchInput,
    ),
    StructuredTool.from_function(
        func=_action_noop,
        name="query_rewrite",
        description="改写查询以获得更好的检索结果。当检索结果不理想或需要多角度检索时使用。",
        args_schema=_QueryRewriteInput,
    ),
    StructuredTool.from_function(
        func=_action_noop,
        name="tool_call",
        description="调用特定外部工具（如计算器等）。当需要执行特定计算或操作时使用。",
        args_schema=_ToolCallInput,
    ),
    StructuredTool.from_function(
        func=_action_noop,
        name="generate",
        description="基于已有信息生成回答。当已收集到足够信息可以回答用户问题时使用。",
        args_schema=_GenerateInput,
    ),
    StructuredTool.from_function(
        func=_action_noop,
        name="finish",
        description="输出最终答案并结束循环。当回答已经完整且准确，无需进一步操作时使用。",
        args_schema=_FinishInput,
    ),
]


class ReActAgent:
    """
    ReAct Agent - 真正的Agent系统

    基于Observe→Think→Act循环，LLM自主决策每一步行动。
    与DAG工作流不同，Agent根据观察结果动态选择行动，
    而非遵循预定义的路径。

    使用方式：
    - invoke(question, **kwargs): 同步执行，返回完整结果
    - arun(question, **kwargs): 异步执行，返回完整结果
    - stream_run(question, **kwargs): 异步流式执行，逐事件返回
    """

    def __init__(
        self,
        llm,
        embeddings,
        vectorstore,
        reranker,
        tools: Dict[str, Any],
        prompt_template: str,
        short_term_memory=None,
        long_term_memory=None,
    ):
        """
        初始化ReAct Agent

        参数:
            llm: 大语言模型实例
            embeddings: 嵌入模型实例
            vectorstore: 向量存储实例
            reranker: 重排模型实例
            tools: 外部工具字典 {name: tool_instance}
            prompt_template: 生成回答的提示词模板
            short_term_memory: 短期记忆管理器
            long_term_memory: 长期记忆管理器
        """
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.prompt_template = prompt_template
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory

        self.tool_registry = ToolRegistry()
        self._register_all_tools(tools)

        self.safety = SafetyPolicy()
        self.guard = HallucinationGuard(self.tool_registry)

    def _register_all_tools(self, external_tools: Dict[str, Any]):
        """
        注册所有可用工具

        将外部工具（duckduckgo、calculator等）和RAG内置工具
        （向量检索、网络搜索、查询改写）统一注册到ToolRegistry。

        参数:
            external_tools: 外部工具字典
        """
        for name, tool in external_tools.items():
            self.tool_registry.register(name, tool, category="external")

        self.tool_registry.register(
            "vector_search",
            VectorSearchTool(self.vectorstore, self.reranker),
            category="rag",
        )
        self.tool_registry.register(
            "web_search",
            WebSearchTool(),
            category="rag",
        )
        self.tool_registry.register(
            "query_rewrite",
            QueryRewriteTool(self.llm, self.embeddings),
            category="rag",
        )

    def _init_state(self, question: str, kwargs: Dict) -> Dict[str, Any]:
        """
        创建初始状态

        参数:
            question: 用户问题
            kwargs: 额外参数

        返回:
            初始状态字典
        """
        return {
            "question": question,
            "observations": [],
            "actions": [],
            "current_step": 0,
            "retrieved_docs": [],
            "search_results": [],
            "tool_results": {},
            "rewritten_queries": [],
            "generation": "",
            "answer": "",
            "memory_context": [],
            "conversation_history": [],
            "errors": [],
            "warnings": [],
            "metadata": kwargs,
        }

    async def _load_memories(self, state: Dict, kwargs: Dict):
        """
        加载短期和长期记忆到state中

        参数:
            state: 当前状态
            kwargs: 包含session_id和user_id的参数
        """
        session_id = kwargs.get("session_id")
        user_id = kwargs.get("user_id")
        question = state["question"]

        if self.short_term_memory and session_id:
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

        if self.long_term_memory and user_id:
            try:
                memories = await self.long_term_memory.search(user_id, question)
                if memories:
                    existing = state.get("memory_context", [])
                    state["memory_context"] = existing + [m["content"] for m in memories]
            except Exception as e:
                logger.warning(f"搜索长期记忆失败: {e}")

    async def _save_memories(self, state: Dict, kwargs: Dict):
        """
        保存对话到短期和长期记忆

        参数:
            state: 当前状态
            kwargs: 包含session_id和user_id的参数
        """
        session_id = kwargs.get("session_id")
        user_id = kwargs.get("user_id")
        question = state["question"]
        answer = state.get("answer", "")

        if self.short_term_memory and session_id:
            try:
                await self.short_term_memory.add_message(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    metadata={"steps": state.get("current_step", 0)},
                )
            except Exception as e:
                logger.warning(f"保存短期记忆失败: {e}")

        if self.long_term_memory and user_id:
            try:
                memory_content = self._build_memory_content(state, question)
                if memory_content:
                    await self.long_term_memory.save_memory(
                        user_id=user_id,
                        content=memory_content,
                        session_id=session_id,
                        metadata={
                            "steps": state.get("current_step", 0),
                            "tools_used": list(state.get("tool_results", {}).keys()),
                        },
                    )
            except Exception as e:
                logger.warning(f"保存长期记忆失败: {e}")

    def _build_memory_content(self, state: Dict, question: str) -> Optional[str]:
        """
        构建长期记忆内容

        参数:
            state: 当前状态
            question: 用户问题

        返回:
            记忆内容字符串，无答案时返回None
        """
        answer = state.get("answer", "")
        if not answer:
            return None

        parts = [f"用户问题: {question}"]
        parts.append(f"最终回答: {answer[:500]}")

        if state.get("retrieved_docs"):
            sources = [doc.metadata.get("source", "unknown") for doc in state["retrieved_docs"]]
            parts.append(f"参考文档: {', '.join(set(sources))}")

        return "\n".join(parts)

    # ========== 核心ReAct循环 ==========

    def _observe(self, state: Dict) -> str:
        """
        观察当前状态，提取决策所需的结构化信息

        这是ReAct循环的第一步：Agent需要"看到"当前局势才能做出决策。
        将state中的关键信息组织为自然语言摘要，供Think阶段使用。

        参数:
            state: 当前状态

        返回:
            结构化的观察摘要
        """
        parts = []
        parts.append(f"用户问题: {state['question']}")
        parts.append(f"当前步骤: {state['current_step']}/{self.safety.MAX_STEPS}")

        if state.get("retrieved_docs"):
            parts.append(f"已检索到 {len(state['retrieved_docs'])} 篇文档")
        else:
            parts.append("尚未检索任何文档")

        if state.get("search_results"):
            parts.append(f"已搜索到 {len(state['search_results'])} 条网络结果")

        if state.get("tool_results"):
            tool_names = list(state["tool_results"].keys())
            parts.append(f"已调用工具: {', '.join(tool_names)}")

        if state.get("generation"):
            parts.append(f"已生成回答(长度: {len(state['generation'])})")

        if state.get("actions"):
            history = []
            for a in state["actions"][-5:]:
                history.append(
                    f"  步骤{a.get('step', '?')}: {a.get('action_type', '?')} - {a.get('reasoning', '')[:50]}"
                )
            parts.append("行动历史:\n" + "\n".join(history))

        if state.get("errors"):
            parts.append(f"最近错误: {state['errors'][-1]}")

        return "\n".join(parts)

    async def _think(self, observation: str, state: Dict) -> Action:
        """
        LLM驱动的行动规划（优先使用Function Calling）

        使用llm.bind_tools()绑定行动工具，LLM通过原生function calling
        选择下一步行动，替代脆弱的JSON正则解析。
        当LLM不支持function calling时，回退到JSON解析模式。

        参数:
            observation: 当前观察
            state: 当前状态

        返回:
            Agent选择的行动
        """
        think_prompt = self._build_think_prompt(state)
        messages = [
            SystemMessage(content=think_prompt),
            HumanMessage(content=observation),
        ]

        try:
            llm_with_tools = self.llm.bind_tools(ACTION_TOOLS)
            response = await llm_with_tools.ainvoke(messages)

            if hasattr(response, 'tool_calls') and response.tool_calls:
                tc = response.tool_calls[0]
                action_type = tc['name']
                action_input = dict(tc.get('args', {}))
                reasoning = action_input.pop('reasoning', '') or ''

                # 如果reasoning为空，尝试从response.content中提取
                if not reasoning and hasattr(response, 'content') and response.content:
                    reasoning = str(response.content)[:200]

                valid_types = {"retrieve", "web_search", "query_rewrite", "tool_call", "generate", "finish"}
                if action_type not in valid_types:
                    logger.warning(f"Function calling返回无效行动类型: {action_type}, 回退到generate")
                    return Action(type="generate", input={}, reasoning=f"无效行动类型: {action_type}")

                logger.info(f"Function calling决策: {action_type}({json.dumps(action_input, ensure_ascii=False)[:80]})")
                return Action(type=action_type, input=action_input, reasoning=reasoning)

            # 回退：LLM返回了文本内容而非tool_calls，尝试JSON解析
            raw_text = response.content if hasattr(response, 'content') else str(response)
            raw_text = _clean_think_tags(raw_text)

            if raw_text and raw_text.strip():
                logger.info("LLM未返回tool_calls，回退到JSON解析模式")
                return await self._think_with_json(observation, state)

            return Action(type="generate", input={}, reasoning="LLM未返回有效决策")

        except (NotImplementedError, TypeError):
            # LLM不支持bind_tools，回退到JSON解析模式
            logger.info("LLM不支持function calling，使用JSON解析模式")
            return await self._think_with_json(observation, state)
        except Exception as e:
            logger.warning(f"Think阶段失败: {e}, 回退到generate")
            return Action(type="generate", input={}, reasoning=f"思考失败: {str(e)}")

    async def _think_with_json(self, observation: str, state: Dict) -> Action:
        """
        JSON解析模式的Think（回退方案）

        当LLM不支持function calling时，使用prompt引导LLM输出JSON，
        再通过括号匹配算法提取和解析。支持嵌套JSON结构。

        参数:
            observation: 当前观察
            state: 当前状态

        返回:
            Agent选择的行动
        """
        tools_desc = self.tool_registry.get_description()

        prompt = REACT_SYSTEM_PROMPT_COMPACT.format(
            tools_description=tools_desc,
            question=state["question"],
            step_count=state["current_step"],
            information_summary=self._summarize_information(state),
            action_history=self._format_action_history(state),
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=observation),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw_text = response.content if hasattr(response, 'content') else str(response)
            raw_text = _clean_think_tags(raw_text)

            action = self._parse_action(raw_text)
            return action
        except Exception as e:
            logger.warning(f"JSON模式Think失败: {e}, 回退到generate")
            return Action(type="generate", input={}, reasoning=f"思考失败: {str(e)}")

    def _build_think_prompt(self, state: Dict) -> str:
        """
        构建Think阶段的系统提示词（Function Calling模式）

        使用Function Calling时，工具列表由bind_tools提供，
        提示词只需指导决策逻辑，无需重复列出工具详情。

        参数:
            state: 当前状态

        返回:
            系统提示词字符串
        """
        return f"""你是一个智能AI助手，通过观察-思考-行动的循环来回答用户问题。

## 决策规则

根据当前状态选择最合适的行动：
- 如果还没有检索过且需要查找信息，优先检索知识库（retrieve）
- 如果检索结果不够好，可以改写查询重新检索（query_rewrite）
- 如果知识库信息不足或需要实时数据，搜索互联网（web_search）
- 如果需要执行特定操作，调用外部工具（tool_call）
- 如果已收集到足够信息，生成回答（generate）
- 如果回答已经完整且准确，结束（finish）

## 当前状态

- 问题: {state['question']}
- 已执行步骤: {state['current_step']}
- 已有信息: {self._summarize_information(state)}
- 行动历史: {self._format_action_history(state)}"""

    def _should_skip_think(self, state: Dict, step: int) -> Optional[Action]:
        """
        快速路径判断（仅限首步）

        仅在首步且无任何信息时跳过LLM Think调用，
        后续步骤一律由LLM自主决策，确保Agent的智能性。
        移除了"已有检索结果直接生成"的快捷路径，
        让Agent能根据检索质量自主决定下一步（生成/改写/搜索等）。

        参数:
            state: 当前状态
            step: 当前步骤编号

        返回:
            如果可以跳过Think，返回预定的Action；否则返回None需要LLM决策
        """
        # 首步且无检索结果：直接检索（唯一允许跳过Think的场景）
        if step == 0 and not state.get("retrieved_docs"):
            return Action(
                type="retrieve",
                input={"query": state["question"]},
                reasoning="首步直接检索，跳过LLM决策"
            )

        # 后续步骤：由LLM自主决策
        return None

    def _parse_action(self, raw_text: str) -> Action:
        """
        从LLM输出中解析行动（改进版：支持嵌套JSON）

        使用括号匹配算法提取JSON，替代无法处理嵌套结构的正则表达式。
        原正则r"\\{[^{}]*\\}"无法匹配action_input中包含嵌套对象的场景，
        如tool_call的args参数。括号匹配算法正确处理所有嵌套层级。

        参数:
            raw_text: LLM的原始输出文本

        返回:
            解析后的Action对象
        """
        json_str = self._extract_json(raw_text)

        if not json_str:
            logger.warning(f"无法从LLM输出中提取JSON: {raw_text[:200]}")
            return Action(type="generate", input={}, reasoning="无法解析行动，直接生成")

        try:
            result = json.loads(json_str)
            action_type = result.get("action", "generate")
            action_input = result.get("action_input", {})
            reasoning = result.get("thought", "")

            valid_types = {"retrieve", "web_search", "query_rewrite", "tool_call", "generate", "finish"}
            if action_type not in valid_types:
                logger.warning(f"无效的行动类型: {action_type}, 回退到generate")
                action_type = "generate"

            return Action(type=action_type, input=action_input, reasoning=reasoning)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}, 原始文本: {raw_text[:200]}")
            return Action(type="generate", input={}, reasoning="JSON解析失败，直接生成")

    def _extract_json(self, text: str) -> Optional[str]:
        """
        从文本中提取最外层JSON对象（支持嵌套结构）

        使用括号匹配算法，正确处理嵌套的JSON对象和字符串内的花括号。
        替代原来的正则r"\\{[^{}]*\\}"，该正则无法匹配嵌套JSON。

        参数:
            text: 包含JSON的文本

        返回:
            提取到的JSON字符串，未找到返回None
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

    async def _act(self, action: Action, state: Dict) -> ActionResult:
        """
        执行Agent选择的行动

        根据行动类型分派到具体的执行方法。
        每种行动类型有独立的处理逻辑。

        参数:
            action: 要执行的行动
            state: 当前状态

        返回:
            行动执行结果
        """
        try:
            if action.type == "retrieve":
                return await self._act_retrieve(action, state)
            elif action.type == "web_search":
                return await self._act_web_search(action, state)
            elif action.type == "query_rewrite":
                return await self._act_query_rewrite(action, state)
            elif action.type == "tool_call":
                return await self._act_tool_call(action, state)
            elif action.type == "generate":
                return await self._act_generate(action, state)
            elif action.type == "finish":
                return ActionResult(success=True, data={"finished": True})
            else:
                return ActionResult(success=False, error=f"未知行动类型: {action.type}")
        except Exception as e:
            logger.error(f"行动执行失败: {action.type}, 错误: {e}")
            return ActionResult(success=False, error=str(e))

    async def _act_retrieve(self, action: Action, state: Dict) -> ActionResult:
        """
        执行向量检索行动

        使用VectorSearchTool检索+重排，结果累积到state中
        （而非替换），Agent可以在多步中逐步积累信息。

        参数:
            action: 检索行动（含query参数）
            state: 当前状态

        返回:
            检索结果
        """
        query = action.input.get("query", state["question"])
        tool = self.tool_registry.get("vector_search")

        if not tool:
            return ActionResult(success=False, error="向量检索工具未注册")

        result = tool.invoke({"query": query})
        docs = result.get("documents", [])

        if docs:
            existing = state.get("retrieved_docs", [])
            seen_ids = {doc.metadata.get("id", doc.page_content[:100]) for doc in existing}
            for doc in docs:
                doc_id = doc.metadata.get("id", doc.page_content[:100])
                if doc_id not in seen_ids:
                    existing.append(doc)
                    seen_ids.add(doc_id)
            state["retrieved_docs"] = existing
            return ActionResult(success=True, data=result)

        return ActionResult(success=False, error="未检索到相关文档")

    async def _act_web_search(self, action: Action, state: Dict) -> ActionResult:
        """
        执行网络搜索行动

        参数:
            action: 搜索行动（含query参数）
            state: 当前状态

        返回:
            搜索结果
        """
        query = action.input.get("query", state["question"])
        tool = self.tool_registry.get("web_search")

        if not tool:
            return ActionResult(success=False, error="网络搜索工具未注册")

        result = tool.invoke({"query": query})
        docs = result.get("documents", [])

        if docs:
            existing = state.get("search_results", [])
            existing.extend(docs)
            state["search_results"] = existing
            return ActionResult(success=True, data=result)

        return ActionResult(success=False, error="网络搜索未返回结果")

    async def _act_query_rewrite(self, action: Action, state: Dict) -> ActionResult:
        """
        执行查询改写行动

        改写后自动用新查询重新检索，Agent无需额外步骤。

        参数:
            action: 改写行动（含query和strategy参数）
            state: 当前状态

        返回:
            改写+检索结果
        """
        query = action.input.get("query", state["question"])
        strategy = action.input.get("strategy", "all")
        tool = self.tool_registry.get("query_rewrite")

        if not tool:
            return ActionResult(success=False, error="查询改写工具未注册")

        result = tool.invoke({"query": query, "strategy": strategy})
        rewritten = result.get("rewritten_queries", [])

        if rewritten:
            state["rewritten_queries"] = rewritten

            vector_tool = self.tool_registry.get("vector_search")
            if vector_tool:
                all_docs = []
                seen_ids = {
                    doc.metadata.get("id", doc.page_content[:100])
                    for doc in state.get("retrieved_docs", [])
                }

                for rq in rewritten:
                    r_result = vector_tool.invoke({"query": rq})
                    for doc in r_result.get("documents", []):
                        doc_id = doc.metadata.get("id", doc.page_content[:100])
                        if doc_id not in seen_ids:
                            all_docs.append(doc)
                            seen_ids.add(doc_id)

                if all_docs:
                    existing = state.get("retrieved_docs", [])
                    existing.extend(all_docs)
                    state["retrieved_docs"] = existing

            return ActionResult(success=True, data=result)

        return ActionResult(success=False, error="查询改写未产生新查询")

    async def _act_tool_call(self, action: Action, state: Dict) -> ActionResult:
        """
        执行通用工具调用行动

        参数:
            action: 工具调用行动（含name和args参数）
            state: 当前状态

        返回:
            工具调用结果
        """
        tool_name = action.input.get("name", "")
        tool_args = action.input.get("args", {})

        if not tool_name:
            return ActionResult(success=False, error="未指定工具名称")

        try:
            result = await self.tool_registry.execute(tool_name, tool_args)
            tool_results = state.get("tool_results", {})
            tool_results[tool_name] = result
            state["tool_results"] = tool_results
            return ActionResult(success=True, data={"name": tool_name, "result": result})
        except ToolExecutionError as e:
            return ActionResult(success=False, error=str(e))

    async def _act_generate(self, action: Action, state: Dict) -> ActionResult:
        """
        执行生成回答行动

        基于当前累积的所有信息（检索文档+搜索结果+工具结果+记忆）
        生成回答。

        参数:
            action: 生成行动
            state: 当前状态

        返回:
            生成结果
        """
        context = self._build_context(state)
        prompt = self.prompt_template.format(context=context, question=state["question"])

        try:
            response = await self.llm.ainvoke(prompt)
            generation = response.content if hasattr(response, "content") else str(response)
            generation = _clean_think_tags(generation)

            state["generation"] = generation
            state["answer"] = generation

            return ActionResult(success=True, data={"generation": generation})
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return ActionResult(success=False, error=str(e))

    async def _stream_generate(self, state: Dict) -> AsyncIterator[str]:
        """
        Token级流式生成回答

        使用llm.astream()实现真正的逐token流式输出，
        而非先获取完整文本再分段yield的伪流式。
        这是流式改造的核心：从"延迟后突然快速输出"变为"逐token实时输出"。

        参数:
            state: 当前状态

        产出:
            逐token的文本片段
        """
        context = self._build_context(state)
        prompt = self.prompt_template.format(context=context, question=state["question"])

        full_text = []
        async for chunk in self.llm.astream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                full_text.append(token)
                yield token

        state["generation"] = "".join(full_text)
        state["answer"] = state["generation"]

    def _build_context(self, state: Dict) -> str:
        """
        构建生成回答的上下文

        按优先级合并：记忆 > 对话历史 > 检索文档 > 搜索结果 > 工具结果
        与DAG的generation_node保持一致的上下文构建逻辑，
        确保重构后回答质量不下降。

        参数:
            state: 当前状态

        返回:
            合并后的上下文字符串
        """
        context_parts = []

        memory_context = state.get("memory_context", [])
        if memory_context:
            memory_text = "\n".join(memory_context) if isinstance(memory_context, list) else str(memory_context)
            max_memory_chars = 2000
            if len(memory_text) > max_memory_chars:
                memory_text = memory_text[-max_memory_chars:]
            context_parts.append(f"【相关记忆】\n{memory_text}")

        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            history_lines = []
            recent = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            for msg in recent:
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")
                if len(content) > 500:
                    content = content[:500] + "..."
                history_lines.append(f"{role}: {content}")
            if history_lines:
                context_parts.append("【对话历史】\n" + "\n".join(history_lines))

        retrieved_docs = state.get("retrieved_docs", [])
        if retrieved_docs:
            docs_content = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]])
            context_parts.append(f"【检索到的文档】\n{docs_content}")

        search_results = state.get("search_results", [])
        if search_results:
            search_content = "\n\n".join([doc.page_content for doc in search_results])
            context_parts.append(f"【网络搜索结果】\n{search_content}")

        tool_results = state.get("tool_results", {})
        if tool_results and isinstance(tool_results, dict):
            tool_context = "\n\n【工具调用结果】\n"
            for tool_name, result in tool_results.items():
                tool_context += f"- {tool_name}: {result}\n"
            context_parts.append(tool_context)

        return "\n".join(context_parts) if context_parts else "(无相关上下文)"

    def _update_state(self, state: Dict, action: Action, result: ActionResult):
        """
        根据行动结果更新状态

        主要记录错误信息，供Agent在下一轮Observe中感知失败。

        参数:
            state: 当前状态
            action: 执行的行动
            result: 行动结果
        """
        if not result.success and result.error:
            errors = state.get("errors", [])
            errors.append(f"步骤{state['current_step']}: {action.type}失败 - {result.error}")
            state["errors"] = errors

    def _summarize_information(self, state: Dict) -> str:
        """总结当前已有的信息"""
        parts = []
        if state.get("retrieved_docs"):
            parts.append(f"{len(state['retrieved_docs'])}篇检索文档")
        if state.get("search_results"):
            parts.append(f"{len(state['search_results'])}条搜索结果")
        if state.get("tool_results"):
            parts.append(f"{len(state['tool_results'])}个工具结果")
        if state.get("generation"):
            parts.append("已生成回答")
        return ", ".join(parts) if parts else "无"

    def _format_action_history(self, state: Dict) -> str:
        """格式化行动历史"""
        actions = state.get("actions", [])
        if not actions:
            return "无"

        lines = []
        for a in actions[-5:]:
            input_str = json.dumps(a.get("action_input", {}), ensure_ascii=False)[:50]
            lines.append(f"步骤{a.get('step', '?')}: {a.get('action_type', '?')}({input_str})")
        return "\n".join(lines)

    def _summarize_result(self, result: ActionResult) -> str:
        """总结行动结果"""
        if result.success:
            if isinstance(result.data, dict):
                count = result.data.get("count", "")
                if count:
                    return f"成功，获得{count}条结果"
            return "成功"
        return f"失败: {result.error or '未知错误'}"

    def _format_result(self, state: Dict) -> Dict[str, Any]:
        """
        格式化最终输出，兼容现有API的QueryResponse格式

        确保ReAct Agent的输出与DAG模式的结构一致，
        上层API代码无需修改即可使用。

        参数:
            state: 最终状态

        返回:
            兼容QueryResponse的结果字典
        """
        answer = state.get("answer", "") or state.get("generation", "")

        all_docs = list(state.get("retrieved_docs", []))
        all_docs.extend(state.get("search_results", []))

        return {
            "answer": answer,
            "generation": state.get("generation", ""),
            "refined_answer": state.get("answer", ""),
            "sources": all_docs,
            "reranked_docs": state.get("retrieved_docs", []),
            "evaluation": {
                "steps": state.get("current_step", 0),
                "actions": len(state.get("actions", [])),
                "errors": len(state.get("errors", [])),
            },
            "intent": "react_agent",
            "tools_used": list(state.get("tool_results", {}).keys()),
            "reflection_count": 0,
            "metadata": state.get("metadata", {}),
        }

    # ========== 公开接口 ==========

    def invoke(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        同步执行ReAct循环

        内部创建事件循环运行异步方法，
        保持与DAG模式invoke()相同的调用方式。

        参数:
            question: 用户问题
            **kwargs: 额外参数（session_id, user_id等）

        返回:
            包含答案和元数据的结果字典
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.arun(question, **kwargs))
        finally:
            loop.close()

    async def arun(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        异步执行ReAct循环（非流式）

        核心执行流程：
        1. 初始化state，加载记忆
        2. 循环执行 Observe → Think → Act
        3. 每步检查安全策略
        4. 连续失败时强制生成
        5. 保存记忆，返回结果

        参数:
            question: 用户问题
            **kwargs: 额外参数（session_id, user_id等）

        返回:
            包含答案和元数据的结果字典
        """
        state = self._init_state(question, kwargs)
        await self._load_memories(state, kwargs)

        self.safety.reset()

        for step in range(self.safety.MAX_STEPS):
            state["current_step"] = step + 1

            observation = self._observe(state)
            state["observations"].append(observation)

            # 优化B：快速路径——对确定性步骤跳过LLM Think调用
            skip_action = self._should_skip_think(state, step)
            if skip_action:
                action = skip_action
                logger.info(f"快速路径跳过Think: step={step+1}, action={action.type}")
            else:
                action = await self._think(observation, state)

            action = self.guard.validate_action(action, state)
            allowed = self.safety.check(action)
            if not allowed:
                action = Action(type="generate", input={}, reasoning="安全策略限制，强制生成回答")

            state["actions"].append({
                "step": step + 1,
                "action_type": action.type,
                "action_input": action.input,
                "reasoning": action.reasoning,
            })

            if action.type == "finish":
                break

            result = await self._act(action, state)
            self._update_state(state, action, result)
            self.safety.record_result(result.success)

            if self.safety.is_consecutive_fail_exceeded():
                logger.warning("连续失败次数过多，强制生成回答")
                fallback = Action(type="generate", input={}, reasoning="连续失败，强制生成")
                fallback_result = await self._act(fallback, state)
                self._update_state(state, fallback, fallback_result)
                break

            if action.type == "generate" and result.success:
                break

        if not state.get("answer"):
            state["answer"] = state.get("generation", "抱歉，无法生成回答。")

        await self._save_memories(state, kwargs)
        return self._format_result(state)

    async def stream_run(
        self,
        question: str,
        session_id: str = None,
        user_id: str = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式执行ReAct循环

        与非流式版本不同，每个阶段都yield事件，
        用户可以实时看到Agent的思考过程和行动。

        事件类型：
        - thought: Agent的观察结果
        - action: Agent选择的行动
        - observation: 行动执行结果
        - status: 状态更新
        - token: 生成内容的token级流式（真正的逐token输出）
        - done: 完成信号

        参数:
            question: 用户问题
            session_id: 会话ID
            user_id: 用户ID
            **kwargs: 额外参数

        产出:
            Dict[str, Any]: 流式事件
        """
        state = self._init_state(question, kwargs)
        kwargs["session_id"] = session_id
        kwargs["user_id"] = user_id
        await self._load_memories(state, kwargs)

        # 生成缓存快速路径：重复问题直接返回缓存结果
        settings = get_settings()
        if settings.generation_cache_enabled:
            gen_cache = get_generation_cache()
            cached_gen = gen_cache.get(question, None)
            if cached_gen:
                cached_response = cached_gen.get("response", "")
                logger.info(f"ReAct生成缓存命中: {question[:50]}...")
                # 模拟流式输出缓存结果
                for i in range(0, len(cached_response), 50):
                    chunk = cached_response[i:i+50]
                    if chunk:
                        yield {
                            "type": "token",
                            "content": chunk,
                            "data": {"step": 0, "cached": True}
                        }
                        await asyncio.sleep(0.03)
                yield {
                    "type": "done",
                    "content": "回答生成完成(缓存)",
                    "data": {
                        "session_id": session_id,
                        "user_id": user_id,
                        "intent": "react_agent",
                        "reflection_count": 0,
                        "gen_cache_hit": True,
                        "cached": True,
                        "sources": [],
                    },
                }
                return

        self.safety.reset()

        for step in range(self.safety.MAX_STEPS):
            state["current_step"] = step + 1

            observation = self._observe(state)
            state["observations"].append(observation)
            yield {
                "type": "thought",
                "content": observation,
                "data": {"step": step + 1, "phase": "observe"},
            }

            # 优化B：快速路径——对确定性步骤跳过LLM Think调用
            skip_action = self._should_skip_think(state, step)
            if skip_action:
                action = skip_action
                logger.info(f"快速路径跳过Think(stream): step={step+1}, action={action.type}")
            else:
                action = await self._think(observation, state)
            yield {
                "type": "action",
                "content": f"{action.type}: {json.dumps(action.input, ensure_ascii=False)}",
                "data": {
                    "action_type": action.type,
                    "action_input": action.input,
                    "reasoning": action.reasoning,
                    "step": step + 1,
                    "phase": "think",
                },
            }

            action = self.guard.validate_action(action, state)
            allowed = self.safety.check(action)
            if not allowed:
                action = Action(type="generate", input={}, reasoning="安全策略限制")

            state["actions"].append({
                "step": step + 1,
                "action_type": action.type,
                "action_input": action.input,
                "reasoning": action.reasoning,
            })

            if action.type == "finish":
                break

            if action.type == "generate":
                yield {
                    "type": "status",
                    "content": "正在生成回答...",
                    "data": {"step": step + 1, "phase": "generate"},
                }
                async for token in self._stream_generate(state):
                    yield {
                        "type": "token",
                        "content": token,
                        "data": {"step": step + 1},
                    }
                break
            else:
                result = await self._act(action, state)
                self._update_state(state, action, result)
                self.safety.record_result(result.success)

                yield {
                    "type": "observation",
                    "content": self._summarize_result(result),
                    "data": {
                        "action_type": action.type,
                        "success": result.success,
                        "step": step + 1,
                        "phase": "act",
                    },
                }

                if self.safety.is_consecutive_fail_exceeded():
                    logger.warning("连续失败次数过多，强制生成回答")
                    yield {
                        "type": "status",
                        "content": "连续失败，强制生成回答...",
                        "data": {"step": step + 1, "phase": "fallback"},
                    }
                    async for token in self._stream_generate(state):
                        yield {
                            "type": "token",
                            "content": token,
                            "data": {"step": step + 1},
                        }
                    break

        if not state.get("answer"):
            state["answer"] = state.get("generation", "抱歉，无法生成回答。")

        await self._save_memories(state, kwargs)

        # 生成缓存写入：将本次结果缓存，下次相同问题直接返回
        if settings.generation_cache_enabled:
            gen_cache = get_generation_cache()
            answer = state.get("answer", "")
            if answer and len(answer) > 10:
                gen_cache.set(
                    question,
                    answer,
                    intent=None,
                    metadata={"cached_at": __import__('time').time(), "mode": "react"}
                )
                logger.info(f"ReAct生成缓存已写入: {question[:50]}...")

        all_docs = list(state.get("retrieved_docs", []))
        all_docs.extend(state.get("search_results", []))

        yield {
            "type": "done",
            "content": "回答生成完成",
            "data": {
                "session_id": session_id,
                "user_id": user_id,
                "intent": "react_agent",
                "reflection_count": 0,
                "tools_used": list(state.get("tool_results", {}).keys()),
                "steps": state.get("current_step", 0),
                "actions_count": len(state.get("actions", [])),
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                        "score": doc.metadata.get("score"),
                    }
                    for doc in all_docs[:5]
                ],
            },
        }
