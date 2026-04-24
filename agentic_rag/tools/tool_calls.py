from typing import Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.agents import AgentState

from ..config.logger_config import logger

# 1. 定义工具调用的输出格式
class ToolCallRequest(BaseModel):
    name: str = Field(description="要调用的工具名称")
    parameters: Dict[str, Any] = Field(description="调用工具所需的参数")

class ToolCallRequests(BaseModel):
    calls: list[ToolCallRequest] = Field(description="要调用的工具列表")

# 2. 工具调用 Prompt（生产级版本，支持任意工具）
TOOL_CALL_PROMPT = ChatPromptTemplate.from_template("""
你是一个智能工具调用助手，需要根据用户的问题，判断是否需要调用工具、调用哪个工具、以及传入什么参数。

可用工具列表：
{tools_desc}

用户问题：{question}

请严格按照JSON格式返回需要调用的工具列表,格式如下:
{{
    "calls": [
        {{"name": "工具名", "parameters": {{"参数名": "参数值"}}}}
    ]
}}
如果不需要调用任何工具，返回空列表即可：{{"calls": []}}
""")

def tool_call(state: AgentState, llm: BaseChatModel, tools: Dict[str, BaseTool]) -> AgentState:
    """
    生产级工具调用节点(LLM驱动版)
    自动决定工具选择、参数解析、错误处理
    """
    question = state["question"]
    tool_results = {}

    # --- 步骤1：生成工具描述，给LLM看 ---
    tools_desc = "\n".join([
        f"- 工具名：{name}，功能：{tool.description}，参数：{tool.args}"
        for name, tool in tools.items()
    ])

    # --- 步骤2：调用LLM，获取工具调用请求 ---
    parser = PydanticOutputParser(pydantic_object=ToolCallRequests) # 解析器，将LLM输出解析为ToolCallRequests对象
    chain = TOOL_CALL_PROMPT | llm | parser

    try:
        response = chain.invoke({
            "question": question,
            "tools_desc": tools_desc
        })
    except Exception as e:
        # 错误兜底：解析失败时不调用任何工具，避免流程崩溃
        logger.error("工具调用节点解析失败：{}", str(e))
        return {"tool_results": {}}

    # --- 步骤3：执行工具调用 ---
    for call in response.calls:
        tool_name = call.name
        tool_params = call.parameters

        # 工具存在性校验
        if tool_name not in tools:
            continue
        
        tool = tools[tool_name]
        try:
            # 调用工具（自动处理参数）
            result = tool.invoke(tool_params)
            tool_results[tool_name] = result
        except Exception as e:
            # 单个工具调用失败不影响整体流程
            logger.error("工具调用节点调用工具{}失败：{}", tool_name, str(e))
            tool_results[tool_name] = f"工具调用失败：{str(e)}"

    return {"tool_results": tool_results}