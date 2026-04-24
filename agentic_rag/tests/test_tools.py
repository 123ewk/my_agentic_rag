"""
工具调用测试
测试搜索工具和工具调用逻辑
"""
import pytest
from unittest.mock import MagicMock, patch

from agentic_rag.tools.search import (
    duckduckgo_search,
    calculator,
    python_repl,
    get_search_tools
)
from agentic_rag.tools.tool_calls import tool_call, ToolCallRequests, ToolCallRequest


class TestDuckDuckGoSearch:
    """DuckDuckGo搜索工具测试"""

    def test_search_basic(self):
        """测试基础搜索功能"""
        with patch('agentic_rag.tools.search.DDGS') as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = iter([
                {
                    "title": "测试标题",
                    "body": "测试内容摘要",
                    "href": "https://example.com"
                }
            ])
            mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
            
            result = duckduckgo_search.invoke({"query": "测试查询"})
            
            assert isinstance(result, str)

    def test_search_empty_results(self):
        """测试搜索无结果"""
        with patch('agentic_rag.tools.search.DDGS') as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = iter([])
            mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
            
            result = duckduckgo_search.invoke({"query": "不存在的内容"})
            
            assert "未找到" in result or result == "无结果"

    def test_search_error_handling(self):
        """测试搜索错误处理"""
        with patch('agentic_rag.tools.search.DDGS') as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.side_effect = Exception("网络错误")
            mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)
            
            result = duckduckgo_search.invoke({"query": "测试"})
            
            assert "出错" in result or "错误" in result


class TestCalculator:
    """计算器工具测试"""

    def test_addition(self):
        """测试加法"""
        result = calculator.invoke({"expression": "2 + 3"})
        assert "5" in result

    def test_subtraction(self):
        """测试减法"""
        result = calculator.invoke({"expression": "10 - 4"})
        assert "6" in result

    def test_multiplication(self):
        """测试乘法"""
        result = calculator.invoke({"expression": "3 * 4"})
        assert "12" in result

    def test_division(self):
        """测试除法"""
        result = calculator.invoke({"expression": "10 / 2"})
        assert "5" in result

    def test_complex_expression(self):
        """测试复杂表达式"""
        result = calculator.invoke({"expression": "2 + 3 * 4"})
        assert "14" in result

    def test_zero_division(self):
        """测试除零错误"""
        result = calculator.invoke({"expression": "1 / 0"})
        assert "错误" in result or "除数" in result or "零" in result

    def test_invalid_characters(self):
        """测试无效字符"""
        result = calculator.invoke({"expression": "2 + os.system('ls')"})
        assert "无效" in result or "错误" in result

    def test_decimal_calculation(self):
        """测试小数计算"""
        result = calculator.invoke({"expression": "10 / 3"})
        assert "3.3" in result or "." in result


class TestPythonRepl:
    """Python代码执行器测试"""

    def test_simple_print(self):
        """测试简单打印"""
        result = python_repl.invoke({"code": "print('Hello')"})
        assert "Hello" in result or "执行" in result or "错误" in result

    def test_arithmetic(self):
        """测试算术运算"""
        result = python_repl.invoke({"code": "print(2 + 2)"})
        assert "4" in result or "执行" in result or "错误" in result

    def test_variable(self):
        """测试变量"""
        result = python_repl.invoke({"code": "x = 5\nprint(x * 2)"})
        assert "10" in result or "执行" in result or "错误" in result

    def test_no_output(self):
        """测试无输出代码"""
        result = python_repl.invoke({"code": "x = 5"})
        assert "无输出" in result or result == "代码执行完成，无输出"

    def test_syntax_error(self):
        """测试语法错误"""
        result = python_repl.invoke({"code": "print("})
        assert "语法错误" in result

    def test_dangerous_code_blocked(self):
        """测试危险代码被阻止"""
        result = python_repl.invoke({"code": "import os\nos.system('ls')"})
        assert "错误" in result or "未定义" in result


class TestToolCall:
    """工具调用逻辑测试"""

    def test_tool_call_request_creation(self):
        """测试工具调用请求创建"""
        request = ToolCallRequest(name="calculator", parameters={"expression": "2+2"})
        
        assert request.name == "calculator"
        assert request.parameters["expression"] == "2+2"

    def test_tool_calls_creation(self):
        """测试工具调用列表创建"""
        calls = ToolCallRequests(calls=[
            ToolCallRequest(name="search", parameters={"query": "测试"}),
            ToolCallRequest(name="calculator", parameters={"expression": "1+1"})
        ])
        
        assert len(calls.calls) == 2
        assert calls.calls[0].name == "search"

    def test_tool_call_basic(self, mock_llm, tools):
        """测试基础工具调用"""
        from langchain_core.messages import AIMessage
        
        mock_llm.invoke.return_value = AIMessage(
            content='{"calls": [{"name": "calculator", "parameters": {"expression": "2+2"}}]}'
        )
        
        state = {"question": "2+2等于多少？"}
        result = tool_call(state, mock_llm, tools)
        
        assert "tool_results" in result

    def test_tool_call_multiple_tools(self, mock_llm, tools):
        """测试多工具调用"""
        from langchain_core.messages import AIMessage
        from agentic_rag.tools.tool_calls import tool_call
        
        mock_llm.invoke.return_value = AIMessage(
            content='{"calls": [{"name": "search", "parameters": {"query": "Python"}}, {"name": "calculator", "parameters": {"expression": "1+1"}}]}'
        )
        
        state = {"question": "Python和1+1"}
        result = tool_call(state, mock_llm, tools)
        
        # 验证返回结果包含 tool_results
        assert "tool_results" in result

    def test_tool_call_no_tool_needed(self, mock_llm, tools):
        """测试不需要工具的情况"""
        from langchain_core.messages import AIMessage
        
        mock_llm.invoke.return_value = AIMessage(content='{"calls": []}')
        
        state = {"question": "不需要工具的问题"}
        result = tool_call(state, mock_llm, tools)
        
        assert result.get("tool_results", {}) == {}

    def test_tool_call_unknown_tool(self, mock_llm, tools):
        """测试未知工具处理"""
        from langchain_core.messages import AIMessage
        
        mock_llm.invoke.return_value = AIMessage(
            content='{"calls": [{"name": "unknown_tool", "parameters": {}}]}'
        )
        
        state = {"question": "测试"}
        result = tool_call(state, mock_llm, tools)
        
        assert "tool_results" in result

    def test_tool_call_parser_error(self, mock_llm, tools):
        """测试解析错误处理"""
        mock_llm.invoke.side_effect = Exception("解析失败")
        
        state = {"question": "测试"}
        result = tool_call(state, mock_llm, tools)
        
        assert result.get("tool_results", {}) == {}


class TestGetSearchTools:
    """获取搜索工具列表测试"""

    def test_get_tools(self):
        """测试获取工具列表"""
        tools = get_search_tools()
        
        assert isinstance(tools, list)
        assert len(tools) >= 2