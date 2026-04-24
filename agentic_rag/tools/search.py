"""
搜索工具实现
"""
from typing import Dict, Any
from langchain_core.tools import tool
from duckduckgo_search import DDGS

# DDGS 全称是 DuckDuckGo Search,是一个免费、开源、无需 API 密钥的网络搜索工具
@tool(description="使用DuckDuckGo搜索互联网信息")
def duckduckgo_search(query: str) -> str:
    """使用DuckDuckGo搜索互联网信息
    
    
    Args:
        query: 搜索查询
        
    Returns:
        搜索结果摘要
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        
        if not results:
            return "未找到相关结果"
        
        # 格式化结果
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', '')
            body = result.get('body', '')
            url = result.get('href', '')
            formatted.append(f"{i}. {title}\n   {body}\n   来源: {url}")
        
        return "\n\n".join(formatted)
    
    except Exception as e:
        return f"搜索出错: {str(e)}"


@tool(description="安全计算器")
def calculator(expression: str) -> str:
    """安全计算器
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
        
    Returns:
        计算结果
    """
    try:
        # 安全检查：只允许数字和运算符
        import re
        if not re.match(r'^[\d+\-*/().\s]+$', expression):
            return "错误：表达式包含无效字符"
        
        # 计算结果
        result = eval(expression)
        return f"{expression} = {result}"
    
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool(description="Python代码执行器(沙箱环境)")
def python_repl(code: str) -> str:
    """Python代码执行器(沙箱环境)
    
    Args:
        code: Python代码
        
    Returns:
        代码执行结果
    """
    import io # io:输入输出模块，用来创建一个内存缓冲区，存储 print 的内容
    import sys # sys: Python 系统模块，用来接管程序的输出（print 打印的内容）
    
    try:
        # 捕获输出
        # sys.stdout 是 Python 默认的标准输出（所有 print() 都会打印到这里）
        old_stdout = sys.stdout # 第一步：把原来的输出保存到 old_stdout（执行完要恢复，不然主程序没法打印）
        sys.stdout = io.StringIO() # 第二步：把输出重定向到一个内存字符串流 io.StringIO()→ 意思是：接下来所有 print() 不会显示在控制台，而是存到这个缓冲区里！
        
        # 执行代码（限制时间和资源）,安全执行代码（最关键）
        # exec()：Python 内置函数，执行字符串格式的代码

        exec(code, {"__builtins__": {}}) 
        # {"__builtins__": {}}：沙箱核心！
        # __builtins__ 是 Python 所有内置函数 / 功能的集合（print、open、os、import 都在这）
        # 把它设为空字典 = 禁用所有 Python 内置功能 → 代码里不能用print之外的危险操作（比如读写文件、删数据、联网），非常安全
        
        # 获取输出
        output = sys.stdout.getvalue() # getvalue()：从缓冲区里拿出所有被捕获的print内容
        sys.stdout = old_stdout # 把 sys.stdout 恢复成原来的样子（避免影响主程序）
        
        if output:
            return output
        else:
            return "代码执行完成，无输出"
    
    except SyntaxError as e:
        return f"语法错误: {str(e)}"
    except Exception as e:
        return f"执行错误: {str(e)}"


def get_search_tools() -> Dict[str, Any]:
    """获取搜索工具字典"""
    return {
        "duckduckgo_search": duckduckgo_search,
        "calculator": calculator,
        "python_repl": python_repl
    }