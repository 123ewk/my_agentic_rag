"""
搜索工具实现
"""
from typing import Dict, Any
from langchain_core.tools import tool
from ddgs import DDGS

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
        import re
        # 安全检查：只允许数字、基本运算符和括号（禁止**幂运算防止DoS）
        if not re.match(r'^[\d+\-*/().\s]+$', expression):
            return "错误：表达式包含无效字符"
        
        # 禁止幂运算符，防止 **999999999 导致DoS
        if '**' in expression:
            return "错误：不支持幂运算"
        
        # 限制表达式长度
        if len(expression) > 100:
            return "错误：表达式过长"
        
        # 限制嵌套深度，防止递归炸弹
        if expression.count('(') > 10:
            return "错误：括号嵌套过深"
        
        # 计算结果
        result = eval(expression, {"__builtins__": {}})
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
    import io
    import sys
    
    # 危险代码黑名单检测
    dangerous_patterns = [
        r'__\w+__',                          # 双下划线属性访问（如 __class__, __subclasses__）
        r'import\s+',                        # import语句
        r'from\s+\w+\s+import',              # from xxx import
        r'exec\s*\(',                        # exec调用
        r'eval\s*\(',                        # eval调用
        r'compile\s*\(',                     # compile调用
        r'open\s*\(',                        # 文件操作
        r'os\.',                             # os模块
        r'sys\.',                            # sys模块（除了被重定向的stdout）
        r'subprocess',                       # 子进程
        r'__import__',                       # 动态导入
        r'globals\s*\(',                     # globals访问
        r'locals\s*\(',                      # locals访问
        r'getattr\s*\(',                     # getattr动态属性访问
        r'setattr\s*\(',                     # setattr动态属性设置
        r'delattr\s*\(',                     # delattr动态属性删除
        r'type\s*\(',                        # type动态类型创建
    ]
    
    import re
    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            return f"安全限制：代码包含禁止的操作模式 ({pattern})"
    
    try:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        # 加固沙箱：限制可用内置函数
        safe_builtins = {
            'print': print,
            'range': range,
            'len': len,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'round': round,
            'isinstance': isinstance,
            'type': type,
        }
        
        exec(code, {"__builtins__": safe_builtins})
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        if output:
            return output
        else:
            return "代码执行完成，无输出"
    
    except SyntaxError as e:
        sys.stdout = old_stdout
        return f"语法错误: {str(e)}"
    except Exception as e:
        sys.stdout = old_stdout
        return f"执行错误: {str(e)}"


def get_search_tools() -> Dict[str, Any]:
    """获取搜索工具字典"""
    return {
        "duckduckgo_search": duckduckgo_search,
        "calculator": calculator,
        "python_repl": python_repl
    }