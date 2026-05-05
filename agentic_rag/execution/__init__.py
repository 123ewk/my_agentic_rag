"""
执行模块包
包含并行执行器等执行优化组件
"""
from .parallel_executor import ParallelExecutor, ParallelExecutionResult

__all__ = ["ParallelExecutor", "ParallelExecutionResult"]
