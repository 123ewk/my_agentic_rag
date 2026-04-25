"""
日志配置
"""
from loguru import logger
import sys
import traceback # 异常回溯

def setup_logging(log_level: str = "INFO"):
    """
    配置日志系统(只需在main.py调用一次)
    
    Args:
        log_level: 日志级别，默认为INFO，可选值：DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台输出（包含异常堆栈跟踪）
    logger.add(
        sys.stderr, # 控制台输出
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n{exception}", 
        level=log_level, # 日志级别
        colorize=True, # 开启颜色输出
        backtrace=True, # 开启异常回溯
        diagnose=True # 开启诊断信息
    )
    
    # 添加文件输出（包含异常堆栈跟踪）
    logger.add(
        "logs/agentic_rag_{time:YYYY-MM-DD}.log", # 日志文件路径
        encoding="utf-8", # 编码
        rotation="00:00",  # 每天轮转
        retention="30 days",  # 保留30天, 旧日志自动删除，节省磁盘空间
        compression="zip",  # 压缩, 压缩日志文件，节省磁盘空间
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}", # 日志格式
        level=log_level, # 日志级别
        backtrace=True, # 开启异常回溯
        diagnose=True # 开启诊断信息
    )
    
    return logger
    