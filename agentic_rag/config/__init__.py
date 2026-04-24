"""
Config模块 - 配置管理

包含：
- settings: 应用配置
- logger_config: 日志配置
"""

from .settings import get_settings

__all__ = ["get_settings"]
