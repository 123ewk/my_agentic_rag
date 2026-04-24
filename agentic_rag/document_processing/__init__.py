"""
DocumentProcessing模块 - 文档处理

包含：
- loaders: 文档加载器
- splitters: 文档分块器
"""

from .loaders import get_document_loader
from .splitters import get_splitter

__all__ = ["get_document_loader", "get_splitter"]
