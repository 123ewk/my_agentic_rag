"""
VectorStore模块 - 向量存储管理

包含：
- embeddings: 嵌入向量生成
- milvus_client: Milvus向量数据库客户端
"""

from .embeddings import get_embeddings
from .milvus_client import get_vectorstore

__all__ = ["get_embeddings", "get_vectorstore"]
