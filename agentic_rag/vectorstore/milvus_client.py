"""
Milvus向量数据库客户端

连接管理策略：
- 采用单例模式而非传统连接池（与PostgreSQL/Redis不同）
- Milvus使用gRPC长连接，连接建立开销大，更适合维护单个连接
- LangChain内部已经处理了连接复用
- 不同collection_name对应不同的MilvusClient实例
"""
from dotenv import load_dotenv
import os
import threading
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Milvus

from loguru import logger
from .embeddings import get_embeddings


class MilvusClient:
    """Milvus向量数据库封装（非线程安全，每次创建新实例）"""
    
    _instances: Dict[str, 'MilvusClient'] = {}  # 单例缓存（按collection_name）
    _lock = threading.Lock()

    def __init__(
        self, 
        collection_name: str = "table_agentic_rag",  # 向量数据库表名
        embedding_model: Any = "embedding-3",  # 接受字符串或嵌入模型实例
        connection_args: Optional[Dict[str, Any]] = None,
        index_params: Optional[Dict[str, Any]] = None  # 索引参数配置
    ):
        self.collection_name = collection_name

        if isinstance(embedding_model, str):
            self.embedding_model = get_embeddings(
                embedding_type="zhipu",
                model_name=embedding_model,
                api_key=os.getenv("ZHIPUAI_API_KEY"),
            )
        else:
            self.embedding_model = embedding_model
        
        if connection_args is None:
            connection_args = {
                "host": "localhost",
                "port": 19530,
                "user": "root",
                "password": "",
                "db_name": "db_agentic_rag",
            }
        
        self.connection_args = connection_args
        
        # 默认使用IVF_FLAT索引
        if index_params is None:
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {
                    "nlist": 1024  # 聚类中心数量
                }
            }
        
        self.index_params = index_params
        self.vectorstore: Optional[Milvus] = None  # LangChain Milvus实例
        
        # 连接统计
        self._search_count = 0
        self._last_search_time = 0.0

    def _normalize_metadata(self, documents: List[Document]) -> List[Document]:
        """
        统一文档元数据字段,确保所有文档的metadata结构一致,防止Milvus schema不匹配

        Milvus要求同一collection中所有文档的metadata字段名和类型必须一致,
        不同文件类型的loader会产生不同的metadata字段,直接插入会导致DataNotMatchException

        参数:
            documents: 原始文档列表

        返回:
            元数据统一后的文档列表
        """
        # 定义统一的metadata字段及默认值,所有文档都会包含这些字段
        unified_fields = {
            "source": "",
            "file_type": "unknown",
            "page": 0,
            "total_pages": 0,
            "paragraphs": 0,
            "chunk_id": 0,
            "total_chunks": 0,
        }

        normalized = []
        for doc in documents:
            # 保留原始metadata中与统一字段匹配的值
            new_meta = dict(unified_fields)
            for key in unified_fields:
                if key in doc.metadata:
                    val = doc.metadata[key]
                    # Milvus scalar字段不支持list类型,转为字符串
                    if isinstance(val, list):
                        val = ", ".join(str(v) for v in val)
                    new_meta[key] = val

            # 保留langchain_primaryid(Milvus必需字段)
            if "langchain_primaryid" in doc.metadata:
                new_meta["langchain_primaryid"] = doc.metadata["langchain_primaryid"]

            # 从source推断file_type(如果metadata中没有)
            if new_meta["file_type"] == "unknown" and new_meta["source"]:
                from pathlib import Path
                ext = Path(new_meta["source"]).suffix.lower().lstrip(".")
                type_map = {
                    "pdf": "pdf", "docx": "docx", "txt": "txt",
                    "md": "markdown", "csv": "csv", "xlsx": "excel",
                    "xls": "excel", "xlsm": "excel", "html": "html", "htm": "html"
                }
                new_meta["file_type"] = type_map.get(ext, ext)

            normalized.append(Document(
                page_content=doc.page_content,
                metadata=new_meta
            ))

        return normalized

    def from_documents(
        self,
        documents: List[Document],
        drop_old: bool = False # 是否删除旧的向量数据库
        ) -> Milvus:
        """
        从文档创建向量存储

        参数：
            documents: 文档列表
            drop_old: 是否删除旧的向量数据库

        返回：
            Milvus向量存储实例

        说明：
            注意：langchain-community 0.4.x版本要求当auto_id=False时必须传递ids参数
        """
        import uuid
        from loguru import logger

        # 统一metadata字段,防止schema不匹配
        documents = self._normalize_metadata(documents)

        ids = [str(uuid.uuid4()) for _ in documents]
        for doc, doc_id in zip(documents, ids):
            doc.metadata["langchain_primaryid"] = doc_id

        logger.info(f"创建向量存储: {self.collection_name}, 文档数: {len(documents)}, 维度: {self.embedding_model.dimension}, 索引类型: {self.index_params.get('index_type', 'AUTO')}")

        self.vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            collection_name=self.collection_name, 
            connection_args=self.connection_args,
            drop_old=drop_old,
            ids=ids,
            index_params=self.index_params
        )
        return self.vectorstore
    
    def from_existing_collection(self) -> Milvus:
        """
        从已存在的collection加载向量存储,只建立连接，不做任何数据操作

        返回：
            Milvus向量存储实例

        说明：
            如果collection不存在，抛出异常
        """
        from loguru import logger
        
        try:
            self.vectorstore = Milvus(
                embedding_function=self.embedding_model,
                collection_name=self.collection_name,
                connection_args=self.connection_args
            )
            logger.info(f"成功加载已存在的collection: {self.collection_name}")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载collection '{self.collection_name}' 失败: {str(e)}")
            raise ValueError(f"向量库collection '{self.collection_name}' 不存在或无法加载。请先上传文档初始化向量库。")
    
    def load_or_initialize(self) -> bool:
        """
        尝试加载已存在的collection，如果不存在则返回False

        返回：
            bool: 是否成功加载了已存在的collection
        """
        from loguru import logger
        
        try:
            self.from_existing_collection()
            return True
        except Exception as e:
            logger.info(f"未找到已存在的向量库collection: {self.collection_name}")
            return False
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档到向量存储

        参数：
            documents: 文档列表

        返回：
            文档IDs列表

        说明：
            如果向量库未初始化，先调用from_documents初始化
            注意：langchain-community 0.4.x版本要求当auto_id=False时必须传递ids参数
        """
        import uuid
        from loguru import logger

        # 统一metadata字段,防止schema不匹配
        documents = self._normalize_metadata(documents)

        ids = []
        for doc in documents:
            if doc.metadata.get("langchain_primaryid") is None:
                doc_id = str(uuid.uuid4())
                doc.metadata["langchain_primaryid"] = doc_id
                ids.append(doc_id)
            else:
                ids.append(doc.metadata["langchain_primaryid"])
        
        if not hasattr(self, 'vectorstore') or self.vectorstore is None:
            self.from_documents(documents=documents, drop_old=False)
            return ids
        
        try:
            return self.vectorstore.add_documents(documents, ids=ids)
        except Exception as e:
            error_msg = str(e)
            # 如果是schema不匹配错误,尝试删除旧collection重新创建
            if "DataNotMatch" in error_msg or "schema" in error_msg.lower():
                logger.warning(f"向量库schema不匹配,将重建collection: {error_msg}")
                self.from_documents(documents=documents, drop_old=True)
                return ids
            raise

    def as_retriever(
        self,
        search_type: str = "similarity", # 搜索方式: similarity(相似度搜索)等
        search_kwargs: Dict[str, Any] = {} # 搜索参数
        ) -> BaseRetriever:
        """生成一个 "检索器对象"，给 LangChain 的链 / Agent 用"""

        if not hasattr(self, 'vectorstore'): # hasattr 是 Python 内置的函数,专门用来判断一个对象有没有某个指定的属性 / 方法。
            raise ValueError("向量库未初始化。请先调用 from_documents 方法.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filters: Optional[Dict[str, Any]] = None,  # 过滤条件
        nprobe: int = 16  # IVF_FLAT索引的搜索参数：查询的聚类中心数量
    ) -> List[Document]:
        """
        相似度搜索，返回文档
        
        参数：
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件
            nprobe: IVF_FLAT索引的搜索参数，值越大搜索越精确但速度越慢
            
        注意：
            此方法通过LangChain内部复用gRPC连接，无需手动管理连接池
            连接在首次使用时建立，后续操作自动复用
        """
        if not hasattr(self, 'vectorstore') or self.vectorstore is None:
            raise ValueError("向量库未初始化。请先调用 from_documents 或 from_existing_collection 方法.")
        
        search_kwargs = {}
        if self.index_params.get("index_type") == "IVF_FLAT":
            search_kwargs["nprobe"] = nprobe
        
        import time
        start_time = time.time()
        
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            expr=filters,
            **search_kwargs
        )
        
        # 性能监控
        self._search_count += 1
        self._last_search_time = time.time() - start_time
        
        logger.debug(
            f"向量检索完成: collection={self.collection_name}, "
            f"query长度={len(query)}, k={k}, "
            f"结果数={len(results)}, 耗时={self._last_search_time:.3f}s, "
            f"累计检索={self._search_count}次"
        )
        
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        nprobe: int = 16 # IVF_FLAT索引的搜索参数
    ) -> List[tuple]:
        """
        带分数的相似度搜索，返回文档 + 相似度分数
        
        参数：
            query: 查询文本
            k: 返回结果数量
            nprobe: IVF_FLAT索引的搜索参数
        """
        if not hasattr(self, 'vectorstore'):
            raise ValueError("向量库未初始化。请先调用 from_documents 方法.")
        
        search_kwargs = {}
        if self.index_params.get("index_type") == "IVF_FLAT":
            search_kwargs["nprobe"] = nprobe
        
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            **search_kwargs
        )
    
    def delete_collection(self):
        """删除集合,删除所有文档"""
        if hasattr(self, 'vectorstore'):
            self.vectorstore.delete_collection()

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        if not hasattr(self, 'vectorstore'):
            raise ValueError("向量库未初始化.")
        
        return {
            "collection_name": self.collection_name,
            "dimension": self.embedding_model.dimension,
            "count": self.vectorstore.col().num_entities  # 返回当前集合里的向量总数（也就是你存入的文档片段数量）。
        }


    @classmethod
    def get_instance(
        cls,
        collection_name: str = "table_agentic_rag",
        embedding_model: Any = "embedding-3",
        connection_args: Optional[Dict[str, Any]] = None,
        index_params: Optional[Dict[str, Any]] = None
    ) -> 'MilvusClient':
        """
        获取MilvusClient单例实例（线程安全）
        
        采用单例模式的原因：
        - Milvus使用gRPC长连接，频繁创建/销毁连接开销大
        - 保持单个连接实例，所有检索操作复用同一连接
        - 不同collection_name对应不同实例
        
        Args:
            collection_name: 向量数据库表名
            embedding_model: 嵌入模型（字符串或实例）
            connection_args: 连接参数
            index_params: 索引参数
            
        Returns:
            MilvusClient单例实例
        """
        with cls._lock:
            # 根据collection_name作为key实现单例
            if collection_name not in cls._instances:
                logger.info(f"创建新的MilvusClient实例: collection={collection_name}")
                cls._instances[collection_name] = cls(
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                    connection_args=connection_args,
                    index_params=index_params
                )
            else:
                logger.debug(f"复用已存在的MilvusClient实例: collection={collection_name}")
            
            return cls._instances[collection_name]


def get_vectorstore(
    collection_name: str = "table_agentic_rag",
    embedding_model: Any = "embedding-3",
    connection_args: Optional[Dict[str, Any]] = None,
    index_params: Optional[Dict[str, Any]] = None
) -> MilvusClient:
    """
    获取向量存储单例实例（推荐方式）
    
    与直接创建MilvusClient不同，此函数保证：
    - 同一collection_name只创建一个实例
    - 所有检索操作复用同一gRPC连接
    - 避免频繁创建/销毁连接的开销
    
    Args:
        collection_name: 向量数据库表名
        embedding_model: 嵌入模型（字符串或实例）
        connection_args: 连接参数
        index_params: 索引参数
        
    Returns:
        MilvusClient单例实例
        
    示例:
        # 推荐用法
        vectorstore = get_vectorstore("my_collection", embeddings)
        results = vectorstore.similarity_search("query", k=5)
    """
    return MilvusClient.get_instance(
        collection_name=collection_name,
        embedding_model=embedding_model,
        connection_args=connection_args,
        index_params=index_params
    )
