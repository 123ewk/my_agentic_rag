"""
Milvus向量数据库客户端
"""
from dotenv import load_dotenv
import os

load_dotenv()


from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Milvus


from .embeddings import get_embeddings

class MilvusClient:
    """Milvus向量数据库封装"""

    def __init__(
        self, 
        collection_name: str = "table_agentic_rag", # 向量数据库表名
        embedding_model: Any = "embedding-3", # 接受字符串或嵌入模型实例
        connection_args: Optional[Dict[str, Any]] = None
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

        logger.info(f"创建向量存储: {self.collection_name}, 文档数: {len(documents)}, 维度: {self.embedding_model.dimension}")

        self.vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            collection_name=self.collection_name, 
            connection_args=self.connection_args,
            drop_old=drop_old,
            ids=ids
        )
        return self.vectorstore
    
    def from_existing_collection(self) -> Milvus:
        """
        从已存在的collection加载向量存储

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
            search_kwargs = {"k": 5}
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None # 过滤条件
    ) -> List[Document]:
        """相似度搜索,直接检索，返回文档"""
        if not hasattr(self, 'vectorstore'):
            raise ValueError("向量库未初始化。请先调用 from_documents 方法.")
        
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            expr=filters
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple]:
        """带分数的相似度搜索,直接检索，返回文档 + 相似度分数"""
        if not hasattr(self, 'vectorstore'):
            raise ValueError("向量库未初始化。请先调用 from_documents 方法.")
        
        return self.vectorstore.similarity_search_with_score(query=query, k=k)
    
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

def get_vectorstore(
    collection_name: str = "table_agentic_rag",
    embedding_model: str = "embedding-3",
    connection_args: Optional[Dict[str, Any]] = None
) -> MilvusClient:
    """获取向量存储实例"""
    return MilvusClient(
        collection_name=collection_name,
        embedding_model=embedding_model,
        connection_args=connection_args
    )
