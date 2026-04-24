"""
混合检索实现
结合向量检索和关键词检索
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever # 基于关键词匹配,不需要向量数据库,擅长处理：关键词明确、术语多的场景
from langchain_core.embeddings import Embeddings # Embeddings 是 LangChain 定义的所有 Embedding 模型的抽象基类（ABC）
import numpy as np


class HybridSearchRetriever:
    """混合检索器"""
    def __init__(
        self,
        vector_retriever,
        embeddings: Embeddings,
        vector_weight: float = 0.6, # 向量检索权重
        keyword_weight: float = 0.4, # 关键词检索权重
    ) -> None:
        self.vector_retriever = vector_retriever
        self.embeddings = embeddings
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

        self.bm25_retriever = None
        self.documents = []
        
        # 【关键】预计算所有文档的向量，只算一次！
        self.doc_embeddings = self._precompute_doc_embeddings()

        def add_documents(self, documents: List[Document]):
            """添加文档"""
            self.documents = documents
        
            # 初始化BM25检索器
            if documents:
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]
                
                self.bm25_retriever = BM25Retriever.from_texts(
                    texts=texts,
                    metadatas=metadatas
                )
        
        def _precompute_doc_embeddings(self) -> List[np.ndarray]:
            """预计算所有文档的向量(只算一次！,只是用于文档固定,不常更新的场景)"""
            if not self.documents:
                return []
            
            # 批量计算，比循环单个调用效率高很多
            return self.embeddings.embed_documents(
                [doc.page_content[:500] for doc in self.documents]
            )

        def _get_vector_scores(self, query: str, k: int = 10) -> List[float]:
            """获取向量检索分数"""
            if not self.documents or not self.doc_embeddings:
                return []
            
            # 计算查询向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 批量计算所有相似度（比循环调用函数快很多）
            scores = [
                self._cosine_similarity(query_embedding, doc_emb)
                for doc_emb in self.doc_embeddings
            ]
            
            # 取Top-K（argsort默认升序，所以取最后k个）
            top_indices = np.argsort(scores)[-k:]
            return [(idx, scores[idx]) for idx in top_indices]
        
        @staticmethod
        def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            """计算余弦相似度"""
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            
            return dot_product / (norm1 * norm2)

        def _get_bm25_scores(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
            """获取BM25分数"""
            if self.bm25_retriever is None:
                return []
            
            # 执行BM25检索,调用 BM25Retriever 的 invoke 方法，用关键词匹配的方式，返回和 query 最相关的 Top-K 文档
            docs = self.bm25_retriever.invoke(query)
            return [(i, 1.0 / (i + 1)) for i, doc in enumerate(docs)]
        
        def invoke(self, query: str, k: int = 5) -> List[Document]:
            """混合检索"""
            if not self.documents:
                return []
            
            # 获取两种检索的结果
            vector_scores = self._get_vector_scores(query, k=len(self.documents))
            bm25_scores = self._get_bm25_scores(query, k=len(self.documents))
            
            # 归一化分数
            vector_max = max(score for _, score in vector_scores) if vector_scores else 1
            bm25_max = max(score for _, score in bm25_scores) if bm25_scores else 1
            
            # 计算综合分数
            combined_scores = {}
            
            for idx, score in vector_scores:
                combined_scores[idx] = combined_scores.get(idx, 0) + \
                                    self.vector_weight * (score / vector_max)
            
            for idx, score in bm25_scores:
                combined_scores[idx] = combined_scores.get(idx, 0) + \
                                    self.keyword_weight * (score / bm25_max)
            
            # 排序并返回top_k
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_indices = [idx for idx, _ in sorted_results[:k]]
            
            return [self.documents[idx] for idx in top_k_indices]
