"""
重排模型实现
使用BGE-reranker进行结果重排
"""

from typing import List, Tuple
from langchain_core.documents import Document
import httpx
import math

class Reranker:
    """重排基类"""
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Tuple[Document, float]]:
        """重排文档
        
        Args:
            query: 查询文本
            documents: 候选文档
            top_k: 返回前k个文档
        
        Returns:
            重排后的文档及其分数
        """
        raise NotImplementedError

class BGGReranker(Reranker):
    """
    BGE-reranker重排器:这是一个基于 BGE(BAAI General Embedding)重排模型 实现的 ** 重排器(Reranker)** 类
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "rerank",
        top_k: int = 3
         ):
        # from sentence_transformers import CrossEncoder
        # CrossEncoder 是 sentence-transformers 提供的交叉编码器接口，专门用于「文本对相关性打分」，也就是我们说的重排（Rerank）。 
        self.model = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def normalize_score(score: float) -> float:
        """
        把任意分数归一化到 0 ~ 1 区间
        适用于:BGE / GLM / Qwen / GTE 所有重排模型
        """
        try:
            return 1 / (1 + math.exp(-score))  # sigmoid
        except:
            return 0.0
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Tuple[Document, float]]:
        """重排文档(OpenAI 兼容 API 版本)"""
        if not documents:
            return []

        doc_texts = [doc.page_content for doc in documents]

        # 使用 httpx 直接调用重排 API（/rerank 不是 OpenAI 标准 API，不应通过 OpenAI 客户端调用）
        rerank_url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "top_n": top_k
        }
        response = httpx.post(rerank_url, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status() # 这一行是错误处理：如果 API 返回了 4xx/5xx 等错误状态码（比如鉴权失败、请求格式错误、服务端报错），这行代码会直接抛出异常，方便你定位问题。
        result = response.json()
        # {
        #     "model": "rerank",
        #     "results": [
        #         {"index": 0, "relevance_score": 0.98},
        #         {"index": 2, "relevance_score": 0.85},
        #         {"index": 1, "relevance_score": 0.72}
        #     ]
        # }

        # 组装结果
        doc_scores = []
        for item in result["results"]:
            doc = documents[item["index"]]
            score = item["relevance_score"]
            score = self.normalize_score(score)
            if score > 0.7:
                doc_scores.append((doc, score))

        # 排序并返回
        sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    
        if len(sorted_docs) < top_k:
            top_k = len(sorted_docs)
        return sorted_docs[:top_k]


class SimpleReranker(Reranker):
    """简单的基于关键词的重排"""
    
    def __init__(self):
        pass
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Tuple[Document, float]]:
        """基于关键词重叠重排"""
        query_words = set(query.lower().split())
        
        scored_docs = []
        for doc in documents:
            doc_words = set(doc.page_content.lower().split())
            # Jaccard相似度:J(A,B)= ∣A∩B∣/∣A∪B∣
            overlap = len(query_words & doc_words) # 查询词和文档词的交集大小（共同出现的词数）
            total = len(query_words | doc_words) # 查询词和文档词的并集大小（所有词数）
            score = overlap / total if total > 0 else 0
            scored_docs.append((doc, score))
        
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

def get_reranker(reranker_type: str = "bge", **kwargs) -> Reranker:
    """获取重排器实例"""
    rerankers = {
        "bge": BGGReranker,
        "simple": SimpleReranker
    }
    
    if reranker_type not in rerankers:
        raise ValueError(f"未知的重排序类型: {reranker_type}")
    
    return rerankers[reranker_type](**kwargs)