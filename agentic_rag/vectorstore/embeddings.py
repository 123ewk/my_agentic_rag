"""
嵌入模型配置
支持本地模型和API方式
"""

from abc import ABC, abstractmethod
from typing import List
import re
import numpy as np
import os
from openai import OpenAI
from loguru import logger


def _clean_text(text: str) -> str:
    """
    清洗文本: 去除控制字符、零宽字符、多余空白，防止嵌入API因非法字符拒绝请求

    参数:
        text: 原始文本

    返回:
        清洗后的文本
    """
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u2060-\u206f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class BaseEmbeddings(ABC):
    """嵌入模型基类"""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        pass

class HuggingFaceEmbeddings(BaseEmbeddings):
    """
    HuggingFace嵌入模型(本地模型)
    原理:SentenceTransformer 会自动去 Hugging Face Hub 下载模型权重，然后在你本地的 Python 环境里运行。
    """

    def __init__(self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cuda", # 控制模型跑在 CPU(cpu)/GPU(cuda) 上
        normalize_embeddings: bool = True  # 这个参数控制是否对生成的向量做L2 归一化,保证向量相似度计算的准确性。
    ) -> None:

        # SentenceTransformer 专门用来把句子 / 段落转换成向量（嵌入），是目前做语义相似度、RAG 检索最主流的工具之一
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device) # 初始化并加载你指定的嵌入模型。
        self.normalize_embeddings = normalize_embeddings
        self.dimension = self.model.get_sentence_embedding_dimension() # 向量维度,获取模型输出向量的维度，并保存下来。

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False # 不显示进度条
        )
        return embedding.tolist() # 将向量转换为列表，方便后续使用。

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True, # 显示进度条
            batch_size=32 # 每次处理32个文档，根据实际情况调整
        )
        return embeddings.tolist() # 将向量转换为列表，方便后续使用。


class ZhiPuEmbeddings(BaseEmbeddings):
    """
    智谱嵌入模型

    关键限制:
        - Embedding-3 单次请求最多64条文本
        - 单条文本最多3072 tokens
        - dimensions 参数控制输出向量维度,默认2048,可选256/512/1024/2048
    """

    # 智谱API单次请求最大文本条数
    MAX_BATCH_SIZE = 64

    def __init__(self,
        api_key: str = None,
        model_name: str = "embedding-3",
        dimension: int = 1024  # 向量维度,根据模型输出向量的维度设置。
    ) -> None:
        self.api_key = api_key or os.environ.get("ZHIPUAI_API_KEY", "")
        self.model_name = model_name
        self.dimension = dimension
        self.client = OpenAI(api_key=self.api_key, base_url="https://open.bigmodel.cn/api/paas/v4")

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询

        参数:
            text: 查询文本

        返回:
            嵌入向量列表
        """
        cleaned = _clean_text(text)
        if not cleaned:
            raise ValueError("嵌入查询失败: 文本清洗后为空")

        response = self.client.embeddings.create(
            model=self.model_name,
            input=cleaned,
            dimensions=self.dimension
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档,自动分批处理(每批≤64条),清洗非法字符

        参数:
            texts: 文本列表

        返回:
            嵌入向量列表,顺序与输入一致

        异常:
            ValueError: 所有文本清洗后均为空时抛出
        """
        # 清洗每条文本,过滤空文本,同时记录原始索引以保持顺序
        cleaned_pairs = []
        for idx, text in enumerate(texts):
            cleaned = _clean_text(text)
            if cleaned:
                cleaned_pairs.append((idx, cleaned))

        if not cleaned_pairs:
            raise ValueError("嵌入文档失败: 所有文本清洗后均为空")

        # 分批调用API,每批不超过MAX_BATCH_SIZE条
        all_embeddings = [None] * len(texts)
        total_batches = (len(cleaned_pairs) + self.MAX_BATCH_SIZE - 1) // self.MAX_BATCH_SIZE

        for batch_idx in range(total_batches):
            start = batch_idx * self.MAX_BATCH_SIZE
            end = min(start + self.MAX_BATCH_SIZE, len(cleaned_pairs))
            batch_pairs = cleaned_pairs[start:end]
            batch_texts = [t for _, t in batch_pairs]
            batch_indices = [i for i, _ in batch_pairs]

            logger.debug(f"嵌入批次 {batch_idx + 1}/{total_batches}, 文本数: {len(batch_texts)}")

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch_texts,
                dimensions=self.dimension
            )

            for i, embedding_obj in enumerate(response.data):
                all_embeddings[batch_indices[i]] = embedding_obj.embedding

        # 过滤掉原始空文本对应的None,只返回有效嵌入
        valid_embeddings = [e for e in all_embeddings if e is not None]
        if len(valid_embeddings) != len(texts):
            skipped = len(texts) - len(valid_embeddings)
            logger.warning(f"嵌入时跳过 {skipped} 条空文本")

        return valid_embeddings

def get_embeddings(embedding_type: str = "zhipu", **kwargs) -> BaseEmbeddings:
    """
    获取嵌入模型实例

    参数:
        embedding_type: 嵌入类型,可选 "huggingface" 或 "zhipu"
        **kwargs: 传递给嵌入模型构造函数的参数

    返回:
        嵌入模型实例

    异常:
        ValueError: 未知的嵌入类型时抛出
    """
    embeddings = {
        "huggingface": HuggingFaceEmbeddings,
        "zhipu": ZhiPuEmbeddings
    }

    if embedding_type not in embeddings:
        raise ValueError(f"未知的嵌入类型::{embedding_type}")

    return embeddings[embedding_type](**kwargs)
