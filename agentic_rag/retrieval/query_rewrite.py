"""
查询改写模块
支持HyDE、Query Expansion等多种策略
"""
import re
import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

from ..config.logger_config import logger


def _clean_think_tags(text: str) -> str:
    """清理模型输出的思考标签，兼容MiniMax等模型"""
    return re.sub(r'<think.*?>.*?</think\s*>', '', text, flags=re.DOTALL)


def _parse_json_from_llm(raw_text: str, key: str) -> List[str]:
    """
    从LLM输出中提取JSON列表字段，兼容思考标签
    先清理思考标签，再用正则提取JSON
    """
    cleaned = _clean_think_tags(raw_text)
    json_match = re.search(r'\{[^{}]*\}', cleaned)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if key in result and isinstance(result[key], list):
                return result[key]
        except json.JSONDecodeError:
            pass
    return []

class QueryExpansion:
    """
    查询扩展:查询扩展(Query Expansion)是 RAG(检索增强生成)系统中提升召回率和召回精度的关键优化手段
    """
    
    def __init__(self, llm):
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_template("""你是一个查询优化助手。
            根据用户的问题,生成3-5个不同的表述方式,这些表述应该:
            1. 保持原意
            2. 使用不同的词汇
            3. 改变句式结构

            用户问题：{question}

            请以JSON格式返回,格式如下:
            {{"queries": ["query1", "query2", "query3"]}}"""
        )
        
        self.chain = self.prompt | llm

    def expand(self, query: str) -> List[str]:
        """扩展查询"""
        try:
            response = self.chain.invoke({"question": query})
            raw_text = response.content if hasattr(response, 'content') else str(response)
            result = _parse_json_from_llm(raw_text, "queries")
            return result if result else [query]
        except Exception as e:
            logger.error(f"查询扩展错误: {e}")
            return [query]

class HyDE:
    """
    假设性文档嵌入:HyDE(Hypothesis-based Document Embedding)是一种基于假设性文档的嵌入方法,用于将查询转换为向量表示,
    """
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
            
        self.prompt = ChatPromptTemplate.from_template("""你是一个问答系统的助手。
            根据用户的问题，生成一个假设性的回答。这个回答应该:
            1. 直接回答问题
            2. 看起来像是基于真实文档的回答
            3. 包含可能的细节和具体信息

            用户问题：{question}

            假设性回答:"""
        )
            
        self.chain = self.prompt | llm

    def generate_hypothetical_doc(self, query: str) -> str:
        """生成假设性文档"""
        result = self.chain.invoke({"question": query})
        return result.content if hasattr(result, 'content') else str(result)
    
    def embed_hypothetical(self, query: str) -> List[float]:
        """嵌入假设性文档"""
        try:
            hypo_doc = self.generate_hypothetical_doc(query)
            return self.embeddings.embed_query(hypo_doc)
        except Exception as e:
            logger.error(f"假设性文档嵌入错误: {e}")
            return []

class QueryDecomposer:
    """
    查询分解:查询分解(Query Decomposition)是一种将查询分解为多个子查询的技术,用于提高召回率和召回精度
    """

    def __init__(self, llm):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template("""你是一个问题分析助手。
            将复杂问题分解为多个简单的子问题。每个子问题应该：
            1. 可以独立回答
            2. 按逻辑顺序排列
            3. 覆盖原问题的各个方面

            用户问题：{question}

            请以JSON格式返回,格式如下:
            {{"sub_questions": ["sub1", "sub2", "sub3"]}}"""
        )

        self.chain = self.prompt | llm

    def decompose(self, query: str) -> List[str]:
        """分解查询"""
        try:
            response = self.chain.invoke({"question": query})
            raw_text = response.content if hasattr(response, 'content') else str(response)
            result = _parse_json_from_llm(raw_text, "sub_questions")
            return result if result else [query]
        except Exception as e:
            logger.error(f"查询分解错误: {e}")
            return [query]
    

class QueryRewriter:
    """
    查询重写器（综合):查询重写器(Query Rewriter)是一种将查询转换为其他表述方式的技术,用于提高召回率和召回精度
    """
    
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        
        self.expansion = QueryExpansion(llm)
        self.hyde = HyDE(llm, embeddings)
        self.decomposer = QueryDecomposer(llm)

    def rewrite(self, query: str, strategy: str = "expansion") -> List[str]:
        """重写查询
        
        Args:
            query: 原始查询
            strategy: 重写策略
                - expansion: 查询扩展
                - hyde: HyDE
                - decomposition: 查询分解
                - all: 所有策略
        
        Returns:
            重写后的查询列表
        """
        if strategy == "expansion":
            return self.expansion.expand(query)
        elif strategy == "hyde":
            hypo_doc = self.hyde.generate_hypothetical_doc(query)
            return [query, hypo_doc]
        elif strategy == "decomposition":
            return self.decomposer.decompose(query)
        elif strategy == "all":
            # 综合多种策略
            queries = set()
            queries.update(self.expansion.expand(query))
            queries.update(self.decomposer.decompose(query))
            return list(queries)
        else:
            return [query]