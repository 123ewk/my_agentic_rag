"""
文档分块策略
支持固定大小、语义、递归等多种分块方式
"""

from typing import List, Callable
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

import spacy # 导入spacy库(NLP库)，用于分块



class TextSplitter:
    """文本分块器基类"""
    def __init__(self):
        # zh_core_web_sm 是一个完整的中文 NLP 模型，包含分词、词性标注、NER 等功能，而你只用到了其中的分句功能。
        # 加载完整模型比只加载 sentencizer 分句器要多占用几倍的内存，完全没必要。

        # 加载一个空的中文模型，只添加分句器，无外部模型依赖,
        self.nlp = spacy.blank("zh")  # spacy,load(zh_core_web_sm),必须安装模型文件，否则报错, 分词、分句、NER、句法分析全都有
        self.nlp.add_pipe("sentencizer")  # 开启分句功能



    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        raise NotImplementedError

class RecursiveChunker(TextSplitter):
    """递归字符分块器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = ["\n\n", "\n", "。", "!", "?", " ", ""]
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len # 使用字符长度作为分块标准
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        return self.splitter.split_documents(documents)

class SemanticChunker(TextSplitter):
    """语义分块器（基于句子边界）"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        split_by: str = "paragraph_and_sentence"  # sentence, paragraph, paragraph_and_sentence
    ):
        super().__init__()  # 调用父类构造函数，初始化 self.nlp
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_by = split_by
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """语义分割文档"""
        split_docs = []
        
        for doc in documents:
            if self.split_by == "sentence":
                chunks = self._split_by_sentence(doc)
            elif self.split_by == "paragraph":
                chunks = self._split_by_paragraph(doc)
            else:
                chunks = self._split_by_paragraph_and_sentence(doc)
            
            for i, chunk_text in enumerate(chunks):
                split_docs.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                ))
        
        return split_docs
    
    def _split_by_sentence(self, doc: Document) -> List[str]:
        """按句子分割"""
        text = doc.page_content
        sentences = []

        # 核心分句
        doc_spacy = self.nlp(text)
        # 遍历所有句子
        for sent in doc_spacy.sents:
            sent_str = str(sent.text).strip()

            if sent_str :
                sentences.append(sent_str)

        # 返回合并的句子块
        return self._merge_chunks(sentences)
            
    def _split_by_paragraph(self, doc: Document) -> List[str]:
        """按段落分割"""

        text = doc.page_content

        # 1. 按换行符分割（支持 \n 和 \r\n）
        lines = text.splitlines()

        paragraphs = []
        current_paragraph = ""

        for line in lines:
            line = line.strip()
            if line:
                current_paragraph += " " + line
            else:
                if current_paragraph.strip():
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = ""

        # 处理最后一个段落
        if current_paragraph.strip():
            paragraphs.append(current_paragraph.strip())
        
        # 返回合并的段落块
        return self._merge_chunks(paragraphs)
        
    def _split_by_paragraph_and_sentence(self, doc: Document) -> List[str]:
        """
        先分段 → 再分句 → 再合并（最稳定的文档切割方案）
        """
        text = doc.page_content
        paragraphs = text.split("\n\n")  # 按双换行分段
        chunks = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
        
            # 核心分句,使用Spacy的句子边界检测
            doc_spacy = self.nlp(para)
            # 遍历所有句子
            for sent in doc_spacy.sents:
                sent_str = str(sent.text).strip()

                if sent_str :
                    chunks.append(sent_str)

        return self._merge_chunks(chunks)


    def _merge_chunks(self, segments: List[str]) -> List[str]:
        """合并分段为块"""
        chunks = []
        current_chunk = ""
        
        for segment in segments:
            # 1. 如果当前块 + 新段落 还没超过 chunk_size → 直接拼进去
            if len(current_chunk) + len(segment) <= self.chunk_size:
                current_chunk += segment + "\n"
            # 2. 超过大小了 → 把当前块保存，然后处理重叠
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # 保留重叠部分,支持滑动窗口重叠（overlap），解决 RAG 里的 “上下文断裂” 问题
                overlap_size = 0
                overlap_chunk = ""
                if chunks:
                    # 倒序遍历上一个chunk的句子，收集到overlap_size为止
                    for seg in reversed(chunks[-1].split("\n")):
                        seg = seg.strip()
                        if not seg:
                            continue

                        if overlap_size + len(seg) +1 <= self.chunk_overlap:
                            overlap_chunk = seg + "\n" + overlap_chunk
                            overlap_size += len(seg)
                        else:
                            break
                # 新的当前块 = 重叠部分 + 新段落
                current_chunk = overlap_chunk + segment + "\n"

        # 3. 循环结束，把最后剩下的current_chunk保存
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class AdaptiveChunker(TextSplitter):
    """自适应分块器（根据内容类型动态调整）"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """自适应分割"""
        split_docs = []
        
        for doc in documents:
            doc_type = doc.metadata.get("type", "unknown")
            
            # 根据文档类型选择分块策略
            if doc_type == "pdf":
                # PDF文档使用较大块
                splitter = RecursiveChunker(
                    chunk_size=int(self.chunk_size * 1.5),
                    chunk_overlap=self.chunk_overlap
                )
            elif doc_type == "code":
                # 代码文档按行/函数分割
                splitter = RecursiveChunker(
                    chunk_size=int(self.chunk_size * 0.7),
                    chunk_overlap=self.chunk_overlap,
                    separators=["\ndef ", "\nclass ", "\n\n", "\n", " "]
                )
            else:
                # 默认策略
                splitter = RecursiveChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            
            chunks = splitter.split_documents([doc])
            split_docs.extend(chunks)
        
        return split_docs

def get_splitter(splitter_type: str = "semantic", **kwargs) -> TextSplitter:
    """
    获取分块器实例(默认语义分块器)
    :param splitter_type: 分块器类型，可选值为 "recursive"、"semantic"、"adaptive"
    :param kwargs: 其他分块器的参数
    :return: 分块器实例
    """
    splitters = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "adaptive": AdaptiveChunker
    }
    
    if splitter_type not in splitters:
        raise ValueError(f"Unknown splitter type: {splitter_type}")
    
    return splitters[splitter_type](**kwargs)