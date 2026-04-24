from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class BaseRAG:
    """基础RAG类"""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: ChatOpenAI,
        prompt_template: str = None,
    ) -> None:

        self.retriever = retriever
        self.llm = llm

        if prompt_template is None:
            prompt_template = """
            你是一个专业的问答机器人,你的任务是根据用户的问题和上下文,使用简洁风趣的语言生成符合要求的回答。
            上下文：{context}
            问题：{question}
            回答：
            """
        
        # 初始化提示模板,它是 LangChain 里用来生成给大模型的输入文本的工具，相当于一个 “填空模板”。
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        # 初始化输出解析器,把大模型的输出 “标准化成字符串”
        self.output_parser = StrOutputParser()

        # 构建链
        self.chain = (
            {"context": self._format_context, "question": lambda x: x["question"]}
            | self.prompt
            | self.llm
            | self.output_parser
        )

    def _format_context(self, inputs: Dict[str, Any]) -> str:
        """格式化上下文"""
        docs = inputs.get("context", [])
        if isinstance(docs, list) and len(docs) > 0:
            if isinstance(docs[0], Document):
                return "\n\n".join([doc.page_content for doc in docs])
            return "\n\n".join(str(doc) for doc in docs)
        return str(docs)
    
    def invoke(self, question: str) -> Dict[str, Any]:
        """执行RAG"""
        # 检索相关文档
        retrieved_docs = self.retriever.invoke(question)
        
        # 生成答案
        answer = self.chain.invoke({
            "question": question,
            "context": retrieved_docs
        })
        
        return {
            "question": question,
            "answer": answer,
            "source_documents": retrieved_docs
        }
    
    def batch_invoke(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量处理"""
        return [self.invoke(q) for q in questions]

        