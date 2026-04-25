"""
全局配置模块
包含环境变量、模型参数、检索配置等
"""
from pydantic_settings import BaseSettings
# pydantic-settings 是 Python 中专门用来做配置管理的库，基于 pydantic 开发，核心作用是：
#   自动从环境变量、.env 文件、配置文件里加载配置
#   对配置做类型校验，避免配置写错导致程序报错
#   让配置管理变得类型安全、清晰易读
from typing import Optional, Dict
from functools import lru_cache
import os
from dotenv import load_dotenv


load_dotenv()


class Settings(BaseSettings):
    """
    全局配置类
    """
    # LLM 配置
    llm_name: str = "Qwen3.5-plus"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    qwen_api_key: str = os.getenv("QWEN_API_KEY", "")
    qwen_base_url: str = os.getenv("QWEN_BASE_URL", "")

    # MiniMax LLM 配置
    minimax_api_key: str = os.getenv("MINIMAX_API_KEY", "")
    minimax_base_url: str = os.getenv("MINIMAX_BASE_URL", "")
    minimax_model: str = os.getenv("MINIMAX_MODEL", "MiniMax-M2.6")

    # ZhipuAI LLM 配置
    zhipuai_api_key: str = os.getenv("ZHIPUAI_API_KEY", "")
    zhipuai_base_url: str = os.getenv("ZHIPUAI_BASE_URL", "")
    zhipuai_model: str = os.getenv("ZHIPUAI_MODEL", "glm-4")

    # 嵌入模型配置
    embedding_model_name: str = "embedding-3"
    embedding_dimension: int = 1024
    zhipu_api_key: str = os.getenv("ZHIPUAI_API_KEY", "")
    zhipu_base_url: str = os.getenv("ZHIPUAI_BASE_URL", "")

    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "agentic_rag"
    milvus_user: str = "root"
    milvus_password: str = ""
    
    # 数据库配置
    database_url: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/postgres")
    
    # 检索配置
    retrieval_top_k: int = 5
    rerank_top_k: int = 3
    similarity_threshold: float = 0.5
    
    # 记忆配置
    short_term_memory_k: int = 10
    long_term_memory_k: int = 3
    
    # 反思配置
    max_reflection_steps: int = 2
    
    # API配置
    api_port: int = 8000
    log_level: str = "INFO"
    api_key: str = os.getenv("API_KEY", "")  # 从环境变量加载 API Key，默认值为空字符串
    
    # 前端API配置(生产环境)
    frontend_api_base: str = os.getenv("FRONTEND_API_BASE", "http://localhost:8000")
    frontend_api_key: str = os.getenv("FRONTEND_API_KEY", "")
    frontend_timeout: int = 60
    
    # 意图识别缓存配置
    intent_cache_enabled: bool = True
    intent_cache_max_size: int = 200
    intent_cache_ttl: int = 1800  # 30分钟
    
    # 生成节点上下文截断配置
    context_truncation_enabled: bool = True
    max_context_tokens: int = 8000  # 最大上下文token数
    max_docs_for_context: int = 5   # 最多使用多少个文档

    # 生成结果缓存配置
    generation_cache_enabled: bool = True
    generation_cache_max_size: int = 100
    generation_cache_ttl: int = 3600  # 1小时

    class Config:
        env_file = ".env"
        case_sensitive = False # 不区分大小写
        extra = "ignore" # 忽略额外的配置项
    
    #  # 告诉它：去读 .env 文件
    # model_config = SettingsConfigDict(env_file=".env")


    def get_model_config(self, model_name: str) -> Dict[str, str]:
        """
        根据模型名称获取对应的 API 配置
        
        参数：
            model_name: 模型名称
        
        返回：
            包含 api_key 和 base_url 的字典
        """
        model_configs = {
            # MiniMax 模型
            "MiniMax-M2.7": {
                "api_key": self.minimax_api_key,
                "base_url": self.minimax_base_url
            },
            # Qwen 模型
            "deepseek-v3.2": {
                "api_key": self.qwen_api_key,
                "base_url": self.qwen_base_url
            },
        }
        
        return model_configs.get(model_name, {
            "api_key": self.qwen_api_key,
            "base_url": self.qwen_base_url
        })
    
    def get_available_models(self) -> list:
        """
        获取所有可用的模型列表
        
        返回：
            可用模型名称列表
        """
        models = []
        if self.minimax_api_key and self.minimax_model:
            models.append(self.minimax_model)
        if self.qwen_api_key:
            qwen_model = os.getenv("QWEN_MODEL", "deepseek-v3.2")
            models.append(qwen_model)
        
        return models if models else ["deepseek-v3.2"]


@lru_cache()  # 缓存装饰器,给函数的返回结果做缓存，避免重复计算，大幅提升性能。
def get_settings() -> Settings:
    """获取配置实例（单例模式）"""
    return Settings()
