"""
Agentic RAG主程序
"""
from pathlib import Path
from dotenv import load_dotenv
import asyncio

from agentic_rag.config.settings import get_settings
from agentic_rag.config.logger_config import setup_logging
from agentic_rag.vectorstore.milvus_client import get_vectorstore
from agentic_rag.vectorstore.embeddings import get_embeddings
from agentic_rag.retrieval.reranker import get_reranker
from agentic_rag.agent.graph import AgenticRAGGraph
from agentic_rag.tools.search import duckduckgo_search, calculator
from agentic_rag.evaluation.metrics import evaluate_response
from agentic_rag.api.routes import create_app
from agentic_rag.api.db_init import init_memory_tables_sync
from agentic_rag.memory.short_term import ShortTermMemory
from agentic_rag.memory.long_term import LongTermMemory
from agentic_rag.schedulers.short_scheduler import get_scheduler as get_short_scheduler
from agentic_rag.schedulers.long_scheduler import get_scheduler as get_long_scheduler

# 加载环境变量
load_dotenv()

# 配置日志
logger = setup_logging()

def initialize_components():
    """初始化所有组件"""
    settings = get_settings()
    
    logger.info("初始化组件...")
    
    # 1. 初始化数据库表（如果不存在）
    logger.info("初始化数据库表...")
    init_memory_tables_sync(settings.database_url)
    
    # 2. 初始化嵌入模型,使用配置中的dimension参数
    embeddings = get_embeddings(
        embedding_type="zhipu",
        api_key=settings.zhipu_api_key,
        model_name=settings.embedding_model_name,
        dimension=settings.embedding_dimension  # 从配置传递dimension: 1024
    )
    logger.info(f"嵌入模型: {settings.embedding_model_name}")
    
    # 3. 初始化向量存储(传递embeddings实例而非字符串,确保dimension等参数一致)
    # 使用单例模式，避免频繁创建gRPC连接
    vectorstore = get_vectorstore(
        collection_name=settings.milvus_collection,
        embedding_model=embeddings
    )
    
    # 尝试加载已存在的向量库collection
    if vectorstore.load_or_initialize():
        logger.info(f"已加载已存在的向量库: {settings.milvus_collection}")
    else:
        logger.warning(f"向量库collection '{settings.milvus_collection}' 不存在，请先使用 /api/v1/upload 接口上传文档初始化向量库")
    logger.info("向量存储初始化完成")
    
    # 4. 初始化重排模型
    reranker = get_reranker(
        reranker_type="bge",
        api_key=settings.zhipu_api_key,
        base_url=settings.zhipu_base_url,
        )
    logger.info("重排模型初始化完成")
    
    # 5. 初始化LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=settings.llm_name,
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens
    )
    logger.info(f"LLM: {settings.llm_name}")
    
    # 6. 初始化短期记忆管理器
    logger.info("初始化短期记忆管理器...")
    short_term_memory = ShortTermMemory(
        database_url=settings.database_url,
        max_messages=settings.short_term_memory_k,
        max_tokens=5000,
        ttl_hours=24
    )
    # 连接数据库
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(short_term_memory.connect())
    logger.info("短期记忆管理器初始化完成")
    
    # 7. 初始化长期记忆管理器
    logger.info("初始化长期记忆管理器...")
    long_term_memory = LongTermMemory(
        embeddings=embeddings,
        database_url=settings.database_url,
        k=settings.long_term_memory_k,
        similarity_threshold=settings.similarity_threshold
    )
    # 连接数据库
    loop.run_until_complete(long_term_memory.connect())
    logger.info("长期记忆管理器初始化完成")
    
    # 8. 定义提示词模板
    prompt_template = """你是一个专业的AI助手。请基于提供的上下文信息回答用户的问题。

        上下文信息：
        {context}

        用户问题：{question}

        请提供准确、详细的回答。如果上下文中没有相关信息，请明确告知用户。
    """
    
    # 9. 初始化工具
    tools = {
        "duckduckgo": duckduckgo_search,
        "calculator": calculator,
    }
    logger.info(f"已加载 {len(tools)} 个工具")
    
    # 10. 初始化Agent（传入记忆管理器）
    agent = AgenticRAGGraph(
        llm=llm,
        embeddings=embeddings,
        vectorstore=vectorstore,
        reranker=reranker,
        tools=tools,
        prompt_template=prompt_template,
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory
    )
    logger.info("Agent初始化完成")
    
    return {
        "agent": agent,
        "vectorstore": vectorstore,
        "llm": llm,
        "embeddings": embeddings,
        "settings": settings,
        "short_term_memory": short_term_memory,
        "long_term_memory": long_term_memory
    }

def setup_short_term_scheduler(short_term_memory):
    """
    配置短期记忆定时任务调度器
    
    Args:
        short_term_memory: ShortTermMemory实例
    """
    scheduler = get_short_scheduler()
    scheduler.add_cleanup_expired_task(short_term_memory)
    scheduler.start()
    logger.info("短期记忆定时任务调度器配置完成")
    return scheduler


def setup_long_term_scheduler(long_term_memory, user_ids: list = None, retention_days: int = 90):
    """
    配置长期记忆定时任务调度器
    
    Args:
        long_term_memory: LongTermMemory实例
        user_ids: 需要清理的用户ID列表（None表示从数据库动态获取）
        retention_days: 记忆保留天数，默认90天
    """
    scheduler = get_long_scheduler()
    scheduler.add_cleanup_old_memories_task(long_term_memory, user_ids, retention_days)
    scheduler.start()
    logger.info("长期记忆定时任务调度器配置完成")
    return scheduler


def get_all_user_ids(long_term_memory) -> list:
    """
    从长期记忆数据库中获取所有活跃用户ID
    
    Args:
        long_term_memory: LongTermMemory实例
        
    Returns:
        用户ID列表
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            stats = loop.run_until_complete(long_term_memory.get_stats(""))
            logger.info(f"获取到 {len(stats)} 个用户的长期记忆统计")
            return list(stats.keys())
        finally:
            loop.close()
    except Exception as e:
        logger.warning(f"获取用户列表失败: {e}")
        return []

def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("Agentic RAG 系统启动")
    logger.info("=" * 50)
    
    # 初始化组件
    components = initialize_components()
    
    # 配置短期记忆定时任务调度器
    short_term_scheduler = setup_short_term_scheduler(components["short_term_memory"])
    
    # 配置长期记忆定时任务调度器
    long_term_scheduler = setup_long_term_scheduler(components["long_term_memory"])
    
    # 创建API应用
    app = create_app()
    
    # 启动服务
    import uvicorn
    settings = components["settings"]
    
    logger.info(f"API服务启动: http://127.0.0.1:{settings.api_port}")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )

if __name__ == "__main__":
    main()
