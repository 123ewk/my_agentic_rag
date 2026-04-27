"""
API路由定义(生产级)

生产环境优化：
1. 使用依赖注入管理组件生命周期
2. 添加API Key认证保护接口
3. 添加请求限流防止滥用
4. 添加请求追踪中间件
5. 全局异常处理
6. 限制CORS白名单
7. 使用lifespan管理应用生命周期
8. 支持流式响应输出
9. 支持根据模型名称动态选择LLM
"""
from contextlib import asynccontextmanager # 从上下文库导入异步上下文管理器
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Dict, Any, AsyncIterator
import uuid
import os
from datetime import datetime
import time
import json
import asyncio
import threading

# 自定义JSON编码器，处理datetime等特殊类型
class DateTimeEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理datetime等特殊类型"""
    def default(self, obj):
        if isinstance(obj, datetime):
            # 转换为ISO格式，如：2023-12-25T12:00:00.000Z
            return obj.isoformat()
        return super().default(obj)

from .schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    DocumentUpload, UploadResponse, HealthResponse, ErrorResponse
)
from .db_init import init_memory_tables, init_memory_tables_sync

# 配置日志
from loguru import logger

# 全局Agent实例
_agent_instance: Optional[Any] = None
_llm_lock = threading.Lock()  # 用于保护LLM更新的线程锁


def create_agent() -> Any:
    """
    Agent工厂函数:创建并配置AgenticRAGGraph实例
    
    初始化流程：
    1. 加载配置参数
    2. 初始化LLM(ChatOpenAI)
    3. 初始化嵌入模型
    4. 初始化向量存储
    5. 初始化重排模型
    6. 加载工具
    7. 加载记忆模块（短期记忆、长期记忆）
    8. 构建Agent图
    
    返回：
        配置好的AgenticRAGGraph实例
    """
    from langchain_openai import ChatOpenAI
    from agentic_rag.agent.graph import AgenticRAGGraph
    from agentic_rag.config.settings import get_settings
    from agentic_rag.vectorstore.embeddings import get_embeddings
    from agentic_rag.vectorstore.milvus_client import get_vectorstore
    from agentic_rag.retrieval.reranker import get_reranker
    from agentic_rag.tools.search import get_search_tools
    from agentic_rag.memory.short_term import ShortTermMemory
    from agentic_rag.memory.long_term import LongTermMemory

    settings = get_settings()
    
    llm = ChatOpenAI(
        model=settings.llm_name,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url if settings.qwen_base_url else None
    )
    
    embeddings = get_embeddings(
        embedding_type="zhipu",
        api_key=settings.zhipu_api_key,
        model_name=settings.embedding_model_name,
        dimension=settings.embedding_dimension
    )
    
    # 使用单例模式获取向量存储，避免频繁创建gRPC连接
    from agentic_rag.vectorstore.milvus_client import get_vectorstore
    vectorstore = get_vectorstore(
        collection_name=settings.milvus_collection,
        embedding_model=embeddings,
    )
    
    if vectorstore.load_or_initialize():
        logger.info(f"已加载已存在的向量库: {settings.milvus_collection}")
    else:
        logger.warning(f"向量库collection '{settings.milvus_collection}' 不存在，请先使用 /api/v1/upload 接口上传文档初始化向量库")
    
    reranker = get_reranker(
        api_key=settings.zhipu_api_key,
        base_url=settings.zhipu_base_url
    )
    
    tools = get_search_tools()
    
    # 初始化数据库表
    if settings.database_url:
        try:
            init_memory_tables_sync(settings.database_url)
        except Exception as e:
            logger.warning(f"数据库表初始化失败: {e}")
    
    # 初始化短期记忆
    short_term_memory = None
    if settings.database_url:
        try:
            short_term_memory = ShortTermMemory(
                database_url=settings.database_url,
                max_messages=20,
                max_tokens=5000,
                ttl_hours=24
            )
            import asyncio
            try:
                # 作用是获取当前正在运行的事件循环（event loop）对象。只能在已有事件循环正在运行的异步环境里调用
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def init_short_term():
                await short_term_memory.connect()
            
            if loop.is_running(): # 说明当前已经有一个事件循环正在跑了
                import nest_asyncio # 作用是允许在已经运行的事件循环里，再嵌套运行新的协程（也就是 “嵌套异步”）。
                nest_asyncio.apply()
                loop.run_until_complete(init_short_term())
            else:
                loop.run_until_complete(init_short_term())
            
            logger.info("短期记忆模块初始化成功")
        except Exception as e:
            logger.warning(f"短期记忆初始化失败: {e}")
            short_term_memory = None
    
    # 初始化长期记忆
    long_term_memory = None
    if settings.database_url and embeddings:
        try:
            long_term_memory = LongTermMemory(
                embeddings=embeddings,
                database_url=settings.database_url,
                k=5,
                similarity_threshold=0.7
            )
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def init_long_term():
                await long_term_memory.connect()
            
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                loop.run_until_complete(init_long_term())
            else:
                loop.run_until_complete(init_long_term())
            
            logger.info("长期记忆模块初始化成功")
        except Exception as e:
            logger.warning(f"长期记忆初始化失败: {e}")
            long_term_memory = None
    
    prompt_template = """
        你是一个专业的问答助手，需要根据检索到的上下文信息回答用户问题。

        上下文信息：
        {context}

        用户问题：{question}

        请基于上述上下文信息，用简洁专业的语言生成回答。如果有对话历史，请参考历史对话上下文。
"""
    
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
    
    logger.info("Agent实例创建成功")
    return agent


def get_llm_for_model(model_name: Optional[str] = None):
    """
    根据模型名称获取对应的 LLM 实例
    
    参数：
        model_name: 模型名称，如果为 None 则使用默认配置
    
    返回：
        ChatOpenAI LLM 实例
    """
    from langchain_openai import ChatOpenAI
    from agentic_rag.config.settings import get_settings
    
    settings = get_settings()
    
    # 如果没有指定模型，使用默认配置
    if not model_name:
        return ChatOpenAI(
            model=settings.llm_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url if settings.qwen_base_url else None
        )
    
    # 根据模型名称获取对应的 API 配置
    model_config = settings.get_model_config(model_name)
    
    logger.info(f"为模型 '{model_name}' 选择 API 配置")
    
    base_url = model_config.get("base_url")
    api_key = model_config.get("api_key", "")
    
    # 为 MiniMax 模型构建请求参数，禁用思考能力以避免 JSON 解析问题
    if "minimax" in model_name.lower() or "MiniMax" in model_name:
        logger.info(f"为 MiniMax 模型禁用思考能力 (reasoning_level=none)")
    
    return ChatOpenAI(
        model=model_name,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=api_key,
        base_url=base_url if base_url else None
    )


@asynccontextmanager  # 异步上下文管理器
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    功能：
    - 启动时:初始化Agent实例并存储到app.state
    - 关闭时:清理资源
    
    参数：
        app: FastAPI应用实例
    """
    global _agent_instance
    
    # 记录启动时间（用于计算运行时长）
    app.state.start_time = time.time()
    
    # 启动时初始化
    logger.info("正在初始化Agent实例...")
    try:
        _agent_instance = create_agent()
        app.state.agent = _agent_instance
        logger.info("Agent实例初始化完成")
    except Exception as e:
        logger.error("Agent实例初始化失败: {}", str(e), exc_info=True)
        app.state.agent = None
    
    yield  # 应用运行中  
    # 效果：函数里的 yield 会自动分成两段：
    # yield 之前：应用启动时执行
    # yield 之后：应用关闭时执行
    
    # 关闭时清理
    logger.info("正在清理资源...")
    _agent_instance = None
    logger.info("资源清理完成")


# 创建应用
app = FastAPI(
    title="Agentic RAG API",
    description="企业级Agentic RAG智能知识库系统",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan  # 添加生命周期管理
)

# CORS配置（限制白名单域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:8501", "http://127.0.0.1:8501", "http://192.168.1.199:8501"],  # 生产环境必须配置
    allow_credentials=True, # 允许跨域请求携带凭证
    allow_methods=["GET", "POST", "PUT", "DELETE"], 
    allow_headers=["Authorization", "Content-Type", "X-API-Key","X-Request-Id","X-Process-Time"], # 允许的请求头
    max_age=3600  # 预检请求缓存1小时
)


# 依赖注入：获取Agent实例
async def get_agent(request: Request):
    """
    获取Agent实例:依赖注入
    
    从应用状态中获取Agent实例,用于处理RAG查询请求
    
    参数：
        request: FastAPI请求对象
    
    返回：
        AgenticRAGGraph实例
    
    异常：
        HTTPException: Agent未初始化时抛出503错误
    """
    agent = getattr(request.app.state, 'agent', None) # getattr(object, attribute_name, default_value) 是 Python 的内置函数，全称是 get attribute，作用是安全地获取对象的属性。
    
    if agent is None:
        logger.error("Agent实例未初始化")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent未初始化,请检查系统配置"
        )
    
    return agent

# 依赖注入:获取向量数据库
async def get_vectorstore(request: Request):
    """
    获取向量数据库:依赖注入
    
    从应用状态中获取向量数据库实例,用于检索上下文信息
    
    参数：
        request: FastAPI请求对象
    
    返回：
        VectorStore实例
    
    异常：
        HTTPException: Agent未初始化时抛出503错误
        HTTPException: 向量数据库未初始化时抛出503错误
    """
    agent = getattr(request.app.state, 'agent', None)
    
    if agent is None:
        logger.error("Agent实例未初始化")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent未初始化,请检查系统配置"
        )
    
    vectorstore = getattr(agent, 'vectorstore', None)
    
    if vectorstore is None:
        logger.error("向量数据库实例未初始化")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="向量数据库未初始化,请检查系统配置"
        )
    
    return vectorstore



# 依赖注入：验证API Key
async def verify_api_key(request: Request):
    """
    验证API Key
    
    从请求头获取X-API-Key并验证
    """
    from agentic_rag.config.settings import get_settings
    settings = get_settings()
    
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少API Key"
        )
    
    # 实际应从数据库或配置验证
    valid_keys = settings.api_key  # 生产环境应使用环境变量
    if api_key != valid_keys and valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无效的API Key"
        )
    
    return api_key


# 请求限流依赖/中间件
class RateLimiter:
    """请求限流器: (简单滑动窗口实现)"""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {client_ip: [timestamp1, timestamp2, ...]}
    
    async def __call__(self, request: Request):
        client_ip = request.client.host
        now = time.time()
        
        # 清理过期请求
        if client_ip in self.requests:
            self.requests[client_ip] = [
                t for t in self.requests[client_ip]
                if now - t < self.window_seconds
            ]
        else:
            self.requests[client_ip] = []
        
        # 检查限流
        if len(self.requests[client_ip]) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求过于频繁，限制{self.max_requests}次/{self.window_seconds}秒"
            )
        
        # 记录请求
        self.requests[client_ip].append(now)

# 创建请求限流器实例
limiter = RateLimiter(max_requests=60, window_seconds=60)


# 请求追踪中间件,给每个请求生成唯一追踪 ID，并记录接口处理耗时，最后把信息放到响应头里返回给客户端。
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    添加请求追踪ID和处理时间
    
    参数：
        request: FastAPI请求对象
    
        call_next: 下一个中间件或路由处理
    
    返回：
        FastAPI响应对象
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request) # 调用后续的路由处理逻辑
    process_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# 请求限流中间件
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    try:
        await limiter(request)  # 直接调用限流
        response = await call_next(request)
        return response
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail}
        )


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常捕获和日志记录"""
    import traceback
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    # 记录详细错误日志
    error_type = type(exc).__name__
    error_message = str(exc)
    
    # 检查 exc 是否是字符串（异常情况）
    if isinstance(exc, str):
        logger.error(
            "Request {} failed with STRING instead of Exception: {}",
            request_id, repr(exc),
            exc_info=False
        )
        logger.error("异常源追踪 - 字符串内容: {}", exc)
        logger.error(f"请求路径: {request.url.path}")
        logger.error(f"请求方法: {request.method}")
        error_response_message = "服务器内部错误，请稍后重试"
    else:
        # 记录完整堆栈跟踪
        stack_trace = traceback.format_exc()
        logger.error(
            "Request {} failed: {}: {}",
            request_id, error_type, error_message,
            exc_info=True
        )
        logger.error("完整堆栈跟踪:\n{}", stack_trace)
        logger.error(f"请求路径: {request.url.path}")
        logger.error(f"请求方法: {request.method}")
        # 针对KeyError做特殊处理，提取缺失的键名
        if isinstance(exc, KeyError):
            key_name = exc.args[0] if exc.args else '未知'
            # 确保key_name是字符串
            if isinstance(key_name, str):
                error_response_message = f"缺少必需的数据字段: {key_name}"
            else:
                error_response_message = f"缺少必需的数据字段: {str(key_name)}"
        else:
            error_response_message = "服务器内部错误，请稍后重试"
    
    # 返回友好错误响应
    try:
        error_response = ErrorResponse(
            error=error_type,
            message=error_response_message,
            request_id=request_id
        )
        response_content = error_response.model_dump()
    except Exception:
        # 如果ErrorResponse创建失败，返回最基础的错误格式
        response_content = {
            "error": error_type,
            "message": error_response_message,
            "request_id": request_id
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_content
    )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    summary="RAG查询接口",
    description="基于知识库回答用户问题，支持多轮对话和工具调用"
)
async def query(
    request: QueryRequest,
    http_request: Request,
    agent = Depends(get_agent),
    api_key: str = Depends(verify_api_key)
):
    """RAG查询接口: (生产级)"""
    request_id = http_request.state.request_id
    # 检查Agent是否初始化
    
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent未初始化,请检查系统配置"
        )
    
    try:
        start_time = time.time()
        
        # 生成session_id
        session_id = request.session_id or str(uuid.uuid4())
        
        # 如果指定了模型名称，动态更新 agent 的 LLM
        if request.model_name:
            llm = get_llm_for_model(request.model_name)
            agent.llm = llm
            logger.info(f"[{request_id}] 已切换到模型: {request.model_name}")
        
        # 记录调用参数
        logger.info(f"[{request_id}] 开始处理查询: question='{request.question[:50]}...', session_id={session_id}")
        logger.debug(f"[{request_id}] 调用参数: use_tools={request.use_tools}, max_reflection={request.max_reflection}, temperature={request.temperature}")
        
        # 执行查询
        try:
            result = agent.invoke(
                question=request.question,
                session_id=session_id,
                user_id=request.user_id,
                use_tools=request.use_tools,
                max_reflection_steps=request.max_reflection,
                temperature=request.temperature
            )
            logger.info(f"[{request_id}] Agent调用成功")
        except Exception as e:
            logger.error("[{}] Agent.invoke() 抛出异常: {}: {}", request_id, type(e).__name__, str(e), exc_info=True)
            raise
        
        processing_time = time.time() - start_time
        
        # 构建响应 - 防御性处理
        tool_results = result.get("tool_results", {})
        if not isinstance(tool_results, dict):
            tool_results = {}
        
        sources = [
            SourceDocument(
                content=doc.page_content[:500],
                metadata=doc.metadata,
                score=doc.metadata.get("score"),
                source=doc.metadata.get("source")
            )
            for doc in result.get("reranked_docs", [])
        ]
        
        answer = result.get("refined_answer") or result.get("generation", "")
        if not answer:
            answer = "抱歉，当前无法生成回答，请检查系统状态或稍后重试。"
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            metrics=result.get("evaluation", {}),
            session_id=session_id,
            intent=result.get("intent", "unknown"),
            tools_used=list(tool_results.keys()),
            reflection_count=result.get("reflection_count", 0),
            processing_time=processing_time
        )
    
    except ValueError as e:
        # 用户输入错误
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # 服务器错误
        logger.error("Query failed for request {}: {}", request_id, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="查询处理失败，请稍后重试"
        )


async def generate_stream_response(
    agent, 
    question: str, 
    session_id: str, 
    user_id: Optional[str] = None,
    use_tools: bool = False,
    temperature: float = 0.7,
    max_reflection: int = 2,
    model_name: Optional[str] = None,
    use_fast_path: bool = False
) -> AsyncIterator[bytes]:
    """
    流式响应生成器
    
    参数：
        agent: Agent实例
        question: 用户问题
        session_id: 会话ID
        user_id: 用户ID
        use_tools: 是否使用工具
        temperature: 温度参数
        max_reflection: 最大反思次数
        model_name: 模型名称(用于动态选择LLM)
        use_fast_path: 快速路径模式(缓存命中时跳过评估)
    
    产出：
        bytes: SSE格式的响应数据:SSE 是一种让服务器能主动向浏览器 / 客户端持续推送数据的技术
    """
    try:
        # 如果指定了模型名称，动态更新 agent 的 LLM
        if model_name:
            llm = get_llm_for_model(model_name)
            agent.llm = llm
            logger.info(f"已切换到模型: {model_name}")
        
        # 调用流式invoke方法
        async for event in agent.stream_invoke(
            question=question,
            session_id=session_id,
            user_id=user_id,
            use_tools=use_tools,
            temperature=temperature,
            max_reflection_steps=max_reflection,
            use_fast_path=use_fast_path
        ):
            # 将事件转换为SSE格式
            event_type = event.get("type", "unknown")
            event_content = event.get("content", "")
            event_data = event.get("data", {})
            
            # 构建SSE数据
            sse_data = {
                "type": event_type,
                "content": event_content,
                **event_data
            }
            
            # 使用data:前缀，这是SSE的标准格式，使用自定义编码器处理datetime
            yield f"data: {json.dumps(sse_data, ensure_ascii=False, cls=DateTimeEncoder)}\n\n".encode('utf-8')
            
            # 如果是完成事件，发送一个特殊的标记
            if event_type == "done":
                yield b"event: done\ndata: [DONE]\n\n"
        
    except Exception as e:
        logger.error("Stream generation failed: {}", str(e), exc_info=True)
        error_data = {
            "type": "error",
            "content": "生成失败",
            "error": str(e)
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False, cls=DateTimeEncoder)}\n\n".encode('utf-8')


@app.post(
    "/api/v1/query/stream",
    summary="RAG流式查询接口",
    description="基于知识库回答用户问题，支持流式输出和实时状态更新"
)
async def query_stream(
    request: QueryRequest,
    http_request: Request,
    agent = Depends(get_agent),
    api_key: str = Depends(verify_api_key),
    #limiter: RateLimiter = Depends(lambda: limiter) #实例依赖,要用lambda
):
    """
    RAG流式查询接口: (生产级)
    
    支持SSE(Server-Sent Events)流式响应,
    实时返回处理状态和生成内容
    """
    request_id = http_request.state.request_id
    
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent未初始化,请检查系统配置"
        )
    
    try:
        # 生成session_id
        session_id = request.session_id or str(uuid.uuid4())
        
        # 创建流式响应
        return StreamingResponse(
            generate_stream_response(
                agent=agent,
                question=request.question,
                session_id=session_id,
                user_id=request.user_id,
                use_tools=request.use_tools,
                temperature=request.temperature,
                max_reflection=request.max_reflection,
                model_name=request.model_name,
                use_fast_path=request.use_fast_path
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
                "Access-Control-Allow-Origin": "*",
            }
        )
    
    except Exception as e:
        logger.error("Stream query failed for request {}: {}", request_id, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="流式查询失败，请稍后重试"
        )


@app.post(
    "/api/v1/upload",
    response_model=UploadResponse,
    summary="文档上传接口",
    description="上传文档并自动索引到知识库"
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="上传文件"),
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    vectorstore = Depends(get_vectorstore),  
    api_key: str = Depends(verify_api_key)
):
    """
    文档上传接口: (生产级)
    
    支持文件类型: PDF, Word(.docx), TXT, Markdown, CSV, Excel(.xlsx/.xls), HTML
    文件大小限制: 50MB
    """
    if vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="向量存储未初始化,请检查系统配置"
        )
    
    try:
        all_chunks = []
        
        # 允许的MIME类型与文件后缀的映射,用于兼容浏览器可能发送的各种content_type
        allowed_types = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "text/csv": ".csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.ms-excel": ".xls",
            "text/html": ".html",
            "application/octet-stream": None,  # 浏览器对未知类型可能发送此值,需通过文件名判断
        }
        
        # 通过文件后缀名判断是否为支持的类型(比MIME类型更可靠)
        supported_extensions = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".xls", ".xlsm", ".html", ".htm"}
        
        for file in files:
            # 优先通过文件名后缀判断类型,因为浏览器发送的content_type不一定准确
            from pathlib import Path as FilePath
            file_ext = FilePath(file.filename).suffix.lower() if file.filename else ""
            
            if file_ext not in supported_extensions:
                # 如果后缀不在支持列表,再检查content_type
                if file.content_type not in allowed_types:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"不支持的文件类型: {file.filename or file.content_type}, 支持的格式: {', '.join(sorted(supported_extensions))}"
                    )
                # content_type允许但后缀未知时,从映射中获取后缀
                mapped_ext = allowed_types.get(file.content_type)
                if mapped_ext:
                    file_ext = mapped_ext
            
            # 验证文件大小（限制50MB）
            if file.size and file.size > 50 * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="文件大小超过50MB限制"
                )
            
            # 处理文档
            from agentic_rag.document_processing.loaders import DocumentLoader
            from agentic_rag.document_processing.splitters import get_splitter
            
            loader = DocumentLoader()
            splitter = get_splitter("semantic", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # 读取文件内容
            content = await file.read()
            
            # 保存到临时文件,使用正确的文件后缀确保DocumentLoader能识别文件类型
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                # 加载并分割
                documents = loader.load(tmp_path)
                chunks = splitter.split_documents(documents)
                
                # 过滤空chunk,防止嵌入API因空文本报错
                valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
                if len(valid_chunks) < len(chunks):
                    logger.warning(f"文件 {file.filename}: 跳过 {len(chunks) - len(valid_chunks)} 个空分块")
                
                all_chunks.extend(valid_chunks)
            finally:
                # 确保临时文件被删除
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        if not all_chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="上传的文件未能提取到有效文本内容"
            )
        
        # 批量添加到向量存储
        vectorstore.add_documents(all_chunks)
        
        return UploadResponse(
            success=True,
            documents_processed=len(files),
            chunks_created=len(all_chunks),
            collection_info={},
            message=f"成功处理 {len(files)} 个文件，创建 {len(all_chunks)} 个分块"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload failed: {}", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档上传失败: {str(e)}"
        )


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查系统各组件健康状态"
)
async def health_check(request: Request):
    """
    健康检查（生产级）
    
    检查各组件状态：
    - database: PostgreSQL数据库连接
    - vectorstore: Milvus向量存储
    - agent: Agent实例是否初始化
    """
    components = {}
    
    # 检查数据库连接
    try:
        from agentic_rag.memory.short_term import ShortTermMemory
        from agentic_rag.config.settings import get_settings
        settings = get_settings()
        memory = ShortTermMemory(database_url=settings.database_url)
        await memory.connect()
        await memory.close()
        components["database"] = "healthy"
    except Exception as e:
        logger.warning("数据库健康检查失败: {}", str(e))
        components["database"] = "unhealthy"
    
    # 检查向量存储
    try:
        from agentic_rag.vectorstore.milvus_client import MilvusClient
        from pymilvus import connections
        from agentic_rag.config.settings import get_settings
        settings = get_settings()
        connections.connect(
            host=settings.milvus_host,
            port=settings.milvus_port,
            user=settings.milvus_user or "root",
            password=settings.milvus_password or ""
        )
        connections.disconnect("default")
        components["vectorstore"] = "healthy"
    except Exception as e:
        logger.warning("向量存储健康检查失败: {}", str(e))
        components["vectorstore"] = "unhealthy"
    
    # 检查Agent实例
    agent = getattr(request.app.state, 'agent', None)
    components["agent"] = "healthy" if agent is not None else "unhealthy"
    
    # 确定总体状态
    healthy_count = sum(1 for v in components.values() if v == "healthy")
    total_count = len(components)
    
    if healthy_count == total_count:
        status = "healthy"
    elif healthy_count >= total_count // 2:
        status = "degraded"  # 至少一半组件健康，降级运行
    else:
        status = "unhealthy"
    
    # 计算运行时长
    start_time = getattr(request.app.state, 'start_time', time.time())
    uptime_seconds = time.time() - start_time
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        components=components,
        uptime_seconds=uptime_seconds
    )


def create_app() -> FastAPI:
    """
    创建应用实例: (工厂模式)
    
    返回：
        配置好的FastAPI应用
    """
    return app