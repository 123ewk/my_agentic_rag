# > **💡 优化说明**：
# > - 使用依赖注入（Dependency Injection）管理组件生命周期
# > - 添加请求限流（Rate Limiting）防止滥用
# > - 添加请求验证和输入清洗
# > - 使用中间件添加请求追踪和错误日志
# > - 添加 API Key 认证保护接口
# > - 移除全局变量，使用应用状态管理
# > - CORS 配置限制白名单域名

"""
API数据模型(生产级)

优化：
1. 添加字段验证和约束
2. 添加自定义验证错误信息
3. 添加示例值方便OpenAPI文档生成
"""
from pydantic import BaseModel, Field, field_validator, field_serializer
from typing import List, Optional, Dict, Any
from datetime import datetime
import re


class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(
        ...,
        description="用户问题",
        min_length=1,
        max_length=2000,
        examples=["什么是人工智能？"]
    )
    session_id: Optional[str] = Field(
        None,
        description="会话ID(用于多轮对话)",
        max_length=128,
        examples=["session-123"]
    )
    user_id: Optional[str] = Field(
        None,
        description="用户ID(用于长期记忆)",
        max_length=128
    )
    model_name: Optional[str] = Field(
        None,
        description="模型名称(用于动态选择LLM)",
        examples=["deepseek-v3.2", "MiniMax-M2.6", "glm-4"]
    )
    use_tools: bool = Field(True, description="是否使用工具")
    use_fast_path: bool = Field(False, description="快速路径模式(缓存命中时跳过评估,响应更快)")
    max_reflection: int = Field(2, ge=0, le=5, description="最大反思次数")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    
    @field_validator('question')
    def validate_question_not_empty(cls, v):
        """验证问题不为空"""
        if not v.strip():
            raise ValueError("问题不能为空")
        return v.strip()
    
    @field_validator('session_id')
    def validate_session_id(cls, v):  # 在 @field_validator 装饰的方法中，cls 是模型类本身，v 就是当前被验证的字段值。
        """验证session_id格式"""
        if v and not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("session_id只能包含字母、数字、下划线和连字符")
        return v


class SourceDocument(BaseModel):
    """来源文档"""
    content: str = Field(..., description="文档内容片段")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    score: Optional[float] = Field(None, description="相关性分数")
    source: Optional[str] = Field(None, description="来源文件名")


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str = Field(..., description="AI回答")
    sources: List[SourceDocument] = Field(default_factory=list, description="来源文档")
    metrics: Dict[str, float] = Field(default_factory=dict, description="评估指标")
    session_id: str = Field(..., description="会话ID")
    intent: str = Field(..., description="识别的意图")
    tools_used: List[str] = Field(default_factory=list, description="使用的工具")
    reflection_count: int = Field(0, description="反思次数")
    processing_time: float = Field(..., description="处理时间（秒）")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="时间戳")
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        """序列化时间戳为ISO格式字符串"""
        if value is None:
            return datetime.now().isoformat()
        return value.isoformat()


class DocumentUpload(BaseModel):
    """文档上传"""
    files: List[str] = Field(..., description="文件路径列表", min_length=1, max_length=50)
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    chunk_size: int = Field(500, ge=100, le=2000, description="分块大小")
    chunk_overlap: int = Field(50, ge=0, le=500, description="分块重叠")


class UploadResponse(BaseModel):
    """上传响应"""
    success: bool = Field(..., description="是否成功")
    documents_processed: int = Field(0, description="处理文档数")
    chunks_created: int = Field(0, description="创建分块数")
    collection_info: Dict[str, str] = Field(default_factory=dict, description="集合信息")
    message: str = Field(..., description="提示信息")

    @field_validator('collection_info', mode='before')
    @classmethod
    def validate_collection_info(cls, v):
        """确保 collection_info 只包含可序列化的值"""
        if not isinstance(v, dict):
            return {}
        result = {}
        for key, value in v.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                result[str(key)] = value
            else:
                result[str(key)] = str(value)
        return result


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="健康状态")
    version: str = Field(..., description="版本号")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="时间戳")
    components: Dict[str, str] = Field(default_factory=dict, description="组件状态")
    uptime_seconds: float = Field(0, description="运行时长")
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        """序列化时间戳为ISO格式字符串"""
        if value is None:
            return datetime.now().isoformat()
        return value.isoformat()


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误信息")
    request_id: str = Field(..., description="请求追踪ID")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="时间戳")
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        """序列化时间戳为ISO格式字符串"""
        if value is None:
            return datetime.now().isoformat()
        return value.isoformat()