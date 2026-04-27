"""
API模块测试
测试API数据模型和路由
"""
import pytest
from pydantic import ValidationError
from datetime import datetime


class TestQueryRequest:
    """查询请求模型测试"""

    def test_valid_request(self):
        """测试有效请求"""
        from agentic_rag.api.schemas import QueryRequest
        
        request = QueryRequest(
            question="什么是Python？",
            session_id="test-session-123",
            user_id="user-123",
            use_tools=True,
            max_reflection=2,
            temperature=0.7
        )
        
        assert request.question == "什么是Python？"
        assert request.session_id == "test-session-123"
        assert request.use_tools is True
        assert request.max_reflection == 2
        assert request.temperature == 0.7

    def test_empty_question_rejected(self):
        """测试空问题被拒绝"""
        from agentic_rag.api.schemas import QueryRequest
        
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="   ")
        
        assert "问题不能为空" in str(exc_info.value)

    def test_question_too_long(self):
        """测试问题过长"""
        from agentic_rag.api.schemas import QueryRequest
        
        with pytest.raises(ValidationError):
            QueryRequest(question="a" * 3000)

    def test_session_id_format(self):
        """测试session_id格式验证"""
        from agentic_rag.api.schemas import QueryRequest
        
        valid_request = QueryRequest(
            question="测试",
            session_id="valid_session-123"
        )
        assert valid_request.session_id == "valid_session-123"

    def test_invalid_session_id(self):
        """测试无效session_id"""
        from agentic_rag.api.schemas import QueryRequest
        
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                question="测试",
                session_id="invalid session!"
            )
        
        assert "session_id只能包含" in str(exc_info.value)

    def test_temperature_range(self):
        """测试温度参数范围"""
        from agentic_rag.api.schemas import QueryRequest
        
        request = QueryRequest(
            question="测试",
            temperature=0.0
        )
        assert request.temperature == 0.0
        
        request = QueryRequest(
            question="测试",
            temperature=2.0
        )
        assert request.temperature == 2.0
        
        with pytest.raises(ValidationError):
            QueryRequest(question="测试", temperature=3.0)

    def test_max_reflection_range(self):
        """测试最大反思次数范围"""
        from agentic_rag.api.schemas import QueryRequest
        
        request = QueryRequest(
            question="测试",
            max_reflection=0
        )
        assert request.max_reflection == 0
        
        request = QueryRequest(
            question="测试",
            max_reflection=5
        )
        assert request.max_reflection == 5
        
        with pytest.raises(ValidationError):
            QueryRequest(question="测试", max_reflection=10)

    def test_default_values(self):
        """测试默认值"""
        from agentic_rag.api.schemas import QueryRequest
        
        request = QueryRequest(question="测试")
        
        assert request.session_id is None
        assert request.user_id is None
        assert request.model_name is None
        assert request.use_tools is True
        assert request.max_reflection == 2
        assert request.temperature == 0.7


class TestQueryResponse:
    """查询响应模型测试"""

    def test_valid_response(self):
        """测试有效响应"""
        from agentic_rag.api.schemas import QueryResponse, SourceDocument
        
        response = QueryResponse(
            answer="Python是一种编程语言",
            sources=[],
            metrics={"faithfulness": 0.9},
            session_id="test-session",
            intent="factual",
            tools_used=["search"],
            reflection_count=1,
            processing_time=1.5
        )
        
        assert response.answer == "Python是一种编程语言"
        assert response.session_id == "test-session"
        assert response.intent == "factual"

    def test_timestamp_serialization(self):
        """测试时间戳序列化"""
        from agentic_rag.api.schemas import QueryResponse
        
        response = QueryResponse(
            answer="测试回答",
            sources=[],
            metrics={},
            session_id="test",
            intent="factual",
            processing_time=1.0
        )
        
        timestamp_str = response.timestamp.isoformat()
        assert "T" in timestamp_str


class TestSourceDocument:
    """来源文档模型测试"""

    def test_valid_document(self):
        """测试有效文档"""
        from agentic_rag.api.schemas import SourceDocument
        
        doc = SourceDocument(
            content="这是一个测试文档的内容",
            metadata={"source": "test.pdf", "page": 1},
            score=0.95,
            source="test.pdf"
        )
        
        assert doc.content == "这是一个测试文档的内容"
        assert doc.metadata["source"] == "test.pdf"
        assert doc.score == 0.95


class TestUploadResponse:
    """上传响应模型测试"""

    def test_valid_response(self):
        """测试有效响应"""
        from agentic_rag.api.schemas import UploadResponse
        
        response = UploadResponse(
            success=True,
            documents_processed=5,
            chunks_created=100,
            collection_info={"count": "1000"},
            message="上传成功"
        )
        
        assert response.success is True
        assert response.documents_processed == 5
        assert response.chunks_created == 100

    def test_collection_info_validation(self):
        """测试集合信息验证"""
        from agentic_rag.api.schemas import UploadResponse
        
        response = UploadResponse(
            success=True,
            documents_processed=1,
            chunks_created=10,
            collection_info={"count": 100, "name": "test"},
            message="成功"
        )
        
        assert response.collection_info["count"] == "100"
        assert response.collection_info["name"] == "test"


class TestHealthResponse:
    """健康检查响应模型测试"""

    def test_valid_health_response(self):
        """测试有效健康检查响应"""
        from agentic_rag.api.schemas import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            components={
                "database": "healthy",
                "vectorstore": "healthy",
                "agent": "healthy"
            },
            uptime_seconds=3600.0
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert len(response.components) == 3

    def test_degraded_status(self):
        """测试降级状态"""
        from agentic_rag.api.schemas import HealthResponse
        
        response = HealthResponse(
            status="degraded",
            version="1.0.0",
            components={
                "database": "healthy",
                "vectorstore": "unhealthy",
                "agent": "healthy"
            },
            uptime_seconds=100.0
        )
        
        assert response.status == "degraded"


class TestErrorResponse:
    """错误响应模型测试"""

    def test_valid_error_response(self):
        """测试有效错误响应"""
        from agentic_rag.api.schemas import ErrorResponse
        
        response = ErrorResponse(
            error="ValidationError",
            message="输入验证失败",
            request_id="req-12345"
        )
        
        assert response.error == "ValidationError"
        assert response.request_id == "req-12345"


class TestDateTimeEncoder:
    """日期时间编码器测试"""

    def test_datetime_encoding(self):
        """测试datetime编码"""
        from agentic_rag.api.routes import DateTimeEncoder
        import json
        
        dt = datetime(2024, 1, 15, 12, 30, 45)
        result = json.dumps({"timestamp": dt}, cls=DateTimeEncoder)
        
        assert "2024-01-15" in result
        assert "12:30:45" in result

    def test_regular_object_encoding(self):
        """测试常规对象编码"""
        from agentic_rag.api.routes import DateTimeEncoder
        import json
        
        result = json.dumps({"value": 123}, cls=DateTimeEncoder)
        
        assert "123" in result
