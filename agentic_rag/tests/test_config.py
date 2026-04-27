"""
配置模块测试
测试配置加载和环境变量处理
"""
import pytest
import os
from unittest.mock import patch, MagicMock


class TestSettings:
    """设置类测试"""

    def test_settings_defaults(self):
        """测试默认配置值"""
        from agentic_rag.config.settings import Settings
        
        settings = Settings()
        
        assert settings.llm_name == "Qwen3.5-plus"
        assert settings.llm_temperature == 0.7
        assert settings.llm_max_tokens == 4096
        assert settings.retrieval_top_k == 5
        assert settings.rerank_top_k == 3
        assert settings.similarity_threshold == 0.5
        assert settings.max_reflection_steps == 2
        assert settings.api_port == 8000
        assert settings.intent_cache_enabled is True
        assert settings.intent_cache_max_size == 200
        assert settings.generation_cache_enabled is True

    def test_settings_from_env(self):
        """测试从环境变量加载配置"""
        with patch.dict(os.environ, {
            "QWEN_API_KEY": "test_qwen_key",
            "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
            "API_PORT": "9000"
        }):
            from agentic_rag.config.settings import Settings
            
            settings = Settings()
            
            assert settings.qwen_api_key == "test_qwen_key"
            assert settings.database_url == "postgresql://test:test@localhost:5432/test"
            assert settings.api_port == 9000

    def test_get_model_config_qwen(self):
        """测试获取Qwen模型配置"""
        from agentic_rag.config.settings import Settings
        
        settings = Settings(
            qwen_api_key="test_key",
            qwen_base_url="https://test.com"
        )
        
        config = settings.get_model_config("deepseek-v3.2")
        
        assert config["api_key"] == "test_key"
        assert config["base_url"] == "https://test.com"

    def test_get_model_config_minimax(self):
        """测试获取MiniMax模型配置"""
        from agentic_rag.config.settings import Settings
        
        settings = Settings(
            minimax_api_key="minimax_key",
            minimax_base_url="https://minimax.com"
        )
        
        config = settings.get_model_config("MiniMax-M2.7")
        
        assert config["api_key"] == "minimax_key"
        assert config["base_url"] == "https://minimax.com"

    def test_get_model_config_unknown(self):
        """测试获取未知模型配置返回默认"""
        from agentic_rag.config.settings import Settings
        
        settings = Settings(
            qwen_api_key="default_key",
            qwen_base_url="https://default.com"
        )
        
        config = settings.get_model_config("unknown-model")
        
        assert config["api_key"] == "default_key"
        assert config["base_url"] == "https://default.com"

    def test_get_available_models_with_qwen(self):
        """测试获取可用模型列表（Qwen）"""
        with patch.dict(os.environ, {"QWEN_API_KEY": "test_key"}):
            from agentic_rag.config.settings import Settings
            
            settings = Settings()
            models = settings.get_available_models()
            
            assert "deepseek-v3.2" in models

    def test_get_available_models_with_minimax(self):
        """测试获取可用模型列表（MiniMax）"""
        with patch.dict(os.environ, {
            "MINIMAX_API_KEY": "minimax_key",
            "MINIMAX_MODEL": "MiniMax-M2.7"
        }):
            from agentic_rag.config.settings import Settings
            
            settings = Settings()
            models = settings.get_available_models()
            
            assert "MiniMax-M2.7" in models

    def test_get_available_models_empty(self):
        """测试没有API密钥时返回默认模型"""
        with patch.dict(os.environ, {
            "QWEN_API_KEY": "",
            "MINIMAX_API_KEY": ""
        }, clear=False):
            from agentic_rag.config.settings import Settings
            
            settings = Settings()
            models = settings.get_available_models()
            
            assert "deepseek-v3.2" in models

    def test_singleton_pattern(self):
        """测试单例模式"""
        from agentic_rag.config.settings import get_settings
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2


class TestLoggerConfig:
    """日志配置测试"""

    def test_logger_setup(self):
        """测试日志设置"""
        from agentic_rag.config.logger_config import setup_logging
        
        logger = setup_logging()
        
        assert logger is not None

    def test_logger_returns_logger_instance(self):
        """测试返回logger实例"""
        from agentic_rag.config.logger_config import setup_logging
        import logging
        
        logger = setup_logging()
        
        assert isinstance(logger, logging.Logger)
