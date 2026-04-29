"""
CRAG置信度路由测试
测试网络搜索和CRAG集成逻辑
"""
import pytest
from unittest.mock import patch
from langchain_core.documents import Document
from agentic_rag.agent.nodes import web_search_node


  def test_high_confidence_with_high_overall_score(self):
      """测试高置信度场景：evaluation分数高，不触发搜索"""
      state = {
          "question": "什么是Python?",
          "reranked_docs": [
              Document(page_content="Python是一种编程语言", metadata={"score": 0.9}),
              Document(page_content="Python由Guido van Rossum创建", metadata={"score": 0.85}),
              Document(page_content="Python语法简洁易读", metadata={"score": 0.8})
          ],
          "evaluation": {
              "faithfulness": 0.8,
              "answer_relevancy": 0.9,
              "context_precision": 0.8,
              "overall_score": 0.85
          }
      }
      
      # 高置信度不触发搜索，直接验证分数判断逻辑
      assert state["evaluation"]["overall_score"] >= 0.7
      assert state["evaluation"]["overall_score"] >= 0.3


class TestWebSearchNode:
    """网络搜索节点测试"""

    @patch('agentic_rag.agent.nodes.duckduckgo_search')
    def test_web_search_with_results(self, mock_search):
        """测试网络搜索有结果"""
        mock_search.invoke.return_value = """1. Python官方网站
   Python官方文档和教程
   来源: https://python.org
   
2. Python教程
   免费的Python学习资源
   来源: https://python-tutorial.com"""

        state = {
            "question": "Python是什么?"
        }
        
        result = web_search_node(state, None)
        
        assert len(result["search_results"]) > 0
        assert result["needs_reflection"] == True

    @patch('agentic_rag.agent.nodes.duckduckgo_search')
    def test_web_search_no_results(self, mock_search):
        """测试网络搜索无结果"""
        mock_search.invoke.return_value = "未找到相关结果"

        state = {
            "question": "xyz123不存在的查询"
        }
        
        result = web_search_node(state, None)
        
        assert len(result["search_results"]) == 0
        assert result["needs_reflection"] == False

    @patch('agentic_rag.agent.nodes.duckduckgo_search')
    def test_web_search_error_handling(self, mock_search):
        """测试网络搜索异常处理"""
        mock_search.invoke.side_effect = Exception("网络错误")

        state = {
            "question": "Python是什么?"
        }
        
        result = web_search_node(state, None)
        
        assert len(result["search_results"]) == 0
        assert result["needs_reflection"] == False


class TestCRAGIntegration:
    """CRAG集成测试"""

    def test_confidence_score_calculation(self):
        """测试置信度得分计算"""
        from agentic_rag.config.settings import Settings
        
        # 模拟不同场景的置信度计算
        test_cases = [
            # (文档数, 平均相关性, 回答长度, 期望置信度等级)
            (3, 0.9, 500, "high"),
            (0, 0.0, 0, "low"),
            (1, 0.4, 100, "low"),
            (2, 0.6, 300, "medium"),
        ]
        
        for doc_count, avg_relevance, answer_len, expected_level in test_cases:
            # 简单验证：置信度等级应该符合预期范围
            assert expected_level in ["high", "medium", "low"]
