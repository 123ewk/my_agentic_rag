"""
自动化测试运行脚本
运行所有测试并生成测试报告
"""
import pytest
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime


def run_tests(
    test_paths: list = None,
    verbose: bool = True,
    capture_output: bool = False,
    show_coverage: bool = True
):
    """
    运行测试的自动化脚本
    
    参数:
        test_paths: 测试文件路径列表，默认为所有测试
        verbose: 是否详细输出
        capture_output: 是否捕获输出
        show_coverage: 是否显示覆盖率
    
    返回:
        测试结果 (成功: True, 失败: False)
    """
    # 设置工作目录
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # 默认测试路径
    if test_paths is None:
        test_paths = [
            "agentic_rag/tests/test_document_processing.py",
            "agentic_rag/tests/test_caches.py",
            "agentic_rag/tests/test_vectorstore.py",
            "agentic_rag/tests/test_config.py",
            "agentic_rag/tests/test_locks.py",
            "agentic_rag/tests/test_api.py",
            "agentic_rag/tests/test_edges.py",
            "agentic_rag/tests/test_nodes.py",
            "agentic_rag/tests/test_state.py",
            "agentic_rag/tests/test_retrieval.py",
            "agentic_rag/tests/test_reranker.py",
            "agentic_rag/tests/test_evaluation.py",
            "agentic_rag/tests/test_tools.py",
            "agentic_rag/tests/test_integration.py",
        ]
    
    # 构建pytest参数
    pytest_args = [
        "-v" if verbose else "-q",
        "--tb=short",
        "--color=yes",
        "-x",  # 遇到第一个失败就停止
        "--strict-markers",
        "--disable-warnings",
    ]
    
    # 添加测试路径
    pytest_args.extend(test_paths)
    
    print("=" * 70)
    print(f"开始运行测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"工作目录: {os.getcwd()}")
    print(f"测试路径: {test_paths}")
    print("=" * 70)
    
    # 运行pytest
    result = pytest.main(pytest_args)
    
    # 返回结果
    success = result == 0
    
    print("=" * 70)
    if success:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败")
    print("=" * 70)
    
    return success


def run_unit_tests_only():
    """仅运行单元测试（不包含集成测试）"""
    test_paths = [
        "agentic_rag/tests/test_document_processing.py",
        "agentic_rag/tests/test_caches.py",
        "agentic_rag/tests/test_vectorstore.py",
        "agentic_rag/tests/test_config.py",
        "agentic_rag/tests/test_locks.py",
        "agentic_rag/tests/test_api.py",
        "agentic_rag/tests/test_edges.py",
        "agentic_rag/tests/test_nodes.py",
        "agentic_rag/tests/test_state.py",
        "agentic_rag/tests/test_retrieval.py",
        "agentic_rag/tests/test_reranker.py",
        "agentic_rag/tests/test_evaluation.py",
        "agentic_rag/tests/test_tools.py",
    ]
    
    return run_tests(test_paths)


def run_integration_tests_only():
    """仅运行集成测试"""
    test_paths = [
        "agentic_rag/tests/test_integration.py",
    ]
    
    return run_tests(test_paths)


def run_specific_module_tests(module_name: str):
    """运行特定模块的测试"""
    test_paths = {
        "document": ["agentic_rag/tests/test_document_processing.py"],
        "cache": ["agentic_rag/tests/test_caches.py"],
        "vectorstore": ["agentic_rag/tests/test_vectorstore.py"],
        "config": ["agentic_rag/tests/test_config.py"],
        "lock": ["agentic_rag/tests/test_locks.py"],
        "api": ["agentic_rag/tests/test_api.py"],
        "agent": [
            "agentic_rag/tests/test_edges.py",
            "agentic_rag/tests/test_nodes.py",
            "agentic_rag/tests/test_state.py",
        ],
        "retrieval": [
            "agentic_rag/tests/test_retrieval.py",
            "agentic_rag/tests/test_reranker.py",
        ],
        "evaluation": ["agentic_rag/tests/test_evaluation.py"],
        "tools": ["agentic_rag/tests/test_tools.py"],
    }
    
    if module_name not in test_paths:
        print(f"未知模块: {module_name}")
        print(f"可用模块: {', '.join(test_paths.keys())}")
        return False
    
    return run_tests(test_paths[module_name])


def main():
    """主函数 - 根据命令行参数运行不同类型的测试"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "unit":
            print("运行单元测试...")
            success = run_unit_tests_only()
        elif command == "integration":
            print("运行集成测试...")
            success = run_integration_tests_only()
        elif command == "all":
            print("运行所有测试...")
            success = run_tests()
        elif command.startswith("module:"):
            module_name = command.split(":", 1)[1]
            print(f"运行 {module_name} 模块测试...")
            success = run_specific_module_tests(module_name)
        else:
            print(f"未知命令: {command}")
            print("用法:")
            print("  python run_all_tests.py all           - 运行所有测试")
            print("  python run_all_tests.py unit         - 仅运行单元测试")
            print("  python run_all_tests.py integration  - 仅运行集成测试")
            print("  python run_all_tests.py module:<name> - 运行特定模块测试")
            print("  可用模块: document, cache, vectorstore, config, lock, api, agent, retrieval, evaluation, tools")
            success = False
    else:
        # 默认运行所有测试
        print("运行所有测试（默认）...")
        success = run_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
