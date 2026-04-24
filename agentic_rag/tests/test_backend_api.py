"""
后端API测试脚本 - 使用requests测试后端接口
"""
import requests
import json
import time
import sys

API_BASE = "http://localhost:8000"
API_KEY = "xiaoyi-cs"

# 修复Windows控制台编码问题
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def test_backend_api():
    """测试后端API接口"""
    print("=" * 60)
    print("Start Testing Backend API")
    print("=" * 60)
    
    errors = []
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        # 1. 健康检查
        print("\n[Step 1] Health Check /api/v1/health")
        response = requests.get(f"{API_BASE}/api/v1/health", headers={"X-API-Key": API_KEY}, timeout=10)
        print(f"Status: {response.status_code}")
        health_data = response.json()
        print(f"Response: status={health_data.get('status')}, version={health_data.get('version')}")
        print(f"Components: {health_data.get('components')}")
        
        if response.status_code == 200 and health_data.get("status") in ["healthy", "degraded"]:
            print("[OK] Service is healthy")
        else:
            errors.append(f"Service status: {health_data.get('status')}")
        
        # 2. 测试查询接口
        print("\n[Step 2] Test Query API /api/v1/query")
        query_payload = {
            "question": "What is Python?",
            "session_id": "test-session-001",
            "use_tools": False,
            "max_reflection": 0,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_BASE}/api/v1/query",
            headers=headers,
            json=query_payload,
            timeout=120
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('answer', 'None')
            print(f"Answer: {answer[:200]}...")
            print(f"Intent: {result.get('intent', 'unknown')}")
            print(f"Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"Reflection count: {result.get('reflection_count', 0)}")
            
            sources = result.get('sources', [])
            print(f"Sources count: {len(sources)}")
            if sources:
                print("Sample sources:")
                for i, source in enumerate(sources[:2], 1):
                    print(f"  {i}. {source.get('source', 'unknown')[:50]} (score: {source.get('score', 0):.4f})")
            
            print("[OK] Query API works correctly")
        else:
            print(f"Error response: {response.text[:500]}")
            errors.append(f"Query API failed: HTTP {response.status_code}")
        
        # 3. 测试流式查询接口
        print("\n[Step 3] Test Stream API /api/v1/query/stream")
        stream_payload = {
            "question": "Hello",
            "session_id": "test-session-002",
            "use_tools": False,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_BASE}/api/v1/query/stream",
            headers=headers,
            json=stream_payload,
            timeout=120,
            stream=True
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("[OK] Stream API is available")
            event_count = 0
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        event_count += 1
                        if event_count <= 3:
                            data_str = line[6:]
                            try:
                                data = json.loads(data_str)
                                print(f"  Event {event_count}: type={data.get('type')}, content={str(data.get('content', ''))[:80]}...")
                            except:
                                pass
                        if event_count >= 10:
                            break
            
            print(f"  Total events received: {event_count}")
        else:
            errors.append(f"Stream API failed: HTTP {response.status_code}")
        
        # 4. 测试工具调用
        print("\n[Step 4] Test Tool Calling")
        tools_payload = {
            "question": "What is 1+1?",
            "session_id": "test-session-003",
            "use_tools": True,
            "max_reflection": 1,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_BASE}/api/v1/query",
            headers=headers,
            json=tools_payload,
            timeout=120
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            tools_used = result.get('tools_used', [])
            print(f"Tools used: {tools_used}")
            if tools_used:
                print("[OK] Tool calling is working")
            else:
                print("[WARN] No tools used (question may not need tools)")
        else:
            errors.append("Tool calling API failed")
        
        # 5. 测试模型切换
        print("\n[Step 5] Test Model Switching")
        model_payload = {
            "question": "Hello",
            "session_id": "test-session-005",
            "model_name": "deepseek-v3.2",
            "use_tools": False
        }
        
        response = requests.post(
            f"{API_BASE}/api/v1/query",
            headers=headers,
            json=model_payload,
            timeout=120
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("[OK] Model switching works")
        else:
            errors.append("Model switching failed")
        
        # 6. 测试无效API Key
        print("\n[Step 6] Test API Authentication")
        bad_headers = {"X-API-Key": "invalid-key"}
        response = requests.get(f"{API_BASE}/api/v1/health", headers=bad_headers, timeout=10)
        print(f"Invalid key test - Status: {response.status_code}")
        if response.status_code in [401, 403]:
            print("[OK] API authentication is working")
        else:
            print(f"[WARN] Authentication may have issues, status: {response.status_code}")
        
    except requests.exceptions.ConnectionError as e:
        errors.append(f"Connection error: {e}")
        print(f"[ERROR] Connection failed: {e}")
    except requests.exceptions.Timeout as e:
        errors.append(f"Request timeout: {e}")
        print(f"[ERROR] Request timeout: {e}")
    except Exception as e:
        errors.append(f"Test exception: {e}")
        print(f"[ERROR] Test exception: {e}")
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    if errors:
        print(f"\n[FAIL] Found {len(errors)} issues:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    else:
        print("\n[SUCCESS] All API tests passed!")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = test_backend_api()
    exit(0 if success else 1)