import requests
import json
import sys

url = "http://127.0.0.1:8000/api/v1/query/stream"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "xiaoyi-cs"  # 使用.env中的API_KEY
}
data = {
    "question": "你好",
    "session_id": "test-session-1"
}

try:
    print("开始测试流式接口...")
    response = requests.post(url, headers=headers, json=data, stream=True, timeout=30)
    
    if response.status_code == 200:
        print(f"请求成功! 状态码: {response.status_code}")
        print("\n接收到的数据流:")
        
        for line in response.iter_lines():
            if line:
                try:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[6:]
                        try:
                            event_data = json.loads(json_str)
                            print(f"事件类型: {event_data.get('type', 'unknown')}")
                            if event_data.get('type') == 'chunk':
                                print(f"内容: {event_data.get('content', '')[:50]}...")
                            elif event_data.get('type') == 'error':
                                print(f"错误: {event_data.get('error', '未知错误')}")
                                print(f"内容: {event_data.get('content', '')}")
                                sys.exit(1)
                            elif event_data.get('type') == 'done':
                                print(f"完成: {event_data.get('content', '')}")
                                break
                        except json.JSONDecodeError:
                            print(f"JSON解析失败: {json_str}")
                    else:
                        print(f"其他数据: {decoded_line}")
                except Exception as e:
                    print(f"处理行时出错: {e}")
    else:
        print(f"请求失败! 状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        sys.exit(1)
        
except requests.exceptions.Timeout:
    print("请求超时!")
    sys.exit(1)
except Exception as e:
    print(f"请求异常: {e}")
    sys.exit(1)
