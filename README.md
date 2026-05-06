# Agentic RAG 智能知识库系统

## 📖 项目简介

Agentic RAG 是一个基于 LangChain/LangGraph 的**四路径智能知识库问答系统**，具有自主决策能力的 Agent 系统：

- **RAG能力**：向量检索 + 查询改写 + 混合检索 + BGE重排
- **Agent能力**：双模式执行（DAG/ReAct）+ 工具调用（DuckDuckGo搜索、计算器）
- **记忆能力**：短期会话记忆（PostgreSQL）+ 长期价值记忆（pgvector）
- **四路径执行**：超快速并行 → 快速流式 → 标准DAG → ReAct复杂推理

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           API 网关层                                     │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│   │  请求追踪   │  │  限流保护   │  │  API认证    │  │  CORS防护   │   │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agent 执行层 (四路径)                                │
│                                                                           │
│   ┌──────────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│   │ 超快速并行路径        │  │  快速流式路径   │  │  DAG 模式          │  │
│   │ ultra_fast_stream    │  │ fast_stream     │  │  (预定义路径)       │  │
│   │ ParallelExecutor     │  │ 手动执行+流式   │  │  graph.astream     │  │
│   │ 三路并行+真流式      │  │ 首token<3s     │  │ 真流式generation   │  │
│   │ 首token<2s           │  │                │  │                    │  │
│   └──────────────────────┘  └────────────────┘  └────────────────────┘  │
│                                                                           │
│   ┌────────────────────┐                                                  │
│   │  ReAct 模式        │  ← 复杂问题自动切换（score≥0.6）                  │
│   │  (自主决策)        │                                                  │
│   │  Observe→Think→Act │                                                  │
│   │  token级流式       │                                                  │
│   └────────────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           工具执行层                                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐         │
│   │ 向量检索 │  │ 网络搜索 │  │ 查询改写 │  │ 外部工具(DuckDuck │         │
│   │ (内置)   │  │ (内置)   │  │ (内置)   │  │ Go,Calculator)   │         │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 四路径执行架构

| 路径 | 入口方法 | 核心策略 | 首 Token 时间 |
|------|---------|---------|-------------|
| **超快速并行** | `ultra_fast_stream_invoke` | ParallelExecutor 三路并行 + 真流式 | **~2-3s** |
| **快速流式** | `fast_stream_invoke` | 手动执行节点 + 真流式 | **~3-5s** |
| **标准 DAG** | `stream_invoke` | 手动执行节点 + 真流式generation | **~5-15s** |
| **ReAct** | `stream_run` | Observe→Think→Act + 真流式（复杂问题自动切换） | **~5-15s** |

### 模式自动切换

- **默认模式**：`dag`（快速流式，响应优先）
- **自动切换**：当问题复杂度 score ≥ 0.6 时，自动切换到 ReAct 模式
- **触发特征**：多步骤推理、调试/排错、深度分析、算法设计、原因分析

---

## ⚡ 性能优化

### 优化后时间线对比

```
【优化后 - stream_invoke (DAG真流式)】
0s──────────────────────────────────────────────────15s
[记忆][意图+改写][检索][重排][首token!→token→token→...]
                                                      ↑ 用户在这里看到答案!

后台: [eval][CRAG?][记忆保存]

【快速流式 - fast_stream_invoke】
0s──────────────────────────────────5s─────────────────────15s
[记忆][意图+改写][检索][重排][首token!→token→token→...]
                              ↑ 用户在这里就看到答案了!

后台: [eval][记忆保存]  ← 不阻塞用户

【终极优化 - ultra_fast_stream_invoke】
0s────────────────────3s─────────────────────12s
[意图+改写][Memory‖Retrieval‖WebSearch][Rerank][首token!→token→...]
           ↑ 三路并行，不等待Memory      ↑ 用户在这里看到答案!
```

### 核心优化点

| 优化项 | 效果 |
|--------|------|
| 真流式生成（llm.astream） | 首token从8-15s降至0.5s |
| 后台异步评估 | 不阻塞用户感知 |
| 移除CRAG阻塞 | 最坏情况节省20s |
| ParallelExecutor三路并行 | Memory+Retrieval+WebSearch同时执行 |
| 生成缓存 | 重复问题<0.5s响应 |

---

## 🚀 快速开始

### 1. 环境配置

```bash
cp .env.example .env
# 编辑 .env 填入 API 密钥
```

### 2. 安装依赖

```bash
uv sync
```

### 3. 启动服务

```bash
python main.py
```

服务启动后访问：`http://localhost:8000/api/docs`

---

## 📚 功能特性

### 多格式文档处理
PDF、Word、Excel、CSV、Markdown、TXT、网页等

### 混合检索
向量检索（0.6权重）+ BM25关键词检索（0.4权重）+ BGE重排

### 四路径执行
1. **超快速并行**：`ultra_fast_stream_invoke` - Memory+Retrieval+WebSearch三路并行
2. **快速流式**：`fast_stream_invoke` - 绕过graph直接llm.astream()真流式
3. **标准DAG**：`stream_invoke` - 手动执行节点 + 真流式generation
4. **ReAct**：`stream_run` - Observe→Think→Act自主决策循环

### 工具调用
- DuckDuckGo 搜索
- Calculator 计算器
- Python REPL

### 记忆系统
- **短期记忆**：PostgreSQL会话持久化，24h TTL，Token级上下文截断
- **长期记忆**：pgvector向量存储，价值筛选+四类型分类+时间衰减

### 流式响应
SSE格式流式输出，支持逐token显示

---

## ❓ 如何查询

### 单次查询

**接口**：`POST /api/v1/query`

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是人工智能？"}'
```

**请求参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| question | string | ✅ | 用户问题（1-2000字符） |
| session_id | string | 自动生成 | 会话ID，用于多轮对话 |
| user_id | string | null | 用户ID，用于长期记忆 |
| model_name | string | qwen3.5-plus | 模型名称（qwen/MiniMax/glm） |
| use_tools | bool | true | 是否启用工具（DuckDuckGo搜索等） |
| use_fast_path | bool | true | 快速流式模式（真流式输出+后台评估） |
| max_reflection | int | 2 | 反思次数（0-5） |
| temperature | float | 0.7 | 生成随机性（0-2） |
| mode | string | dag | 执行模式：`dag`（默认）或 `react`（复杂推理） |

**mode 参数说明**：
- `dag`：默认模式，快速流式响应，适合简单/中等复杂度问题
- `react`：ReAct模式，复杂推理场景自动切换，适合多步骤推理、调试、深度分析

**响应字段**：

| 字段 | 说明 |
|------|------|
| answer | AI回答内容 |
| sources | 来源文档列表 |
| metrics | 评估指标（faithfulness、answer_relevancy等） |
| session_id | 会话ID |
| intent | 识别的意图类型 |
| mode_used | 实际使用的执行模式（dag/react） |
| mode_reason | 模式切换原因（自动切换时填充） |
| tools_used | 使用的工具列表 |
| reflection_count | 反思次数 |
| processing_time | 处理时间（秒） |

**响应示例**：
```json
{
  "answer": "人工智能（Artificial Intelligence，AI）是...",
  "sources": [...],
  "metrics": {"faithfulness": 0.85, "answer_relevancy": 0.92},
  "session_id": "abc123",
  "intent": "factual",
  "mode_used": "dag",
  "mode_reason": null,
  "tools_used": [],
  "reflection_count": 0,
  "processing_time": 2.35
}
```

### 流式查询

**接口**：`POST /api/v1/query/stream`

返回 SSE 格式的流式响应，支持实时显示生成过程。

```bash
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"question": "解释一下为什么会出现这个错误"}'
```

---

## 📤 如何存文档

### 通过 API 上传

**接口**：`POST /api/v1/upload`

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "X-API-Key: your_api_key_here" \
  -F "files=@/path/to/document.pdf" \
  -F "chunk_size=500" \
  -F "chunk_overlap=50"
```

**支持格式**：PDF、Word、Markdown、TXT、Excel、CSV

---

## 🔧 系统维护

### 健康检查

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

---

## 📁 项目结构

```
agentic_rag/
├── agent/                      # Agent 核心
│   ├── graph.py               # 四路径执行入口
│   ├── complexity_analyzer.py # 问题复杂度分析
│   ├── react/                 # ReAct Agent
│   │   └── react_agent.py
│   └── nodes/                 # DAG节点
├── api/                       # API 接口
│   ├── routes.py             # 路由
│   └── schemas.py            # 数据模型
├── config/                    # 配置
├── document_processing/       # 文档处理
├── evaluation/                # 评估指标
├── lock/                      # 分布式锁
├── memory/                    # 记忆系统
│   ├── gen_cache.py          # 生成缓存（双键写入）
│   ├── short_term_memory.py
│   └── long_term_memory.py
├── models/                    # 模型封装
├── retrieval/                 # 检索
│   ├── hybrid_search.py      # 混合检索
│   └── rerank.py             # BGE重排
├── schedulers/               # 定时任务
├── tools/                    # 工具
├── vectorstore/              # 向量存储
├── main.py                   # 主程序入口
└── .env                      # 环境变量配置
```

---

## ❓ 常见问题

**Q: 上传文档失败？**
- 检查 Milvus 服务是否正常运行
- 检查 API Key 是否正确

**Q: 查询返回空结果？**
- 确认已上传相关文档
- 尝试调整问题表述

**Q: 如何选择执行模式？**
- `mode=dag`（默认）：简单问题，响应优先
- `mode=react`：复杂推理、调试、多步决策
- 不指定mode时，系统根据问题复杂度自动选择