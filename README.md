# Agentic RAG 智能知识库系统

## 📖 项目简介

Agentic RAG 是一个基于 LangChain/LangGraph 的智能知识库问答系统，支持：

- **多格式文档处理**：PDF、Word、Excel、CSV、Markdown、TXT、网页等
- **向量检索**：基于 Milvus 的向量数据库存储与检索
- **Agent 智能问答**：支持反思、多轮对话、工具调用（DuckDuckGo 搜索、计算器）
- **语义重排**：BGE 重排模型优化检索结果
- **流式响应**：支持 SSE 流式输出
- **长短记忆**：短期记忆（PostgreSQL）+ 长期记忆（PostgreSQL + pgvector）
- **多级缓存**：意图缓存、生成缓存、LLM 调用缓存
- **定时任务**：短期记忆过期清理、长期记忆自动归档

---

## 🛠️ 环境配置

### 1. 复制环境配置文件

```bash
cp .env.example .env
```

### 2. 填写必要的 API 密钥和配置

编辑 `.env` 文件，填入以下配置：

```env
# ========== LLM 配置 ==========
QWEN_API_KEY=你的通义千问API密钥
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.5-plus

# ========== 嵌入模型配置 ==========
ZHIPUAI_API_KEY=你的智谱AI API密钥
ZP_XIANG_LIANG_MODEL=Embedding-3

# ========== 向量数据库配置 ==========
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=root
MILVUS_PASSWORD=milvus

# ========== 数据库配置 ==========
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/postgres

# ========== 应用配置 ==========
API_PORT=8000

# ========== API 认证密钥 ==========
API_KEY=your_api_key_here
```

### 3. 安装依赖

```bash
uv sync
```

### 4. 启动依赖服务

确保以下服务已启动：
- **Milvus**：向量数据库（默认端口 19530）
- **PostgreSQL**：记忆存储（默认端口 5433）

---

## 🚀 启动服务

### 方式一：直接运行

```bash
python main.py
```

服务启动后：
- API 地址：`http://localhost:8000`
- API 文档：`http://localhost:8000/api/docs`

### 方式二：后台运行

```bash
nohup python main.py > app.log 2>&1 &
```

---

## 📚 如何存文档到向量库

### 通过 API 上传（推荐）

**接口**：`POST /api/v1/upload`

**请求头**：
```
X-API-Key: your_api_key_here
```

**请求体**：multipart/form-data

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| files | File | ✅ | 上传的文件（支持多文件） |
| chunk_size | int | ❌ | 分块大小，默认 500 字符 |
| chunk_overlap | int | ❌ | 分块重叠，默认 50 字符 |

**支持的文件格式**：
- PDF (.pdf)
- Word (.docx)
- Markdown (.md)
- 纯文本 (.txt)
- Excel (.xlsx, .xls)
- CSV (.csv)

**示例（curl）**：
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "X-API-Key: your_api_key_here" \
  -F "files=@/path/to/your/document.pdf" \
  -F "chunk_size=500" \
  -F "chunk_overlap=50"
```

**示例（Python）**：
```python
import requests

url = "http://localhost:8000/api/v1/upload"
headers = {"X-API-Key": "your_api_key_here"}
files = {"files": open("document.pdf", "rb")}
data = {"chunk_size": 500, "chunk_overlap": 50}

response = requests.post(url, headers=headers, files=files, data=data)
print(response.json())
```

**响应示例**：
```json
{
  "success": true,
  "documents_processed": 1,
  "chunks_created": 15,
  "message": "成功处理 1 个文件，创建 15 个分块"
}
```

### 通过代码上传

```python
from agentic_rag.vectorstore.milvus_client import MilvusClient
from agentic_rag.document_processing.loaders import DocumentLoader
from agentic_rag.document_processing.splitters import get_splitter

# 初始化组件
vectorstore = MilvusClient(collection_name="table_agentic_rag")

# 加载文档
loader = DocumentLoader()
documents = loader.load("path/to/your/document.pdf")

# 分块处理
splitter = get_splitter("semantic", chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 存入向量库
vectorstore.from_documents(chunks)
print(f"已存入 {len(chunks)} 个文档块")
```

---

## ❓ 如何查询

### 单次查询

**接口**：`POST /api/v1/query`

**请求头**：
```
X-API-Key: your_api_key_here
Content-Type: application/json
```

**请求体**：
```json
{
  "question": "你的问题",
  "session_id": "可选的会话ID",
  "user_id": "可选的用户ID",
  "use_tools": true,
  "max_reflection": 2,
  "temperature": 0.7
}
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| question | string | ✅ | 用户问题（1-2000字符） |
| session_id | string | 自动生成 | 会话ID，用于多轮对话 |
| user_id | string | null | 用户ID，用于长期记忆 |
| use_tools | bool | true | 是否启用工具（DuckDuckGo搜索等） |
| max_reflection | int | 2 | Agent 反思次数（0-5） |
| temperature | float | 0.7 | 生成随机性（0-2） |

**示例（curl）**：
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是人工智能？"}'
```

**示例（Python）**：
```python
import requests

url = "http://localhost:8000/api/v1/query"
headers = {
    "X-API-Key": "your_api_key_here",
    "Content-Type": "application/json"
}
payload = {
    "question": "什么是人工智能？",
    "use_tools": True,
    "max_reflection": 2
}

response = requests.post(url, headers=headers, json=payload)
result = response.json()

print("回答:", result["answer"])
print("来源:", result["sources"])
print("处理时间:", result["processing_time"], "秒")
```

**响应示例**：
```json
{
  "answer": "人工智能（Artificial Intelligence，AI）是...",
  "sources": [
    {
      "content": "人工智能是计算机科学的一个分支...",
      "metadata": {"source": "document.pdf", "page": 1},
      "score": 0.95,
      "source": "document.pdf"
    }
  ],
  "session_id": "abc123",
  "intent": "definition",
  "tools_used": [],
  "reflection_count": 1,
  "processing_time": 2.35
}
```

### 流式查询

**接口**：`POST /api/v1/query/stream`

返回 SSE 格式的流式响应，实时显示生成过程。

---

## 🔧 系统维护

### 查看系统健康状态

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**响应示例**：
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "vectorstore": "healthy",
    "agent": "healthy"
  },
  "uptime_seconds": 3600.5
}
```

### 查看日志

```bash
tail -f app.log
```

---

## 📁 项目结构

```
agentic_rag/
├── agent/                  # Agent 核心（状态、节点、边、图）
├── api/                    # API 接口（FastAPI 路由、数据模型）
├── config/                 # 配置（设置、日志）
├── db_sql/                 # 数据库 SQL 脚本
├── document_processing/    # 文档处理（加载器、分块器）
├── evaluation/             # 评估指标
├── lock/                   # 分布式锁（Redis、PostgreSQL）
├── memory/                 # 记忆系统（短期、长期、缓存）
├── models/                 # 模型封装
├── retrieval/              # 检索（混合搜索、重排、查询改写）
├── schedulers/             # 定时任务调度器（短期/长期记忆清理）
├── tools/                  # 工具（DuckDuckGo搜索、计算器）
├── ui/                     # Streamlit 前端
├── vectorstore/            # 向量存储（Milvus、嵌入模型）
├── main.py                 # 主程序入口
└── .env                    # 环境变量配置
```

---

## ❓ 常见问题

**Q: 上传文档失败？**
- 检查 Milvus 服务是否正常运行
- 检查 API Key 是否正确
- 确认文件格式是否支持

**Q: 查询返回空结果？**
- 确认已上传相关文档
- 尝试调整问题表述
- 检查向量库是否已初始化

**Q: 如何查看已上传的文档？**
- 通过 `/api/v1/health` 查看向量库状态
- 向量库 collection 信息会返回文档数量