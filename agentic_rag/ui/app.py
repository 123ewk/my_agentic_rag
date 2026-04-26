"""
Streamlit前端界面 - 生产级实现
"""
import streamlit as st
from pathlib import Path
import time
import requests
from datetime import datetime
from typing import Optional, List, Dict, Any, Generator
import json
import re
from dotenv import load_dotenv

def get_available_models() -> List[str]:
    """
    从环境变量中读取可用的模型列表
    
    返回：
        可用模型名称列表
    """
    load_dotenv()
    import os
    
    models = []
    
    if os.getenv("MINIMAX_API_KEY") and os.getenv("MINIMAX_MODEL"):
        models.append(os.getenv("MINIMAX_MODEL"))
    if os.getenv("QWEN_API_KEY"):
        qwen_model = os.getenv("QWEN_MODEL", "deepseek-v3.2")
        models.append(qwen_model)
    
    return models if models else ["deepseek-v3.2"]

class APIClient:
    """
    API客户端 - 封装与后端的通信
    
    职责：
    1. 管理API基础配置(地址、密钥、超时)
    2. 提供同步查询接口
    3. 提供流式查询接口
    4. 统一的错误处理
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 60,
        use_streaming: bool = True
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.use_streaming = use_streaming
        self._session = None
    
    @property
    def session(self) -> requests.Session:
        """获取或创建持久化的HTTP会话"""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            })
        return self._session
    
    def _build_url(self, endpoint: str) -> str:
        """构建完整的API URL"""
        return f"{self.base_url}/{endpoint.lstrip('/')}"
    
    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_tools: bool = True,
        max_reflection: int = 2,
        temperature: float = 0.7,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        发送同步查询请求
        
        参数：
            question: 用户问题
            session_id: 会话ID(用于多轮对话)
            user_id: 用户ID(用于长期记忆)
            use_tools: 是否启用工具调用
            max_reflection: 最大反思次数
            temperature: 温度参数
            model_name: 模型名称(用于动态选择LLM)
        
        返回：
            API响应数据(包含answer、sources、metrics等)
        
        异常：
            requests.exceptions.RequestException: 网络错误时抛出
            ValueError: API返回错误时抛出
        """
        url = self._build_url("/api/v1/query")
        
        payload = {
            "question": question,
            "session_id": session_id,
            "user_id": user_id,
            "use_tools": use_tools,
            "max_reflection": max_reflection,
            "temperature": temperature,
            "model_name": model_name
        }
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 401:
                raise ValueError("认证失败: API Key无效或缺失")
            elif response.status_code == 403:
                raise ValueError("权限不足: API Key无访问权限")
            elif response.status_code == 429:
                raise ValueError("请求过于频繁，请稍后重试")
            elif response.status_code >= 400:
                error_detail = response.json().get("detail", "未知错误")
                raise ValueError(f"请求失败({response.status_code}): {error_detail}")
            
            return response.json()
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"无法连接到API服务: {self.base_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"请求超时({self.timeout}秒)")
    
    def query_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_tools: bool = True,
        max_reflection: int = 2,
        temperature: float = 0.7,
        model_name: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        发送流式查询请求
        
        参数：
            question: 用户问题
            session_id: 会话ID(用于多轮对话)
            user_id: 用户ID(用于长期记忆)
            use_tools: 是否启用工具调用
            max_reflection: 最大反思次数
            temperature: 温度参数
            model_name: 模型名称(用于动态选择LLM)
        
        产出：
            Dict: SSE格式的事件数据,包含type、content、data等字段
        
        异常：
            requests.exceptions.RequestException: 网络错误时抛出
        """
        url = self._build_url("/api/v1/query/stream")
        
        payload = {
            "question": question,
            "session_id": session_id,
            "user_id": user_id,
            "use_tools": use_tools,
            "max_reflection": max_reflection,
            "temperature": temperature,
            "model_name": model_name
        }
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            
            if response.status_code == 401:
                raise ValueError("认证失败: API Key无效或缺失")
            elif response.status_code == 403:
                raise ValueError("权限不足: API Key无访问权限")
            elif response.status_code == 429:
                raise ValueError("请求过于频繁，请稍后重试")
            elif response.status_code >= 400:
                try:
                    error_detail = response.json().get("detail", "未知错误")
                except:
                    error_detail = f"HTTP {response.status_code}"
                raise ValueError(f"请求失败: {error_detail}")
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip():
                            try:
                                data = json.loads(data_str)
                                yield data
                            except json.JSONDecodeError:
                                continue
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"无法连接到API服务: {self.base_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"请求超时({self.timeout}秒)")
    
    def check_health(self) -> Dict[str, Any]:
        """
        健康检查
        
        返回：
            健康状态信息(包含status、components等)
        """
        url = self._build_url("/api/v1/health")
        
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def close(self):
        """关闭会话"""
        if self._session:
            self._session.close()
            self._session = None

    def upload_documents(self, files, chunk_size=500, chunk_overlap=50):
        url = self._build_url("/api/v1/upload")
        files_data = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        
        # 创建临时session用于文件上传（不设置Content-Type让requests自动处理）
        upload_session = requests.Session()
        upload_session.headers.update({"X-API-Key": self.api_key})
        
        response = upload_session.post(
            url, 
            files=files_data, 
            data={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}, 
            timeout=120
        )
        upload_session.close()
        
        if response.status_code >= 400:
            raise ValueError(response.json().get("detail", "未知错误"))
        return response.json()


def parse_think_content(text: str) -> tuple:
    """
    解析文本中的think标签内容

    参数：
        text: 包含think标签的原始文本

    返回：
        tuple: (思考内容列表, 清理后的回复文本)
    """
    if not text:
        return [], ""

    think_content = []

    think_pattern = r'<think\b[^>]*>(.*?)</think\s*>'
    think_matches = re.finditer(think_pattern, text, re.DOTALL)

    for match in think_matches:
        content = match.group(1).strip()
        if content:
            think_content.append(content)

    cleaned_text = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()

    return think_content, cleaned_text


def has_think_content(text: str) -> bool:
    """
    检查文本中是否包含think标签

    参数：
        text: 待检查的文本

    返回：
        bool: 是否包含think标签
    """
    if not text:
        return False
    return bool(re.search(r'<think\b[^>]*>.*?</think\s*>', text, re.DOTALL))


def render_thinking_section(think_content: List[str]):
    """
    渲染思考过程部分（可折叠）

    参数：
        think_content: 思考内容列表
    """
    if not think_content:
        return

    total_thinks = len(think_content)

    with st.expander(f"🧠 **思考过程** ({total_thinks} 段，点击展开查看)", expanded=False):
        for i, think in enumerate(think_content, 1):
            st.markdown(f"**🤔 思考 {i}:**")
            st.markdown(think)
            if i < total_thinks:
                st.markdown("---")


def render_thinking_stream(think_buffer: str, think_placeholder):
    """
    流式渲染思考内容

    参数：
        think_buffer: 当前的思考内容缓冲区
        think_placeholder: Streamlit占位符
    """
    if think_buffer:
        with think_placeholder.container():
            st.markdown("**🧠 思考中...**")
            st.info(think_buffer + "▌")


def init_session_state():
    """初始化Streamlit会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    
    if "api_client" not in st.session_state:
        st.session_state.api_client = None


def render_sidebar() -> Dict[str, Any]:
    """
    渲染侧边栏配置界面
    
    返回：
        包含用户配置的字典
    """
    with st.sidebar:
        st.header("⚙️ 配置")
        
        st.subheader("🔌 API连接")
        api_base_url = st.text_input(
            "API地址",
            value=st.session_state.get("api_base_url", "http://localhost:8000"),
            placeholder="http://localhost:8000",
            help="后端API服务地址"
        )
        st.session_state.api_base_url = api_base_url
        
        api_key = st.text_input(
            "API密钥",
            type="password",
            value=st.session_state.get("api_key", ""),
            placeholder="输入您的API密钥",
            help="后端API访问密钥"
        )
        st.session_state.api_key = api_key
        
        st.info("💡 请先填写API地址和密钥，然后点击下方按钮初始化连接后再使用聊天功能")
        
        use_streaming = st.checkbox(
            "启用流式响应",
            value=st.session_state.get("use_streaming", True),
            help="启用后答案将逐字显示"
        )
        
        if st.button("🔄 初始化API连接", type="primary", use_container_width=True):
            if api_base_url and api_key:
                st.session_state.api_client = APIClient(
                    base_url=api_base_url,
                    api_key=api_key,
                    timeout=60,
                    use_streaming=use_streaming
                )
                st.session_state.use_streaming = use_streaming
                with st.spinner("检查连接状态..."):
                    health = st.session_state.api_client.check_health()
                    if health.get("status") in ["healthy", "degraded"]:
                        st.success(f"✅ 连接成功 (状态: {health.get('status')})")
                    else:
                        st.error(f"⚠️ 服务异常: {health.get('error', '未知错误')}")
                        st.session_state.api_client = None
            else:
                st.warning("⚠️ 请填写API地址和密钥")
        
        st.markdown("---")
        
        st.subheader("🤖 模型参数")
        available_models = get_available_models()
        current_model = st.session_state.get("model_name", available_models[0] if available_models else "deepseek-chat")
        try:
            default_index = available_models.index(current_model)
        except ValueError:
            default_index = 0
        model_name = st.selectbox(
            "选择模型",
            available_models,
            index=default_index
        )
        st.session_state.model_name = model_name
        
        retrieval_k = st.slider(
            "检索数量",
            min_value=1,
            max_value=10,
            value=5
        )
        st.session_state.retrieval_k = retrieval_k
        
        temperature = st.slider(
            "温度参数",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        st.session_state.temperature = temperature
        
        st.markdown("---")
        
        st.subheader("🧠 高级功能")
        use_tools = st.checkbox("启用工具调用", value=True)
        use_reflection = st.checkbox("启用反思机制", value=True)
        max_reflection = 2 if use_reflection else 0
        
        st.markdown("---")
        
        with st.expander("📁 上传文档"):
            client = st.session_state.api_client
            
            if not client:
                st.warning("⚠️ 请先在侧边栏初始化API连接")
            else:
                uploaded_files = st.file_uploader(
                    "选择文件",
                    type=['pdf', 'docx', 'txt', 'md', 'csv', 'xlsx', 'xls', 'xlsm', 'html', 'htm'],
                    accept_multiple_files=True,
                    key="doc_uploader"
                )
                
                if uploaded_files:
                    if st.button("📤 上传并索引", type="primary", use_container_width=True):
                        try:
                            with st.spinner("正在处理文档..."):
                                result = client.upload_documents(uploaded_files)
                                if result.get("success"):
                                    st.success(f"✅ {result.get('message', '文档上传成功')}")
                                    st.info(f"📊 处理了 {result.get('documents_processed', 0)} 个文件，创建了 {result.get('chunks_created', 0)} 个分块")
                                else:
                                    st.error(f"❌ 上传失败: {result.get('message', '未知错误')}")
                        except ValueError as e:
                            st.error(f"⚠️ {str(e)}")
                        except Exception as e:
                            st.error(f"❌ 处理失败: {str(e)}")
                else:
                    st.info("👆 请先上传文件")
        
        st.markdown("---")
        
        return {
            "model_name": model_name,
            "retrieval_k": retrieval_k,
            "temperature": temperature,
            "use_tools": use_tools,
            "max_reflection": max_reflection
        }


def render_message(message: Dict[str, Any]):
    """
    渲染单条聊天消息

    参数：
        message: 消息数据字典
    """
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            think_content = message.get("think_content", [])
            answer = message.get("content", "")

            if think_content:
                render_thinking_section(think_content)
                st.markdown("**📝 最终回复:**")
                st.markdown(answer)
            else:
                st.markdown(answer)
        else:
            st.markdown(message["content"])

        if "sources" in message and message["sources"]:
            with st.expander("📄 查看来源"):
                for i, source in enumerate(message["sources"], 1):
                    source_name = source.get("source", f"来源 {i}")
                    score = source.get("score")
                    content = source.get("content", "")

                    st.markdown(f"**{source_name}** {f'(相似度: {score:.2f})' if score else ''}")
                    st.text(content[:300] + "..." if len(content) > 300 else content)
                    st.markdown("---")

        if "metrics" in message and message["metrics"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "忠实度",
                    f"{message['metrics'].get('faithfulness', 0):.2f}"
                )
            with col2:
                st.metric(
                    "相关性",
                    f"{message['metrics'].get('answer_relevancy', 0):.2f}"
                )
            with col3:
                st.metric("反思次数", message.get("reflection_count", 0))

        if "processing_time" in message:
            st.caption(f"⏱️ 处理耗时: {message['processing_time']:.2f}秒")


def render_chat_history():
    """渲染聊天历史"""
    for message in st.session_state.messages:
        render_message(message)


def handle_user_input(
    prompt: str,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    处理用户输入并调用API
    
    参数：
        prompt: 用户输入的问题
        config: 配置参数
    
    返回：
        响应结果字典,失败时返回None
    """
    client = st.session_state.api_client
    
    if client is None:
        st.error("⚠️ 请先在侧边栏初始化API连接")
        return None
    
    try:
        if st.session_state.use_streaming:
            return handle_stream_query(
                prompt=prompt,
                client=client,
                config=config
            )
        else:
            return handle_sync_query(
                prompt=prompt,
                client=client,
                config=config
            )
    
    except ConnectionError as e:
        st.error(f"❌ 连接错误: {e}")
    except TimeoutError as e:
        st.error(f"⏰ 超时错误: {e}")
    except ValueError as e:
        st.error(f"⚠️ {e}")
    except Exception as e:
        st.error(f"❌ 未知错误: {str(e)}")
    
    return None


def handle_sync_query(
    prompt: str,
    client: APIClient,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    处理同步查询请求
    
    参数：
        prompt: 用户问题
        client: API客户端
        config: 配置参数
    
    返回：
        响应结果
    """
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            start_time = time.time()
            
            result = client.query(
                question=prompt,
                session_id=st.session_state.session_id,
                use_tools=config["use_tools"],
                max_reflection=config["max_reflection"],
                temperature=config["temperature"],
                model_name=config.get("model_name")
            )
            
            processing_time = time.time() - start_time
        
        answer = result.get("answer", "抱歉，未能获取到回答")
        sources = result.get("sources", [])
        metrics = result.get("metrics", {})
        reflection_count = result.get("reflection_count", 0)
        
        st.markdown(answer)
        
        if sources:
            with st.expander("📄 查看来源"):
                for i, source in enumerate(sources, 1):
                    source_name = source.get("source", f"来源 {i}")
                    score = source.get("score")
                    content = source.get("content", "")
                    
                    st.markdown(f"**{source_name}** {f'(相似度: {score:.2f})' if score else ''}")
                    st.text(content[:300] + "..." if len(content) > 300 else content)
                    st.markdown("---")
        
        if metrics:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("忠实度", f"{metrics.get('faithfulness', 0):.2f}")
            with col2:
                st.metric("相关性", f"{metrics.get('answer_relevancy', 0):.2f}")
            with col3:
                st.metric("反思次数", reflection_count)
        
        st.caption(f"⏱️ 处理耗时: {processing_time:.2f}秒")
        
        return {
            "answer": answer,
            "sources": sources,
            "metrics": metrics,
            "reflection_count": reflection_count,
            "processing_time": processing_time
        }


def handle_stream_query(
    prompt: str,
    client: APIClient,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    处理流式查询请求

    参数：
        prompt: 用户问题
        client: API客户端
        config: 配置参数

    返回：
        响应结果
    """
    thinking_placeholder = st.empty()
    response_placeholder = st.empty()
    full_response = ""
    sources = []
    metrics = {}
    reflection_count = 0
    start_time = time.time()
    error_occurred = False

    think_content = []
    has_think = False

    try:
        for event in client.query_stream(
            question=prompt,
            session_id=st.session_state.session_id,
            use_tools=config["use_tools"],
            max_reflection=config["max_reflection"],
            temperature=config["temperature"],
            model_name=config.get("model_name")
        ):
            event_type = event.get("type", "")

            if event_type == "status":
                continue

            elif event_type == "chunk" or event_type == "token":
                token = event.get("content", "") or ""

                full_response += token

                think_pattern = r'<think\b[^>]*>(.*?)</think\s*>'
                matches = list(re.finditer(think_pattern, full_response, re.DOTALL))

                if matches:
                    has_think = True
                    new_thinks = []
                    for match in matches:
                        content = match.group(1).strip()
                        if content and content not in think_content:
                            new_thinks.append(content)

                    for new_think in new_thinks:
                        if new_think not in think_content:
                            think_content.append(new_think)

                    if think_content:
                        last_think = think_content[-1]
                        thinking_placeholder.info(f"🧠 **正在思考...** (共 {len(think_content)} 段)\n\n{last_think[:200]}▌")
                elif has_think:
                    thinking_placeholder.info(f"🧠 **思考完成** (共 {len(think_content)} 段)")
                else:
                    thinking_placeholder.empty()

                cleaned = re.sub(think_pattern, '', full_response, flags=re.DOTALL)
                response_placeholder.markdown(cleaned + "▌")

            elif event_type == "sources":
                data = event.get("data", {})
                if isinstance(data, dict) and "documents" in data:
                    for doc in data["documents"]:
                        sources.append(doc)

            elif event_type == "source":
                source_data = event.get("data", {})
                if isinstance(source_data, dict):
                    sources.append(source_data)

            elif event_type == "metrics":
                metrics_data = event.get("data", {})
                if isinstance(metrics_data, dict):
                    metrics = metrics_data

            elif event_type == "reflection":
                reflection_count += 1

            elif event_type == "done":
                event_data = event.get("data", {})
                think_content_from_event = event_data.get("think_content", [])
                has_think_from_event = event_data.get("has_think", False)

                st.sidebar.info(f"🔍 调试: has_think={has_think_from_event}, think_count={len(think_content_from_event)}")

                if think_content:
                    with st.sidebar.expander("🔍 提取的思考内容"):
                        for i, think in enumerate(think_content, 1):
                            st.text(f"思考 {i}: {think[:100]}...")

                if think_content_from_event:
                    for new_think in think_content_from_event:
                        if new_think.strip() and new_think.strip() not in think_content:
                            think_content.append(new_think.strip())
                    has_think = True

                if not full_response.strip():
                    full_response = "抱歉，当前无法生成回答，请检查系统状态或稍后重试。"
                    thinking_placeholder.empty()
                    response_placeholder.markdown(full_response)
                else:
                    thinking_placeholder.empty()
                    response_placeholder.empty()

                    if has_think and think_content:
                        render_thinking_section(think_content)
                        st.markdown("**📝 最终回复:**")

                        cleaned_final = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', full_response, flags=re.DOTALL)
                        st.markdown(cleaned_final)
                    else:
                        st.markdown(full_response)
                break

            elif event_type == "error":
                error_occurred = True
                error_msg = event.get("content", "未知错误")
                st.error(f"服务端错误: {error_msg}")
                full_response = f"处理失败: {error_msg}"
                thinking_placeholder.empty()
                response_placeholder.markdown(full_response)
                return None

    except GeneratorExit:
        pass
    except Exception as e:
        error_occurred = True
        st.error(f"请求处理异常: {str(e)}")
        full_response = f"请求处理异常: {str(e)}"
        thinking_placeholder.empty()
        response_placeholder.markdown(full_response)
        return None

    processing_time = time.time() - start_time

    if sources:
        with st.expander("📄 查看来源"):
            for i, source in enumerate(sources, 1):
                source_name = source.get("source", f"来源 {i}")
                score = source.get("score")
                content = source.get("content", "")

                if not isinstance(content, str):
                    content = str(content) if content else ""

                st.markdown(f"**{source_name}** {f'(相似度: {score:.2f})' if score else ''}")
                st.text(content[:300] + "..." if len(content) > 300 else content)
                st.markdown("---")

    if metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("忠实度", f"{metrics.get('faithfulness', 0):.2f}")
        with col2:
            st.metric("相关性", f"{metrics.get('answer_relevancy', 0):.2f}")
        with col3:
            st.metric("反思次数", reflection_count)

    st.caption(f"⏱️ 处理耗时: {processing_time:.2f}秒")

    return {
        "answer": full_response,
        "sources": sources,
        "metrics": metrics,
        "reflection_count": reflection_count,
        "processing_time": processing_time,
        "think_content": think_content if has_think else []
    }


def main():
    """主界面"""
    st.set_page_config(
        page_title="Agentic RAG 智能助手",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Agentic RAG 智能知识库助手")
    st.markdown("提供基于知识库的智能问答服务，支持工具调用和反思机制")
    st.markdown("---")
    
    init_session_state()
    
    config = render_sidebar()
    
    render_chat_history()
    
    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        result = handle_user_input(prompt, config)

        if result:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "metrics": result.get("metrics", {}),
                "reflection_count": result.get("reflection_count", 0),
                "processing_time": result.get("processing_time", 0),
                "think_content": result.get("think_content", []),
                "timestamp": datetime.now().isoformat()
            })


if __name__ == "__main__":
    main()
