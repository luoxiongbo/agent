import streamlit as st
from uuid import uuid4
import os

from src.core import Agent
from src.memory import MemoryManager
from src.store import MemoryStore
from src.tools import add, count_letter_in_string, compare, get_current_datetime, search_wikipedia
from src.vector_store import VectorStore
from openai import OpenAI

# --- 页面配置 ---
st.set_page_config(
    page_title="Tiny Agent Demo",  # 页面标题
    page_icon="🤖",  # 页面图标
    layout="centered",  # 页面布局
    initial_sidebar_state="auto",  # 侧边栏初始状态
)

# --- OpenAI客户端初始化 ---
# DeepSeek 官方 API（默认配置）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your deepseek api key"),
    base_url="https://api.deepseek.com",
)

# 旧版 SiliconFlow 配置，保留作教学对照
# client = OpenAI(
#     api_key="your siliconflow api key",
#     base_url="https://api.siliconflow.cn/v1",
# )

# --- 记忆系统初始化 ---
@st.cache_resource
def load_memory_manager():
    """创建并缓存记忆管理器。"""
    store = MemoryStore(db_path="data/memory.db")
    vector_store = VectorStore(persist_directory="data/chroma")
    return MemoryManager(
        client=client,
        store=store,
        vector_store=vector_store,
    )

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid4())

if "agent" not in st.session_state:
    st.session_state.agent = Agent(
        client=client,
        model="deepseek-v4-flash",
        tools=[get_current_datetime, add, compare, count_letter_in_string, search_wikipedia],
        memory_manager=load_memory_manager(),
        user_id=st.session_state.user_id,
    )

agent = st.session_state.agent

# --- UI组件 ---
st.title("🤖 Happy-LLM Tiny Agent")  # 设置页面标题
st.markdown("""欢迎来到 Tiny Agent web 界面！

在下方输入您的提示，查看 Agent 的实际操作。
这个版本已经接入了长期记忆能力。
""")  # 显示Markdown格式的欢迎信息

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 在应用重新运行时显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 响应用户输入
if prompt := st.chat_input("我能为您做些什么？"):
    # 在聊天消息容器中显示用户消息
    st.chat_message("user").markdown(prompt)
    # 将用户消息添加到聊天记录中
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner('思考中...'):
        response = agent.get_completion(prompt)  # 获取Agent的响应
        
    # 在聊天消息容器中显示助手响应
    with st.chat_message("assistant"):
        st.markdown(response)
    # 将助手响应添加到聊天记录中
    st.session_state.messages.append({"role": "assistant", "content": response})
