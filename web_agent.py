import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
st.set_page_config(page_title="从芯的 AI 助理", page_icon="🤖")

st.title("🤖 Bao-Agent 网页版")

# 使用 st.spinner 让加载过程可视化，防止白屏
with st.status("🚀 正在启动 AI 引擎...", expanded=True) as status:
    st.write("🔍 正在定位数据库路径...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    
    st.write("🧠 正在唤醒本地语义模型 (这一步可能较慢)...")
    # 这一步最容易卡住，我们加上缓存
    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    
    embeddings = load_embeddings()
    
    st.write("📦 正在连接知识库...")
    if os.path.exists(db_path):
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        st.error(f"找不到数据库文件夹：{db_path}")
        st.stop()

    st.write("🔗 正在对接 DeepSeek 大脑...")
    llm = ChatOpenAI(
        model_name="deepseek-chat", 
        api_key=os.getenv("DEEPSEEK_API_KEY"), 
        base_url="https://api.deepseek.com"
    )
    
    template = "你是一个基于本地资料的助手。背景资料：{context}\n问题：{question}"
    prompt_temp = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_temp
        | llm
        | StrOutputParser()
    )
    status.update(label="✅ 系统就绪！", state="complete", expanded=False)

# --- 以下是聊天界面逻辑 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("问我关于西游记的问题吧..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chain.invoke(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})