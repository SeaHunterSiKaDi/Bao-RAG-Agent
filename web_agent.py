__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

# 1. 配置页面
st.set_page_config(page_title="从芯的 AI 助理", page_icon="🤖")
st.title("🤖 Bao-Agent 网页版")

# 2. 获取 API KEY
if "DEEPSEEK_API_KEY" in st.secrets:
    deepseek_api_key = st.secrets["DEEPSEEK_API_KEY"]
else:
    load_dotenv()
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    st.error("未找到 API Key，请在 Secrets 中配置。")
    st.stop()

# 3. 初始化引擎（增加自动修复逻辑）
with st.status("🚀 正在启动 AI 引擎...", expanded=True) as status:
    # 路径准备
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    pdf_path = os.path.join(current_dir, "西游记.pdf")
    
    st.write("🧠 正在唤醒本地语义模型...")
    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    embeddings = load_embeddings()
    
    st.write("📦 正在连接知识库...")
    
    def initialize_vectorstore():
        # 如果数据库文件夹存在且不为空，尝试直接加载
        if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
            try:
                return Chroma(persist_directory=db_path, embedding_function=embeddings)
            except Exception as e:
                st.warning(f"本地数据库兼容性异常，正在重构：{e}")
                shutil.rmtree(db_path) # 报错就删了重来
        
        # 现场读取 PDF 并生成数据库
        if os.path.exists(pdf_path):
            st.write("📖 正在读取 PDF 并生成云端索引（仅限首次）...")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            return Chroma.from_documents(documents=pages, embedding=embeddings, persist_directory=db_path)
        else:
            st.error("错误：未找到 西游记.pdf 文件")
            st.stop()

    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.write("🔗 正在对接 DeepSeek 大脑...")
    llm = ChatOpenAI(
        model_name="deepseek-chat", 
        api_key=deepseek_api_key, 
        base_url="https://api.deepseek.com"
    )
    
    template = "你是一个基于本地资料的助手。请结合背景资料回答问题。\n背景资料：{context}\n问题：{question}"
    prompt_temp = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_temp
        | llm
        | StrOutputParser()
    )
    status.update(label="✅ 系统就绪！", state="complete", expanded=False)

# 4. 聊天界面逻辑
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
