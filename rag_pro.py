import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 配置路径 ---
# 获取当前文件所在目录，确保数据库文件夹永远跟代码在一起
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "chroma_db")
pdf_path = os.path.join(current_dir, "西游记.pdf") # 确保文件名正确

# --- 第一步：初始化 Embedding 模型 ---
# 这一步很快，因为模型已经下载到你电脑里了
print("🧠 正在加载本地语义引擎...")
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

# --- 第二步：判断并加载知识库 (秒开逻辑) ---
if os.path.exists(db_path):
    print("🚀 发现现有数据库，正在秒速加载知识库...")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
else:
    print("📚 未发现数据库，正在初次解析 PDF，请稍候...")
    if not os.path.exists(pdf_path):
        print(f"❌ 错误：找不到文件 {pdf_path}，请检查文件名！")
        exit()
        
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(data)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=db_path
    )
    print("✅ 知识库创建完成！")

# 设为检索模式
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 第三步：连接 DeepSeek 大脑 ---
llm = ChatOpenAI(
    model_name="deepseek-chat", 
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

template = """你是一个基于本地资料的助手，主人是从芯。请根据背景资料回答问题：
{context}

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 构建 LCEL 流水线
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 第四部分：启动对话 ---
print("\n✨ Bao-Agent (高性能版) 已就绪！输入 quit 退出。")
while True:
    query = input("👤 从芯: ")
    if query.lower() in ['quit', 'exit']: break
    
    try:
        response = chain.invoke(query)
        print(f"🤖 Agent: {response}")
    except Exception as e:
        print(f"❌ 运行出错: {e}")