import os
import PyPDF2  # 新增：用于读取PDF
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# 新增：读取 PDF 文字的函数
def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def chat_with_agent():
    # 1. 尝试读取资料
    print("📚 正在加载本地资料...")
    try:
        pdf_content = read_pdf("西游记.pdf") # 确保你的文件名叫 data.pdf
        knowledge_context = f"以下是参考资料：\n{pdf_content[:2000]}" # 先取前2000字，防止太长报错
    except:
        knowledge_context = "暂无本地参考资料。"
        print("⚠️ 未找到 西游记.pdf，将进入普通对话模式。")

    # 2. 修改 System Prompt，加入资料背景
    messages = [
        {
            "role": "system", 
            "content": f"你是一个全能助手，主人是‘从芯’。{knowledge_context}\n请优先根据提供的参考资料回答问题。如果资料里没写，再根据你的常识回答。"
        }
    ]

    print("🤖 Bao-Agent (增强版) 已上线！")

    while True:
        user_input = input("👤 从芯: ")
        if user_input.lower() in ['quit', 'exit']: break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages
            )
            reply = response.choices[0].message.content
            print(f"🤖 Agent: {reply}")
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"❌ 出错了: {e}")

if __name__ == "__main__":
    chat_with_agent()