import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from dotenv import load_dotenv

load_dotenv() 

def main():
    if os.environ.get("OPENAI_API_KEY") is None:
        raise RuntimeError("環境変数 OPENAI_API_KEY が設定されていません。")

    # 1) PromptTemplate を定義
    template = """
    あなたはとても賢いアシスタントです。
    次の質問に日本語で答えてください。

    質問: {question}
    回答:
    """
    prompt = PromptTemplate(input_variables=["question"], template=template)

    # 2) OpenAI LLM を初期化
    llm = OpenAI(temperature=0.5)

    # 3) RunnableSequence で「prompt → llm」をパイプでつなぐ
    sequence = RunnableSequence(prompt, llm)

    # 4) ユーザー入力を受け取って実行
    user_q = input("質問をどうぞ > ")
    answer = sequence.invoke({"question": user_q})
    print("\n🧠 回答:\n", answer.strip())

if __name__ == "__main__":
    main()