import os
from langchain import OpenAI, PromptTemplate, LLMChain
from dotenv import load_dotenv

load_dotenv() 

def main():
    # 環境変数からキーを読み込む例
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("環境変数 OPENAI_API_KEY が設定されていません。")
    
    # 1. LLM の初期化（環境変数 OPENAI_API_KEY 必須）
    llm = OpenAI(temperature=0.5)

    # 2. プロンプトテンプレートを定義
    template = """
    あなたはとても賢いアシスタントです。
    次の質問に日本語で答えてください。

    質問: {question}
    回答:
    """
    prompt = PromptTemplate(input_variables=["question"], template=template)

    # 3. チェーンを組み立て
    chain = LLMChain(llm=llm, prompt=prompt)

    # 4. ユーザー入力を受け取って実行
    user_q = input("質問をどうぞ > ")
    answer = chain.run({"question": user_q})
    print("\n🧠 回答:\n", answer.strip())

if __name__ == "__main__":
    main()
