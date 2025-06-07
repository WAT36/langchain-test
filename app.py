import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableLambda
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any

# 環境変数の読み込み
load_dotenv()

class SimpleChatSystem:
    def __init__(self):
        # OpenAI APIキーの設定（ChatOpenAIを使用）
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,  # 創造性のレベル（0-1）
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # チャットプロンプトテンプレートの定義
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "あなたは親切なAIアシスタントです。質問に日本語で丁寧に答えてください。"),
            ("human", """
            質問: {question}
            
            追加情報: {context}
            
            以下の会話履歴も考慮してください:
            {chat_history}
            """)
        ])
        
        # メモリの設定（会話履歴を保持）
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # RunnableSequence（LCEL）でチェーンを作成
        self.qa_chain = self._create_chain()
    
    def _create_chain(self):
        """RunnableSequenceでチェーンを作成"""
        def format_chat_history(messages):
            """メッセージ履歴を文字列に変換"""
            if not messages:
                return "会話履歴はありません。"
            
            formatted_history = []
            for msg in messages[-6:]:  # 直近6件のメッセージのみ
                if hasattr(msg, 'type'):
                    role = "ユーザー" if msg.type == "human" else "アシスタント"
                    formatted_history.append(f"{role}: {msg.content}")
            return "\n".join(formatted_history)
        
        def get_memory_variables(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """メモリから会話履歴を取得"""
            chat_history = self.memory.chat_memory.messages
            inputs["chat_history"] = format_chat_history(chat_history)
            return inputs
        
        # RunnableSequenceの構築
        chain = (
            RunnableLambda(get_memory_variables)  # メモリから履歴を取得
            | self.prompt_template  # プロンプトテンプレートを適用
            | self.llm  # LLMで処理
            | RunnableLambda(self._save_to_memory)  # メモリに保存
        )
        
        return chain
    
    def _save_to_memory(self, ai_message):
        """AIの回答をメモリに保存"""
        # 直前の入力（質問）もメモリに保存する必要がある場合はここで処理
        return ai_message.content
    
    def ask_question(self, question: str, context: str = "特になし") -> str:
        """質問を処理して回答を返す"""
        try:
            # 人間の質問をメモリに保存
            self.memory.chat_memory.add_message(HumanMessage(content=question))
            
            # チェーンを実行
            response = self.qa_chain.invoke({
                "question": question,
                "context": context
            })
            
            # AIの回答をメモリに保存
            self.memory.chat_memory.add_message(AIMessage(content=response))
            
            return response.strip()
        except Exception as e:
            return f"エラーが発生しました: {str(e)}"
    
    def get_conversation_history(self):
        """会話履歴を取得"""
        return self.memory.chat_memory.messages

# 使用例
def main():
    # .envファイルにOPENAI_API_KEY=your_api_keyを設定してください
    if os.environ.get("OPENAI_API_KEY") is None:
        raise RuntimeError("環境変数 OPENAI_API_KEY が設定されていません。")

    qa_system = SimpleChatSystem()
    
    print("=== LangChain QAシステム ===")
    print("質問を入力してください（'quit'で終了）")
    
    while True:
        question = input("\n質問: ")
        if question.lower() == 'quit':
            break
        
        # 追加のコンテキストがあれば指定
        context = input("追加情報（なければEnter）: ")
        if not context:
            context = "特になし"
        
        # 質問を処理
        answer = qa_system.ask_question(question, context)
        print(f"\n回答: {answer}")
        
        # 会話履歴の表示（オプション）
        show_history = input("\n会話履歴を表示しますか？ (y/n): ")
        if show_history.lower() == 'y':
            history = qa_system.get_conversation_history()
            print("\n=== 会話履歴 ===")
            for i, message in enumerate(history):
                print(f"{i+1}. {message.content}")

if __name__ == "__main__":
    main()