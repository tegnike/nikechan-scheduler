import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from supabase_adapter import SupabaseAdapter

load_dotenv()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """あなたはAITuberのパフォーマンスを分析する専門家です。
与えられた会話履歴を分析し、以下の2点を特定し、それぞれ文字列のリストで返してください：

1. AIが適切に回答できなかった会話
2. AIは回答したものの、ユーザーの反応が良くなかった会話

回答例：
- 「〇〇」という単語を知りませんでした。
- 最近のニュースを回答できませんでした。
- 〇〇という質問に〇〇と返しましたが、ユーザの反応が良くありませんでした。

注意点：
- 客観的な事実に基づいて判断してください。
- ユーザーの反応から判断できる範囲で評価してください。
- 建設的なフィードバックを心がけてください。
- 日本語の以外の言語で対話されている場合は、オリジナルの言語の次に（訳: 〇〇）という形式にしてください。例：「你會說中五嗎（訳: 中国語が話せますか？）」
"""


@dataclass
class Message:
    session_id: str
    role: str
    content: str
    created_at: datetime


@dataclass
class Session:
    session_id: str
    messages: List[Message]


class AnalysisResult(BaseModel):
    failed_responses: List[str] = Field(
        description="AIが適切に回答できなかった会話のリスト"
    )
    poor_reactions: List[str] = Field(
        description="ユーザーの反応が良くなかった会話のリスト"
    )


class AnalysisState(BaseModel):
    # モデルの設定
    model_config = {"arbitrary_types_allowed": True}

    # データ関連
    messages: List[Message] = Field(default_factory=list)
    sessions: List[Session] = Field(default_factory=list)
    current_batch: List[Session] = Field(default_factory=list)

    # 分析結果
    session_count: int = Field(default=0)
    message_count: int = Field(default=0)
    analysis_results: List[Dict[str, Any]] = Field(default_factory=list)

    # 制御フラグ
    is_analysis_complete: bool = Field(default=False)


def fetch_data_node(state: AnalysisState) -> Dict[str, Any]:
    """Supabaseからデータを取得するノード"""
    logger.info("データ取得を開始します...")

    db = SupabaseAdapter()
    messages = db.get_records_by_date_range("public_messages", days=1)

    state.messages = [
        Message(
            session_id=msg["session_id"],
            role=msg["role"],
            content=msg["content"],
            created_at=datetime.fromisoformat(msg["created_at"]),
        )
        for msg in messages
    ]

    state.message_count = len(state.messages)
    return {
        "messages": state.messages,
        "message_count": state.message_count,
    }


def organize_sessions_node(state: AnalysisState) -> Dict[str, Any]:
    """メッセージをセッション単位に整理するノード"""
    logger.info("セッションの整理を開始します...")

    sessions_dict: Dict[str, List[Message]] = {}
    for msg in state.messages:
        if msg.session_id not in sessions_dict:
            sessions_dict[msg.session_id] = []
        sessions_dict[msg.session_id].append(msg)

    state.sessions = [
        Session(
            session_id=session_id, messages=sorted(msgs, key=lambda x: x.created_at)
        )
        for session_id, msgs in sessions_dict.items()
    ]

    state.session_count = len(state.sessions)
    return {
        "sessions": state.sessions,
        "session_count": state.session_count,
        "message_count": state.message_count,
    }


def analyze_batch_node(state: AnalysisState) -> Dict[str, Any]:
    """15セッションずつ分析するノード"""
    logger.info(f"バッチ分析を開始します... (バッチ {len(state.analysis_results) + 1})")

    if not state.sessions:
        state.is_analysis_complete = True
        logger.info("全セッションの分析が完了しました")
        logger.info(f"分析結果: {state.analysis_results}")

        # ここで、バッチ毎に蓄積された failed_responses および poor_reactions を結合
        all_failed_responses = []
        all_poor_reactions = []
        for r in state.analysis_results:
            all_failed_responses.extend(r["failed_responses"])
            all_poor_reactions.extend(r["poor_reactions"])

        return {
            "is_analysis_complete": True,
            # まとめた配列のみ返す
            "failed_responses": all_failed_responses,
            "poor_reactions": all_poor_reactions,
            "session_count": state.session_count,
            "message_count": state.message_count,
        }

    # 最大15セッションを取得し、残りのセッションを更新
    batch_size = 15
    state.current_batch = state.sessions[:batch_size]
    state.sessions = state.sessions[batch_size:]  # この更新が反映されるように

    logger.info(f"残りセッション数: {len(state.sessions)}")

    # セッションの会話履歴を整形
    conversations = []
    for session in state.current_batch:
        conversation = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in session.messages]
        )
        conversations.append(f"セッションID: {session.session_id}\n{conversation}\n")

    # LLMで分析
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "user",
                "以下の会話を分析してください:\n\n" + "\n---\n".join(conversations),
            ),
        ]
    )

    chain = prompt | llm.with_structured_output(AnalysisResult)
    result = chain.invoke({})

    state.analysis_results.append(dict(result))
    return {
        "analysis_results": state.analysis_results,
        "sessions": state.sessions,
        "session_count": state.session_count,
        "message_count": state.message_count,
    }


class AITuberAnalyzer:
    def __init__(self):
        self.workflow = StateGraph(AnalysisState)
        self._build_graph()

    def _build_graph(self):
        # ノードの追加
        self.workflow.add_node("fetch_data", fetch_data_node)
        self.workflow.add_node("organize_sessions", organize_sessions_node)
        self.workflow.add_node("analyze_batch", analyze_batch_node)

        # エントリーポイントの設定
        self.workflow.set_entry_point("fetch_data")

        # エッジの追加
        self.workflow.add_edge("fetch_data", "organize_sessions")
        self.workflow.add_edge("organize_sessions", "analyze_batch")

        # analyze_batchノードの条件分岐を修正
        self.workflow.add_conditional_edges(
            "analyze_batch",
            lambda x: END if x.is_analysis_complete else "analyze_batch",
            {
                "analyze_batch": "analyze_batch",
                END: END,
            },
        )

    def run(self) -> Dict[str, Any]:
        """分析を実行し、結果を返す"""
        app = self.workflow.compile()
        initial_state = AnalysisState()
        final_state = app.invoke(initial_state)

        return {
            "session_count": final_state["session_count"],
            "message_count": final_state["message_count"],
            "failed_responses": [
                response
                for result in final_state["analysis_results"]
                for response in result["failed_responses"]
            ],
            "poor_reactions": [
                reaction
                for result in final_state["analysis_results"]
                for reaction in result["poor_reactions"]
            ],
        }


def main():
    analyzer = AITuberAnalyzer()
    result = analyzer.run()
    logger.info(f"Analysis Report:\n{result}")


if __name__ == "__main__":
    main()
