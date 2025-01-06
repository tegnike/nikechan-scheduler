import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from langchain import hub
from supabase_adapter import SupabaseAdapter

load_dotenv()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

    target_date: date | None = Field(default=None)


def parse_datetime(dt_str: str) -> datetime:
    """様々な形式の日時文字列を安全にパースする"""
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        # マイクロ秒が6桁になるようにパディング
        if '.' in dt_str:
            main_part, ms_part = dt_str.split('.')
            ms_timezone = ms_part.split('+')
            if len(ms_timezone) > 1:
                ms = ms_timezone[0].ljust(6, '0')
                return datetime.fromisoformat(f"{main_part}.{ms}+{ms_timezone[1]}")
            ms_timezone = ms_part.split('-')
            if len(ms_timezone) > 1:
                ms = ms_timezone[0].ljust(6, '0')
                return datetime.fromisoformat(f"{main_part}.{ms}-{ms_timezone[1]}")
        return datetime.fromisoformat(dt_str)


def fetch_data_node(state: AnalysisState) -> Dict[str, Any]:
    """Supabaseからデータを取得するノード"""
    logger.info("データ取得を開始します...")

    db = SupabaseAdapter()

    # 指定された日付がある場合は使用し、なければ現在日付を使用
    target_date = state.target_date or datetime.now(timezone.utc).date()

    # 日本時間の00:00をUTCに変換
    jst = timezone(timedelta(hours=9))
    start_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=jst)
    end_dt = start_dt + timedelta(days=1)

    # UTCに変換
    start_utc = start_dt.astimezone(timezone.utc)
    end_utc = end_dt.astimezone(timezone.utc)

    messages = db.get_records_by_date_range(
        "public_messages", start_date=start_utc, end_date=end_utc
    )

    state.messages = [
        Message(
            session_id=msg["session_id"],
            role=msg["role"],
            content=msg["content"],
            created_at=parse_datetime(msg["created_at"]),
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
    prompt = hub.pull("tegnike/aituberkit_analysis")

    chain = prompt | llm.with_structured_output(AnalysisResult)
    result = chain.invoke({"conversations": "\n---\n".join(conversations)})

    state.analysis_results.append(dict(result))
    return {
        "analysis_results": state.analysis_results,
        "sessions": state.sessions,
        "session_count": state.session_count,
        "message_count": state.message_count,
    }


def save_analysis_result(result: Dict[str, Any], target_date: date | None) -> None:
    """分析結果をsummariesテーブルに保存"""
    db = SupabaseAdapter()

    # target_dateをISO形式に変換
    iso_target_date = target_date.isoformat() if target_date else None

    # 同じtarget_dateのレコードを検索
    existing_record = (
        db.get_record_by_condition("summaries", "target_date", iso_target_date)
        if iso_target_date
        else None
    )

    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_date": iso_target_date,
        "tweet": None,
        "public_message": {
            "session_count": result["session_count"],
            "message_count": result["message_count"],
            "failed_responses": result["failed_responses"],
            "poor_reactions": result["poor_reactions"],
            "target_date": iso_target_date,
        },
    }

    if existing_record:
        # 既存レコードがある場合は更新
        db.update_record("summaries", existing_record["id"], data)
    else:
        # 新規レコードとして挿入
        db.insert_record("summaries", data)


class AITuberAnalyzer:
    def __init__(self, target_date: str | None = None):
        self.workflow = StateGraph(AnalysisState)
        self.target_date = (
            datetime.strptime(target_date, "%Y-%m-%d").date() if target_date else None
        )
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
        initial_state = AnalysisState(target_date=self.target_date)
        final_state = app.invoke(initial_state)

        result = {
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

        # 分析結果を保存
        save_analysis_result(result, self.target_date)
        return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AITuber会話分析")
    parser.add_argument("--date", help="分析対象日 (YYYY-MM-DD形式)")

    args = parser.parse_args()
    analyzer = AITuberAnalyzer(target_date=args.date)
    result = analyzer.run()
    logger.info(f"Analysis Report:\n{result}")


if __name__ == "__main__":
    main()
