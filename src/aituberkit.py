import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from aituberkit_analyses import (
    ConversationAnalyses,
    ConversationAnalysisQuery,
    DialoguePatternType,
    DissatisfiedConversationAnalysis,
    UnknownQuestionAnalysis,
)
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


class DetailedAnalysisResult(BaseModel):
    failed_responses: List[str] = Field(
        description="AIが適切に回答できなかった会話のリスト"
    )
    poor_reactions: List[str] = Field(
        description="ユーザーの反応が良くなかった会話のリスト"
    )
    conversation_quality: List[str] = Field(description="会話の質に関する分析結果")


class ConversationLengthMetrics(BaseModel):
    total_sessions: int = Field(default=0)
    total_messages: int = Field(default=0)
    distribution: Dict[str, int] = Field(
        default_factory=lambda: {
            "1-3_turns": 0,
            "4-7_turns": 0,
            "8-10_turns": 0,
            "11-15_turns": 0,
            "over_15_turns": 0,
        }
    )


class AnalysisState(BaseModel):
    # モデルの設定
    model_config = {"arbitrary_types_allowed": True}

    # データ関連
    messages: List[Message] = Field(default_factory=list)
    sessions: List[Session] = Field(default_factory=list)

    # 分析結果
    conversation_analyses: ConversationAnalyses = Field(
        default_factory=ConversationAnalyses
    )
    unknown_question_analysis: UnknownQuestionAnalysis = Field(
        default_factory=UnknownQuestionAnalysis
    )
    dissatisfied_conversation_analysis: DissatisfiedConversationAnalysis = Field(
        default_factory=DissatisfiedConversationAnalysis
    )
    conversation_length_metrics: ConversationLengthMetrics = Field(
        default_factory=ConversationLengthMetrics
    )

    target_date: date | None = Field(default=None)


def parse_datetime(dt_str: str) -> datetime:
    """様々な形式の日時文字列を安全にパースする"""
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        # マイクロ秒が6桁になるようにパディング
        if "." in dt_str:
            main_part, ms_part = dt_str.split(".")
            ms_timezone = ms_part.split("+")
            if len(ms_timezone) > 1:
                ms = ms_timezone[0].ljust(6, "0")
                return datetime.fromisoformat(f"{main_part}.{ms}+{ms_timezone[1]}")
            ms_timezone = ms_part.split("-")
            if len(ms_timezone) > 1:
                ms = ms_timezone[0].ljust(6, "0")
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

    return {"messages": state.messages}


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
            session_id=session_id,
            messages=sorted(msgs, key=lambda x: x.created_at),
        )
        for session_id, msgs in sessions_dict.items()
    ]

    return {
        "sessions": state.sessions,
        "conversation_length_metrics": calculate_conversation_metrics(state),
    }


def calculate_conversation_stats(
    conversations: List[List[Message]],
) -> Dict[str, Any]:
    """定量的なメトリクスを計算"""
    user_chars: List[int] = []
    ai_chars: List[int] = []
    user_questions = ai_questions = 0

    for messages in conversations:
        user_chars.extend(len(msg.content) for msg in messages if msg.role == "user")
        ai_chars.extend(len(msg.content) for msg in messages if msg.role == "assistant")

        # 質問文をカウント
        user_questions += sum(
            1
            for msg in messages
            if msg.role == "user" and ("?" in msg.content or "？" in msg.content)
        )
        ai_questions += sum(
            1
            for msg in messages
            if msg.role == "assistant" and ("?" in msg.content or "？" in msg.content)
        )

    return {
        "user_characters": int(sum(user_chars) / len(user_chars)) if user_chars else 0,
        "ai_characters": int(sum(ai_chars) / len(ai_chars)) if ai_chars else 0,
        "user_ai_question_ratio": {
            "user_questions": user_questions,
            "ai_questions": ai_questions,
        },
    }


def update_conversation_analysis(analysis, metrics, ai_analysis):
    """会話分析結果を更新"""
    analysis.user_characters = metrics["user_characters"]
    analysis.ai_characters = metrics["ai_characters"]
    analysis.user_ai_question_ratio = metrics["user_ai_question_ratio"]
    analysis.topics.extend(ai_analysis.topics)

    # QuestionTypeDistributionの各フィールドを更新
    qt = analysis.question_types
    ai_qt = ai_analysis.question_types

    qt.yes_no_questions += ai_qt.yes_no_questions
    qt.explanation_questions += ai_qt.explanation_questions
    qt.confirmation_questions += ai_qt.confirmation_questions
    qt.how_to_questions += ai_qt.how_to_questions
    qt.opinion_questions += ai_qt.opinion_questions
    qt.choice_questions += ai_qt.choice_questions
    qt.example_questions += ai_qt.example_questions
    qt.experience_questions += ai_qt.experience_questions
    qt.meta_questions += ai_qt.meta_questions

    # DialoguePatternDistributionの更新
    pattern = ai_analysis.dialogue_pattern
    dp = analysis.dialogue_patterns

    if pattern == DialoguePatternType.SIMPLE_RESOLUTION:
        dp.simple_resolution += 1.0
    elif pattern == DialoguePatternType.DEEP_DIVE:
        dp.deep_dive += 1.0
    elif pattern == DialoguePatternType.CORRECTION:
        dp.correction += 1.0
    elif pattern == DialoguePatternType.CASUAL_EXPANSION:
        dp.casual_expansion += 1.0
    elif pattern == DialoguePatternType.MULTI_TOPIC:
        dp.multi_topic += 1.0
    elif pattern == DialoguePatternType.PROBLEM_SOLVING:
        dp.problem_solving += 1.0


def analyze_conversations(state: AnalysisState) -> Dict[str, Any]:
    """長い会話・短い会話を分析するノード"""
    logger.info("会話パターンの分析を開始します...")

    threshold = state.conversation_analyses.conversation_length_threshold
    analyses = ConversationAnalyses(conversation_length_threshold=threshold)

    # 会話を長さで分類
    long_conversations = []
    short_conversations = []
    for session in state.sessions:
        if len(session.messages) >= threshold:
            long_conversations.append(session.messages)
        else:
            short_conversations.append(session.messages)

    # メトリクスの計算
    long_metrics = calculate_conversation_stats(long_conversations)
    short_metrics = calculate_conversation_stats(short_conversations)

    # AI分析の準備
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = hub.pull("tegnike/aituberkit_conversation_analysis")
    chain = prompt | llm.with_structured_output(ConversationAnalysisQuery)

    # 各セッションの分析
    for session in state.sessions:
        conversation = f"セッションID: {session.session_id}\n" + "\n".join(
            f"{msg.role}: {msg.content}" for msg in session.messages
        )
        try:
            ai_analysis = chain.invoke({"conversation": conversation})
            # dialogue_patternが空の場合はデフォルト値を設定
            if not ai_analysis.dialogue_pattern:
                ai_analysis.dialogue_pattern = DialoguePatternType.SIMPLE_RESOLUTION
                logger.warning(
                    "dialogue_patternが空でした。デフォルト値を設定します: %s",
                    ai_analysis.dialogue_pattern,
                )
        except Exception as e:
            logger.error("会話分析中にエラーが発生: %s", e)
            # エラー時はデフォルト値で分析結果を作成
            ai_analysis = ConversationAnalysisQuery(
                topics=[],
                dialogue_pattern=DialoguePatternType.SIMPLE_RESOLUTION,
            )

        # 長さに応じて適切な分析結果に追加
        target_analysis = (
            analyses.long_conversations
            if len(session.messages) >= threshold
            else analyses.short_conversations
        )
        update_conversation_analysis(
            target_analysis,
            long_metrics if len(session.messages) >= threshold else short_metrics,
            ai_analysis,
        )

    return {"conversation_analyses": analyses}


def analyze_unknown_questions(state: AnalysisState) -> Dict[str, Any]:
    """わからないと答えた質問を分析するノード"""
    logger.info("未回答質問の分析を開始します...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = hub.pull("tegnike/aituberkit_unknown_questions")
    chain = prompt | llm.with_structured_output(UnknownQuestionAnalysis)
    result = UnknownQuestionAnalysis()

    for session in state.sessions:
        conversation = f"セッションID: {session.session_id}\n" + "\n".join(
            f"{msg.role}: {msg.content}" for msg in session.messages
        )
        analysis = chain.invoke({"conversation": conversation})

        # 質問内容の追加
        result.questions.extend(analysis.questions)

        # 理由の集計
        result.unknown_reasons.knowledge_limit += (
            analysis.unknown_reasons.knowledge_limit
        )
        result.unknown_reasons.context_misunderstanding += (
            analysis.unknown_reasons.context_misunderstanding
        )
        result.unknown_reasons.technical_limitation += (
            analysis.unknown_reasons.technical_limitation
        )
        result.unknown_reasons.data_insufficiency += (
            analysis.unknown_reasons.data_insufficiency
        )
        result.unknown_reasons.ethical_restriction += (
            analysis.unknown_reasons.ethical_restriction
        )

        # レスポンスパターンの集計
        result.response_patterns.complete_unknown += (
            analysis.response_patterns.complete_unknown
        )
        result.response_patterns.alternative_suggestion += (
            analysis.response_patterns.alternative_suggestion
        )
        result.response_patterns.additional_info_request += (
            analysis.response_patterns.additional_info_request
        )
        result.response_patterns.partial_answer += (
            analysis.response_patterns.partial_answer
        )

    return {"unknown_question_analysis": result}


def analyze_dissatisfied_conversations(state: AnalysisState) -> Dict[str, Any]:
    """未解決のまま終わった会話を分析するノード"""
    logger.info("未解決会話の分析を開始します...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = hub.pull("tegnike/aituberkit_dissatisfied_conversations")
    chain = prompt | llm.with_structured_output(DissatisfiedConversationAnalysis)
    result = DissatisfiedConversationAnalysis()

    for session in state.sessions:
        conversation = f"セッションID: {session.session_id}\n" + "\n".join(
            f"{msg.role}: {msg.content}" for msg in session.messages
        )
        analysis = chain.invoke({"conversation": conversation})

        # 会話ペアの追加
        result.conversation_pairs.extend(analysis.conversation_pairs)

        # 不満タイプの集計
        result.dissatisfaction_types.response_length_issue += (
            analysis.dissatisfaction_types.response_length_issue
        )
        result.dissatisfaction_types.complexity_issue += (
            analysis.dissatisfaction_types.complexity_issue
        )
        result.dissatisfaction_types.intent_mismatch += (
            analysis.dissatisfaction_types.intent_mismatch
        )
        result.dissatisfaction_types.ambiguous_answer += (
            analysis.dissatisfaction_types.ambiguous_answer
        )
        result.dissatisfaction_types.impractical_answer += (
            analysis.dissatisfaction_types.impractical_answer
        )

        # ユーザーパターンの集計
        result.user_patterns.explicit_complaint += (
            analysis.user_patterns.explicit_complaint
        )
        result.user_patterns.question_rephrasing += (
            analysis.user_patterns.question_rephrasing
        )
        result.user_patterns.abrupt_termination += (
            analysis.user_patterns.abrupt_termination
        )
        result.user_patterns.negative_short_response += (
            analysis.user_patterns.negative_short_response
        )
        result.user_patterns.passive_reaction += analysis.user_patterns.passive_reaction

        # 改善可能性の集計
        result.improvement_possibilities.length_adjustment += (
            analysis.improvement_possibilities.length_adjustment
        )
        result.improvement_possibilities.detail_adjustment += (
            analysis.improvement_possibilities.detail_adjustment
        )
        result.improvement_possibilities.intent_confirmation += (
            analysis.improvement_possibilities.intent_confirmation
        )
        result.improvement_possibilities.example_addition += (
            analysis.improvement_possibilities.example_addition
        )
        result.improvement_possibilities.system_enhancement += (
            analysis.improvement_possibilities.system_enhancement
        )

    return {"dissatisfied_conversation_analysis": result}


def calculate_conversation_metrics(state: AnalysisState) -> Dict[str, Any]:
    """会話の長さに関するメトリクスを計算"""
    logger.info("会話長メトリクスの計算を開始します...")

    metrics = ConversationLengthMetrics(
        total_sessions=len(state.sessions), total_messages=len(state.messages)
    )

    for session in state.sessions:
        turns = len(session.messages)

        if turns <= 3:
            metrics.distribution["1-3_turns"] += 1
        elif turns <= 7:
            metrics.distribution["4-7_turns"] += 1
        elif turns <= 10:
            metrics.distribution["8-10_turns"] += 1
        elif turns <= 15:
            metrics.distribution["11-15_turns"] += 1
        else:
            metrics.distribution["over_15_turns"] += 1

    return {"conversation_length_metrics": metrics}


def save_analysis_result(state: AnalysisState) -> None:
    """分析結果をsummariesテーブルに保存"""
    db = SupabaseAdapter()

    # target_dateをISO形式に変換
    iso_target_date = state.target_date.isoformat() if state.target_date else None

    # 同じtarget_dateのレコードを検索
    existing_record = None
    if iso_target_date:
        existing_record = db.get_record_by_condition(
            "summaries", "target_date", iso_target_date
        )

    metrics = state.conversation_length_metrics
    analyses = state.conversation_analyses
    unknown = state.unknown_question_analysis
    dissatisfied = state.dissatisfied_conversation_analysis

    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_date": iso_target_date,
        "tweet": None,
        "public_messages_test": {
            "conversation_length_metrics": {
                "total_sessions": metrics.total_sessions,
                "total_messages": metrics.total_messages,
                "distribution": metrics.distribution,
            },
            "conversation_analyses": analyses.model_dump(),
            "unknown_question_analysis": unknown.model_dump(),
            "dissatisfied_conversation_analysis": dissatisfied.model_dump(),
            "target_date": iso_target_date,
        },
        "version": "2",
    }

    if existing_record:
        db.update_record("summaries", existing_record["id"], data)
    else:
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
        self.workflow.add_node("calculate_metrics", calculate_conversation_metrics)

        # 分析ノードを個別に追加
        self.workflow.add_node("analyze_conversations", analyze_conversations)
        self.workflow.add_node("analyze_unknown_questions", analyze_unknown_questions)
        self.workflow.add_node(
            "analyze_dissatisfied_conversations", analyze_dissatisfied_conversations
        )

        # エントリーポイントの設定
        self.workflow.set_entry_point("fetch_data")

        # エッジの追加
        self.workflow.add_edge("fetch_data", "organize_sessions")
        self.workflow.add_edge("organize_sessions", "calculate_metrics")

        # 並列実行のためのエッジを追加
        self.workflow.add_edge("calculate_metrics", "analyze_conversations")
        self.workflow.add_edge("calculate_metrics", "analyze_unknown_questions")
        self.workflow.add_edge(
            "calculate_metrics", "analyze_dissatisfied_conversations"
        )

        # 並列処理の結果を終了ポイントに接続
        self.workflow.add_edge(
            [
                "analyze_conversations",
                "analyze_unknown_questions",
                "analyze_dissatisfied_conversations",
            ],
            END,
        )

    def run(self) -> Dict[str, Any]:
        """分析を実行し、結果を返す"""
        app = self.workflow.compile()
        initial_state = AnalysisState(target_date=self.target_date)
        final_state_dict = app.invoke(initial_state)
        final_state = AnalysisState(**final_state_dict)

        # 分析結果を保存
        save_analysis_result(final_state)

        metrics = final_state.conversation_length_metrics
        analyses = final_state.conversation_analyses
        unknown = final_state.unknown_question_analysis
        dissatisfied = final_state.dissatisfied_conversation_analysis

        return {
            "conversation_length_metrics": metrics.model_dump(),
            "conversation_analyses": analyses.model_dump(),
            "unknown_question_analysis": unknown.model_dump(),
            "dissatisfied_conversation_analysis": dissatisfied.model_dump(),
        }


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
