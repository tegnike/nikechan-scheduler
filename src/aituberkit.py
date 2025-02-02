import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from supabase_adapter import SupabaseAdapter

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# カテゴリー定義
ISSUES_CATEGORY = {
    "feature_limitations": "機能の制限",
    "conversation_quality": "会話の質",
    "usability": "操作性",
    "response_quality": "応答品質",
    "user_experience": "ユーザー体験",
}

TOPICS_CATEGORY = {
    "technical": "技術・開発",
    "education": "教育・学習",
    "hobby": "趣味・エンターテイメント",
    "business": "仕事・ビジネス",
    "lifestyle": "生活・健康",
    "system": "システム関連",
    "other": "その他",
}

TIME_CATEGORY = {
    "morning": "0-3時",
    "afternoon": "4-7時",
    "evening": "8-11時",
    "night": "12-15時",
    "late_night": "16-19時",
    "midnight": "20-23時",
}

TURN_CATEGORY = {
    "1-3_turns": "1-3回",
    "4-7_turns": "4-7回",
    "8-10_turns": "8-10回",
    "11-15_turns": "11-15回",
    "over_15_turns": "15回以上",
}


# データモデル定義
class Issue(BaseModel):
    category: str
    solution: str
    description: str


class IssueList(BaseModel):
    issues: List[Issue]


class Language(BaseModel):
    languages: Dict[str, int] = Field(
        default_factory=dict,
        description="言語名（日本語）をキー、出現回数を値とする辞書",
    )


class UserType(BaseModel):
    new_user: int
    repeat_user: int


class UserMetrics(BaseModel):
    languages: Language
    user_types: UserType
    repeat_rate: float
    total_users: int
    total_messages: int


class TopicCount(BaseModel):
    count: int
    topic: str


class TopicMetrics(BaseModel):
    technical: List[TopicCount] = Field(default_factory=list)
    education: List[TopicCount] = Field(default_factory=list)
    hobby: List[TopicCount] = Field(default_factory=list)
    business: List[TopicCount] = Field(default_factory=list)
    lifestyle: List[TopicCount] = Field(default_factory=list)
    system: List[TopicCount] = Field(default_factory=list)
    other: List[TopicCount] = Field(default_factory=list)


class LanguageMetrics(BaseModel):
    language: str = Field(description="会話で使用されている主要な言語")
    timezone: str = Field(description="推定されるタイムゾーン")
    offset: int = Field(description="日本からの時差（時間単位）")


class TimeDistribution(BaseModel):
    count: int
    avg_turns: float


class ConversationMetrics(BaseModel):
    time_distribution: Dict[str, TimeDistribution]
    turn_distribution: Dict[str, int]


class AnalysisResult(BaseModel):
    issues: List[Issue]
    user_metrics: UserMetrics
    topic_metrics: TopicMetrics
    conversation_metrics: ConversationMetrics


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


class AnalysisState(BaseModel):
    # モデルの設定
    model_config = {"arbitrary_types_allowed": True}

    # データ関連
    messages: List[Message] = Field(default_factory=list)
    sessions: List[Session] = Field(default_factory=list)

    # 分析結果
    issues: List[Issue] = Field(default_factory=list)
    user_metrics: Optional[UserMetrics] = None
    topic_metrics: Optional[TopicMetrics] = None
    conversation_metrics: Optional[ConversationMetrics] = None

    target_date: date | None = Field(default=None)


def parse_datetime(dt_str: str) -> datetime:
    """様々な形式の日時文字列を安全にパースする"""
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
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
        "public_messages",
        start_date=start_utc,
        end_date=end_utc,
    )

    messages = [
        Message(
            session_id=msg["session_id"],
            role=msg["role"],
            content=msg["content"],
            created_at=parse_datetime(msg["created_at"]),
        )
        for msg in messages
    ]

    return {"messages": messages}


def organize_sessions_node(state: AnalysisState) -> Dict[str, Any]:
    """メッセージをセッション単位に整理するノード"""
    logger.info("セッションの整理を開始します...")

    sessions_dict: Dict[str, List[Message]] = {}
    for msg in state.messages:
        if msg.session_id not in sessions_dict:
            sessions_dict[msg.session_id] = []
        sessions_dict[msg.session_id].append(msg)

    sessions = [
        Session(
            session_id=session_id,
            messages=sorted(msgs, key=lambda x: x.created_at),
        )
        for session_id, msgs in sessions_dict.items()
    ]

    return {"sessions": sessions}


def analyze_user_metrics_node(state: AnalysisState) -> Dict[str, Any]:
    """ユーザーメトリクスの分析を行うノード"""
    logger.info("ユーザーメトリクス分析を開始します...")

    # データベースアダプターの初期化
    db = SupabaseAdapter()

    # 言語使用の分析
    languages: Language = Language(languages={})
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # ユーザータイプの分析
    new_users = 0
    repeat_users = 0

    # 指定された日付がある場合は使用し、なければ現在日付を使用
    target_date = state.target_date or datetime.now(timezone.utc).date()

    # 日本時間の00:00をUTCに変換
    jst = timezone(timedelta(hours=9))
    target_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=jst)
    target_utc = target_dt.astimezone(timezone.utc)

    for session in state.sessions:
        # セッションIDに対応するpublic_chat_sessionsレコードを検索
        chat_session = db.get_record_by_condition(
            "public_chat_sessions", "id", session.session_id
        )

        if chat_session:
            # created_atをdatetimeオブジェクトに変換
            session_created_at = parse_datetime(chat_session["created_at"])

            # 対象日より前に作成されたセッションがあればリピートユーザー
            if session_created_at < target_utc:
                repeat_users += 1
            else:
                new_users += 1
        else:
            # セッション情報が見つからない場合は新規ユーザーとして扱う
            new_users += 1

        # 言語分析の処理（既存のコード）
        if not any(msg.role == "user" for msg in session.messages):
            continue

        conversation = get_conversation(session)
        try:
            result = llm.invoke(
                f"""以下の会話テキストを分析し、ユーザー（role: user）が使用している
                言語を判定してください。AIアシスタント（role: assistant）の発言は
                文脈の参考として使用してください。

                複数の言語が使用されている場合は、最も多く使用されている言語を
                1つ選んでください。
                回答は日本語で言語名のみを記載してください
                （例：日本語、英語、ドイツ語、フランス語など）。

                会話テキスト：
                {conversation}
                """
            )

            normalized_language = str(
                result.content[0]
                if isinstance(result.content, list)
                else result.content
            ).strip()

            languages.languages[normalized_language] = (
                languages.languages.get(normalized_language, 0) + 1
            )

        except Exception as e:
            logger.error("言語判定中にエラーが発生: %s", e)
            continue

    total_users = new_users + repeat_users
    repeat_rate = round((repeat_users / total_users * 100), 1) if total_users > 0 else 0
    total_messages = sum(len(session.messages) for session in state.sessions)

    user_metrics = {
        "languages": languages,
        "user_types": UserType(
            new_user=new_users, repeat_user=repeat_users
        ).model_dump(),
        "repeat_rate": repeat_rate,
        "total_users": total_users,
        "total_messages": total_messages,
    }

    return {"user_metrics": user_metrics}


def analyze_topics_node(state: AnalysisState) -> Dict[str, Any]:
    """トピックの分析を行うノード"""
    logger.info("トピック分析を開始します...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # prompt = hub.pull("tegnike/aituberkit_topics")
    prompt = PromptTemplate.from_template("""あなたは会話分析の専門家です。与えられた会話を分析し、トピックを適切なカテゴリーに分類してください。

以下のカテゴリーに基づいて分類してください：
- technical: 技術・開発関連（プログラミング、環境構築、技術的な質問など）
- education: 教育・学習関連（言語学習、文化、教育コンテンツなど）
- hobby: 趣味・エンターテイメント関連（趣味の話題、日常会話など）
- business: ビジネス関連（ビジネスの話題、ビジネスの質問など）
- lifestyle: 生活・健康関連（生活相談、健康、ペットなど）
- system: システム関連（この会話システム「AITuberKit」の機能の質問、バグ報告、改善要望など）
- other: その他（上記カテゴリーに分類できない話題, 挨拶、日常会話など）

各カテゴリーについて、以下の情報を提供してください：
1. topic: 具体的な話題（例：「プログラミング質問」「日本語学習」など）
2. count: 必ず1を返してください

会話テキスト:
{conversation}

応答は以下の形式で返してください：
{{
    "technical": [
        {{
            "count": 1,
            "topic": "トピック名"
        }},
        ...
    ],
    "education": [...],
    "hobby": [...],
    "business": [...],
    "lifestyle": [...],
    "system": [...],
    "other": [...]
}}

注意事項：
- 1つの会話から複数のトピックを抽出可能です
- 各カテゴリーは空配列でも構いません
- トピック名は具体的かつ簡潔にしてください
- 分類が曖昧な場合は、最も適切なカテゴリーを1つ選んでください""")

    chain = prompt | llm.with_structured_output(TopicMetrics)

    topic_metrics = TopicMetrics()
    for session in state.sessions:
        conversation = get_conversation(session)
        try:
            session_topics = chain.invoke({"conversation": conversation})
            # 各カテゴリーのトピックをマージ
            for category in TOPICS_CATEGORY.keys():
                current_topics = getattr(topic_metrics, category)
                new_topics = getattr(session_topics, category)
                for new_topic in new_topics:
                    # 既存のトピックの更新または新規追加
                    found = False
                    for existing in current_topics:
                        if existing.topic == new_topic.topic:
                            existing.count += new_topic.count
                            found = True
                            break
                    if not found:
                        current_topics.append(new_topic)
        except Exception as e:
            logger.error("トピック分析中にエラーが発生: %s", e)

    return {"topic_metrics": topic_metrics.model_dump()}


def analyze_conversation_metrics_node(
    state: AnalysisState,
) -> Dict[str, Any]:
    """会話メトリクスの分析を行うノード"""
    logger.info("会話メトリクス分析を開始します...")

    time_dist = {time: {"count": 0, "total_turns": 0} for time in TIME_CATEGORY}
    turn_dist = {turn: 0 for turn in TURN_CATEGORY}

    for session in state.sessions:
        if not session.messages:
            continue

        conversation = get_conversation(session)

        ## ここでAIを使用して会話の言語が何か、タイムゾーンはどこか、日本と比較してプラスマイナス何時間かを取得する
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = PromptTemplate.from_template("""以下の会話テキストを分析し、会話の言語、タイムゾーン、日本と比較したときの時刻がプラスマイナス何時間かを取得してください。
日本の場合はオフセットは0時間です。

会話テキスト:
{conversation}

応答は以下の形式で返してください：
{{
    "language": "言語名",
    "timezone": "タイムゾーン名",
    "offset": "プラスマイナス何時間か"
}}

例：
{{
    "language": "英語",
    "timezone": "アメリカ",
    "offset": 13
}}
""")
        chain = prompt | llm.with_structured_output(LanguageMetrics)
        language_metrics: LanguageMetrics = chain.invoke({"conversation": conversation})
        offset: int = language_metrics.offset

        # 時間帯の分析（最初のメッセージの作成時間を使用）
        hour = session.messages[0].created_at.hour + offset
        time_slot = None
        if 0 <= hour < 4:
            time_slot = "morning"
        elif 4 <= hour < 8:
            time_slot = "afternoon"
        elif 8 <= hour < 12:
            time_slot = "evening"
        elif 12 <= hour < 16:
            time_slot = "night"
        elif 16 <= hour < 20:
            time_slot = "late_night"
        else:
            time_slot = "midnight"

        turns = len(session.messages) / 2
        time_dist[time_slot]["count"] += 1
        time_dist[time_slot]["total_turns"] += turns

        # ターン数の分布
        if turns <= 3:
            turn_dist["1-3_turns"] += 1
        elif turns <= 7:
            turn_dist["4-7_turns"] += 1
        elif turns <= 10:
            turn_dist["8-10_turns"] += 1
        elif turns <= 15:
            turn_dist["11-15_turns"] += 1
        else:
            turn_dist["over_15_turns"] += 1

    # 平均ターン数の計算
    time_distribution = {}
    for time_slot, data in time_dist.items():
        if data["count"] > 0:
            avg_turns = data["total_turns"] / data["count"]
        else:
            avg_turns = 0
        time_distribution[time_slot] = TimeDistribution(
            count=data["count"],
            avg_turns=avg_turns,
        )

    conversation_metrics = ConversationMetrics(
        time_distribution=time_distribution,
        turn_distribution=turn_dist,
    )

    return {"conversation_metrics": conversation_metrics}


def analyze_issues_node(state: AnalysisState) -> Dict[str, Any]:
    """問題カテゴリーの分類と集計を行うノード"""
    logger.info("問題分析を開始します...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # prompt = hub.pull("tegnike/aituberkit_issues")
    prompt = PromptTemplate.from_template("""あなたは会話分析の専門家です。与えられた会話を分析し、AIアシスタントの応答における問題点や制限を特定してください。

以下のカテゴリーに基づいて分類してください：
- feature_limitations: 機能の制限（技術的な制限、知識の制限など）
- conversation_quality: 会話の質（説明の不十分さ、理解の不正確さなど）
- usability: 操作性（使いにくさ、インターフェースの問題など）
- response_quality: 応答品質（不適切な応答、誤った情報など）
- user_experience: ユーザー体験（文脈の維持、対話の一貫性など）

各問題について以下の情報を提供してください：
1. category: 上記のカテゴリーから最も適切なもの
2. description: 具体的にどのような問題が発生したか、やり取りを意訳して記載してください
3. solution: その問題に対する実践的な解決策や改善方法

会話テキスト:
{conversation}

応答は以下の形式のリストで返してください：
[
    {{
        "category": "カテゴリー名",
        "description": "問題の詳細な説明",
        "solution": "具体的な解決策"
    }},
    ...
]

注意事項：
- 1つの会話から複数の問題を抽出可能です
- 明確な問題が見られない場合は空のリストを返してください
- 問題の深刻度に関係なく、改善の余地がある点を具体的に指摘してください
- 解決策は実践的で具体的なものを提案してください
- ユーザの回答に日時が含まれていたり、AIの回答に感情タグ[neutral|happy|sad]が含まれてる場合がありますが、それらは正常なので無視してください
- キャラアプリという特性上、AIの返信に対して、ユーザが反応しない場合が多いです。そのため、AIの返信で終わっている事象は問題として判断しないでください""")

    chain = prompt | llm.with_structured_output(IssueList)

    issues: List[Issue] = []
    for session in state.sessions:
        conversation = get_conversation(session)
        try:
            result = chain.invoke({"conversation": conversation})
            issues.extend(result.issues)
        except Exception as e:
            logger.error("問題分析中にエラーが発生: %s", e)

    return {"issues": issues}


def summarize_issues_node(state: AnalysisState) -> Dict[str, Any]:
    """問題の要約を行うノード"""
    logger.info("問題の要約を開始します...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    summarize_prompt = PromptTemplate.from_template("""あなたは会話分析の専門家です。複数の問題報告を分析し、重要と考えられるものの選択及び類似問題の統合を行ってください。

この会話はwebアプリにおけるユーザとAIキャラとの会話です。
今回のカテゴリは{category}です。

入力された問題リストを以下の基準で整理してください：
- 類似の問題は統合する。具体的に類似度が8割と考えられた場合に限る。
- 統合した結果、複数の解決策がある場合は、最も効果的なものを選択または組み合わせる
- 説明は具体的かつ簡潔にまとめるが、抽象的にならないようにする
- まとめる必要がない課題はそのまま返す
- 各課題の重要度を考慮し、優先度順に並び替える
- 優先度順に最大5つまでを返す

問題リスト:
{issues}

以下の形式で最大5つまで返してください：
[
    {{
        "category": "{category}",
        "description": "統合された問題の説明",
        "solution": "最適化された解決策"
    }},
    ...
]

注意事項：
- 少しでも異なると考えられる問題は別々に維持してください
- 最終的なリストは元のリストより少ない項目数になるはずです（最大5つ）
- ユーザの回答に日時が含まれていたり、AIの回答に感情タグ[neutral|happy|sad]が含まれてる場合がありますが、それらは正常なので無視してください
- キャラアプリという特性上、AIの返信に対して、ユーザが反応しない場合が多いです。そのため、AIの返信で終わっている事象は問題として判断しないでください""")
    summarize_chain = summarize_prompt | llm.with_structured_output(IssueList)

    category_issues = {}
    for issue in state.issues:
        if issue.category not in category_issues:
            category_issues[issue.category] = []
        category_issues[issue.category].append(issue)

    # 問題の要約を実行
    summarized_issues: List[Issue] = []
    try:
        for category, category_specific_issues in category_issues.items():
            result = summarize_chain.invoke(
                {"category": category, "issues": category_specific_issues}
            )
            # リストを結合する
            summarized_issues.extend(result.issues[:5])
    except Exception as e:
        logger.error("問題の要約中にエラーが発生: %s", e)

    return {"issues": summarized_issues}


def get_conversation(session: Session) -> str:
    """セッションの会話を文字列に変換"""
    return "\n".join(f"{msg.role}: {msg.content}" for msg in session.messages)


class AITuberAnalyzer2:
    def __init__(self, target_date: str | None = None):
        self.workflow = StateGraph(AnalysisState)
        self.target_date = (
            datetime.strptime(target_date, "%Y-%m-%d").date() if target_date else None
        )
        self._build_graph()

    def _build_graph(self):
        # ノードの追加
        self.workflow.add_node("fetch_data", fetch_data_node)
        self.workflow.add_node(
            "organize_sessions",
            organize_sessions_node,
        )

        # 分析ノードを個別に追加
        self.workflow.add_node(
            "analyze_user_metrics",
            analyze_user_metrics_node,
        )
        self.workflow.add_node("analyze_topics", analyze_topics_node)
        self.workflow.add_node(
            "analyze_conversation_metrics",
            analyze_conversation_metrics_node,
        )
        self.workflow.add_node("analyze_issues", analyze_issues_node)
        self.workflow.add_node("summarize_issues", summarize_issues_node)

        # エントリーポイントの設定
        self.workflow.set_entry_point("fetch_data")

        # エッジの追加
        self.workflow.add_edge("fetch_data", "organize_sessions")

        # 並列実行のためのエッジを追加
        self.workflow.add_edge(
            "organize_sessions",
            "analyze_user_metrics",
        )
        self.workflow.add_edge("organize_sessions", "analyze_topics")
        self.workflow.add_edge(
            "organize_sessions",
            "analyze_conversation_metrics",
        )
        self.workflow.add_edge("organize_sessions", "analyze_issues")
        self.workflow.add_edge("analyze_issues", "summarize_issues")

        # 並列処理の結果を終了ポイントに接続
        self.workflow.add_edge(
            [
                "analyze_user_metrics",
                "analyze_topics",
                "analyze_conversation_metrics",
                "summarize_issues",
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

        return {
            "issues": final_state.issues,
            "user_metrics": final_state.user_metrics,
            "topic_metrics": final_state.topic_metrics,
            "conversation_metrics": final_state.conversation_metrics,
        }


def save_analysis_result(state: AnalysisState) -> None:
    """分析結果をdaily_summariesテーブルに保存"""
    db = SupabaseAdapter()

    # target_dateをISO形式に変換
    iso_target_date = state.target_date.isoformat() if state.target_date else None

    # 同じtarget_dateのレコードを検索
    existing_record = None
    if iso_target_date:
        existing_record = db.get_record_by_condition(
            "daily_summaries",
            "target_date",
            iso_target_date,
        )

    # ユーザーメトリクスから値を取得
    total_users = state.user_metrics.total_users if state.user_metrics else 0
    total_messages = state.user_metrics.total_messages if state.user_metrics else 0
    repeat_users = (
        state.user_metrics.user_types.repeat_user if state.user_metrics else 0
    )

    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_date": iso_target_date,
        "tweet": None,
        "public_message": {
            "issues": [issue.model_dump() for issue in state.issues],
            "user_metrics": (
                state.user_metrics.model_dump() if state.user_metrics else None
            ),
            "topic_metrics": (
                state.topic_metrics.model_dump() if state.topic_metrics else None
            ),
            "conversation_metrics": (
                state.conversation_metrics.model_dump()
                if state.conversation_metrics
                else None
            ),
            "target_date": iso_target_date,
        },
        "version": "3",
        "public_chat_session_count": total_users,
        "public_message_count": total_messages,
        "repeat_count": repeat_users,
    }

    if existing_record:
        db.update_record("daily_summaries", existing_record["id"], data)
    else:
        db.insert_record("daily_summaries", data)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AITuber会話分析")
    parser.add_argument("--date", help="分析対象日 (YYYY-MM-DD形式)")

    args = parser.parse_args()
    analyzer = AITuberAnalyzer2(target_date=args.date)
    result = analyzer.run()
    logger.info(f"Analysis Report:\n{result}")


if __name__ == "__main__":
    main()
