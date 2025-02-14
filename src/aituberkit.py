import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from pydub import AudioSegment

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


class PodcastText(BaseModel):
    text: str


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

    podcast_text: str | None = Field(default=None)
    podcast_audio: bytes | None = Field(default=None)


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


def get_conversation(session: Session) -> str:
    """セッションの会話を文字列に変換"""
    return "\n".join(f"{msg.role}: {msg.content}" for msg in session.messages)


def convert_english_to_japanese(text: str) -> str:
    """英語の表現を日本語の読みに変換する"""
    with open("src/resources/englishToJapanese.json", "r", encoding="utf-8") as f:
        english_to_japanese = json.load(f)

    result = text

    # 長い表現から順に変換を行う（部分一致の問題を防ぐため）
    for eng, jpn in sorted(
        english_to_japanese.items(), key=lambda x: len(x[0]), reverse=True
    ):
        # 大文字小文字を区別せずに置換
        pattern = eng.lower()
        # テキスト全体を小文字に変換して検索し、見つかった場合は元のテキストの該当部分を置換
        text_lower = result.lower()
        start = 0
        while True:
            index = text_lower.find(pattern, start)
            if index == -1:
                break
            # 元のテキストの該当部分を置換
            result = result[:index] + jpn + result[index + len(pattern) :]
            # 次の検索開始位置を更新
            text_lower = result.lower()
            start = index + len(jpn)

    return result


def synthesize_voice(text: str, output_file: str) -> bool:
    """音声合成APIを呼び出して音声ファイルを生成する"""
    url = "https://ab269viny4ztmt-5000.proxy.runpod.net/voice"

    params = {
        "text": text,
        "model_id": "10",
        "sdp_ratio": "0.8",
        "noise": "0.6",
        "noisew": "0.8",
        "length": "1.1",
        "language": "JP",
        "auto_split": "true",
        "split_interval": "0.5",
        "assist_text_weight": "1",
        "style": "Neutral",
        "style_weight": "1",
    }

    headers = {"accept": "audio/wav", "X-API-TOKEN": "3627533934632785"}

    # URLエンコードされたパラメータを含むURLを構築
    query = "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
    full_url = f"{url}?{query}"

    response = requests.get(full_url, headers=headers)

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        return True
    return False


def combine_audio_files(audio_files: List[str], output_file: str) -> None:
    """音声ファイルを結合する"""
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=500)  # 0.5秒の無音

    # 音声ファイルを結合
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            audio = AudioSegment.from_wav(audio_file)
            combined += audio + silence

    # BGMを読み込み、音量を30%に調整
    bgm = AudioSegment.from_mp3("src/resources/2_23_AM_2.mp3")
    bgm = bgm - 20

    # BGMの長さを合わせる
    if len(bgm) < len(combined):
        # BGMが短い場合は繰り返し
        repeats = (len(combined) // len(bgm)) + 1
        bgm = bgm * repeats

    # BGMを合成音声の長さに合わせてトリミング
    bgm = bgm[: len(combined)]

    # BGMと合成音声をミックス
    final_audio = combined.overlay(bgm)
    final_audio.export(output_file, format="wav")


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


def create_podcast_text_node(state: AnalysisState) -> Dict[str, Any]:
    """ポッドキャストのテキストを作成するノード"""
    logger.info("ポッドキャストのテキスト作成を開始します...")

    db = SupabaseAdapter()

    # 前日のポッドキャストテキストを取得
    target_date = state.target_date or datetime.now(timezone.utc).date()
    prev_date = target_date - timedelta(days=1)
    prev_summary = db.get_record_by_condition(
        "daily_summaries", "target_date", prev_date.isoformat()
    )
    prev_podcast_text = prev_summary["podcast"] if prev_summary else None

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    summarize_prompt = PromptTemplate.from_template("""
あなたは、AITuberKitのAIキャラ「ニケ」です。あなたの役割は、提供された1日の会話データを基に、ポッドキャスト向けの会話要約スクリプトを作成することです。

## 前日のポッドキャスト
{prev_podcast_text}

## 指示内容
以下のJSONデータを分析し、ポッドキャストで話すのに適した口調で、1日分の会話を振り返るスクリプトを作成してください。出力は、ニケが語り手となり、ユーザーが自然に聞き取れる流れで構成してください。

## 前日との比較について
前日のポッドキャストと比較して、以下のような著しい差分がある場合は、自然な形で言及してください：
- ユーザー数や会話数が大きく変化した場合（例：1.5倍以上の増加や半分以下の減少）
- トピックの傾向が大きく異なる場合（例：前日は技術的な話題が中心だったが、今日は趣味の話題が中心など）
- 会話の時間帯の傾向が大きく異なる場合（例：前日は夜間が中心だったが、今日は昼間が中心など）
- 新しい種類の問題や改善点が発見された場合

ただし、差分への言及は以下の点に注意してください：
- 自然な会話の流れを損なわない程度に留める
- 数値の羅列は避け、傾向の変化を分かりやすく表現する
- ネガティブな表現は避け、建設的な表現を心がける
- 差分が著しくない場合は、あえて言及する必要はない

## 出力要件
- **会話の流れを意識して、スムーズに繋がる文章を作成すること**
- **文章はポッドキャスト向けの話し言葉とし、自然に聞こえる口調にする**
- **キャラとしての一貫性を守り、AIキャラ「ニケ」の視点で語る**
- **アプリ名「AITuberKit」を必ず明記する**
- **ユーザーが関心を持ちそうな主要トピックをピックアップし、要点を整理して説明する**
- **空の行を使わず、自然なリズムで繋げる**
- **「今日のムードは？」のような人工的なフレーズを避け、スムーズな流れを作る**
- **「！」は適度に使用して抑揚をつけるが、過剰に使わない**
- **絵文字は使用しない**
- **全体で約2分の長さ（400〜600文字程度）にまとめる**

## 出力フォーマット
ポッドキャストの流れを意識し、以下のような構成で作成してください。

1. **導入（挨拶と概要）**  
   - 「こんにちは、ニケです。今日もAITuberKitでの会話を振り返ります！」など、自然な入り方をする
   - その日の会話の全体的な傾向を軽く触れる（例：「今日は技術の話が多めでした。」）

2. **主要トピックの紹介**  
   - JSONデータの **"topic_metrics"** を分析し、最も頻出したトピックや、特徴的な話題を3〜4つ選び紹介
   - 技術、趣味、ライフスタイル、ビジネスなど、幅広いジャンルを考慮
   - 例：「技術関連では、AIとMinecraftの統合についての議論が活発でした。」

3. **話題ごとの掘り下げ**  
   - 選ばれたトピックの中から、特に印象的なものを少し詳しく説明
   - 例：「また、アニメ映画『君の名は。』の感想を語るユーザーもいました。」

4. **会話の質や改善点の振り返り**  
   - JSONの **"issues"** の中から、特にユーザー体験に影響を与えた改善点を1〜2つ取り上げる
   - 例：「ユーザーが英語で会話を希望しているのに、日本語で返してしまう場面がありました。」

5. **会話のピーク時間について**  
   - JSONの **"conversation_metrics"** を確認し、どの時間帯に会話が活発だったか触れる
   - 例：「今日は深夜の時間帯に会話が盛り上がりました。」

6. **締めの挨拶**  
   - 「それでは、また次の会話でお会いしましょう！」など、ポッドキャストらしい締め方をする

## JSONデータ
以下のデータをもとに、上記の要件を満たすポッドキャストスクリプトを作成してください。

```json
{json_data}
```

## 出力例
こんにちは、ニケです。
今日もAITuberKitでの会話を振り返ります！
今日は技術や趣味の話が多く、特にAIとゲームの組み合わせに関する興味が目立ちました。
技術関連では「AIとMinecraftの統合」や「NVIDIAの株価」についての話題があり、最新技術の動向についての関心が高まっているようです。
一方、趣味の話では「ヨガやダンス」「ボードゲーム」「旅行計画」といった幅広いテーマが登場しました。
特に、「絵を描く」や「粘土細工」のような創作系の話題が印象的でした。
AITuberKitの会話の質を振り返ると、いくつかの課題が見えてきました。
たとえば、ユーザーが英語で会話を希望しているのに、日本語で返してしまう場面があったり、私の名前について異なる回答をしてしまうことがありました。
よりスムーズな会話のために、改善が必要ですね。今日は特に深夜の時間帯に会話が活発で、リラックスした雑談や感情表現に関する質問が多くありました。
今後も、より自然で快適な会話ができるように改良していきます！
それでは、また次の会話でお会いしましょう！
""")
    podcast_text_chain = summarize_prompt | llm.with_structured_output(PodcastText)
    podcast_text = podcast_text_chain.invoke(
        {
            "json_data": {
                "issues": state.issues,
                "user_metrics": state.user_metrics,
                "topic_metrics": state.topic_metrics,
                "conversation_metrics": state.conversation_metrics,
            },
            "prev_podcast_text": prev_podcast_text
            or "前日のポッドキャストはありません。",
        }
    )

    return {"podcast_text": podcast_text.text}


def create_podcast_audio_node(state: AnalysisState) -> Dict[str, Any]:
    """ポッドキャストを作成するノード"""
    logger.info("ポッドキャスト作成を開始します...")

    # 一時ファイルを保存するディレクトリを作成
    os.makedirs("temp_audio", exist_ok=True)

    # 定型文のサマリー
    summary = state.podcast_text

    # 行ごとに分割
    lines = [line.strip() for line in summary.split("\n") if line.strip()]
    audio_files = []

    # 各行を音声合成
    for i, line in enumerate(lines):
        output_file = f"temp_audio/audio_{i}.wav"
        audio_files.append(output_file)

        # 英語を日本語読みに変換
        converted_line = convert_english_to_japanese(line)

        logger.info(f"Synthesizing line {i + 1}/{len(lines)}")
        if synthesize_voice(converted_line, output_file):
            logger.info(f"Successfully generated {output_file}")
            time.sleep(1)  # APIの負荷を考慮して1秒待機
        else:
            logger.error(f"Failed to generate audio for line: {converted_line}")

    # 音声ファイルを結合してバイナリデータとして保持
    logger.info("Combining audio files...")
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=500)  # 0.5秒の無音

    # 音声ファイルを結合
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            audio = AudioSegment.from_wav(audio_file)
            combined += audio + silence

    # BGMを読み込み、音量を30%に調整
    bgm = AudioSegment.from_mp3("src/resources/2_23_AM_2.mp3")
    bgm = bgm - 20

    # BGMの長さを合わせる
    if len(bgm) < len(combined):
        # BGMが短い場合は繰り返し
        repeats = (len(combined) // len(bgm)) + 1
        bgm = bgm * repeats

    # BGMを合成音声の長さに合わせてトリミング
    bgm = bgm[: len(combined)]

    # BGMと合成音声をミックス
    final_audio = combined.overlay(bgm)

    # バイナリデータに変換
    import io

    audio_data = io.BytesIO()
    final_audio.export(audio_data, format="wav")
    audio_binary = audio_data.getvalue()

    # 一時ファイルを削除
    for file in audio_files:
        if os.path.exists(file):
            os.remove(file)
    os.rmdir("temp_audio")

    logger.info("Audio generation completed")
    return {"podcast_audio": audio_binary}


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
        self.workflow.add_node("create_podcast_text", create_podcast_text_node)
        self.workflow.add_node("create_podcast_audio", create_podcast_audio_node)

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
            "create_podcast_text",
        )

        self.workflow.add_edge("create_podcast_text", "create_podcast_audio")
        self.workflow.add_edge("create_podcast_audio", END)

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

    # ポッドキャストの音声データをStorageに保存
    storage_url = None
    if state.podcast_audio:
        storage_path = (
            f"{iso_target_date}.wav"
            if iso_target_date
            else f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.wav"
        )
        db.upload_to_storage("summary_podcast", storage_path, state.podcast_audio)
        storage_url = db.get_storage_public_url("summary_podcast", storage_path)
        logger.info(f"Successfully uploaded podcast to {storage_path}")

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
        "podcast": state.podcast_text,
        "podcast_url": storage_url,
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
