from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field


class QuestionTypeDistribution(BaseModel):
    yes_no_questions: int = 0  # "できますか？" "〜ですか？"
    explanation_questions: int = 0  # "なぜ？" "どうして？"
    confirmation_questions: int = 0  # "〜で合っていますか？"
    how_to_questions: int = 0  # "どうやって？" "どのように？"
    opinion_questions: int = 0  # "どう思いますか？"
    choice_questions: int = 0  # "AとBどちら？"
    example_questions: int = 0  # "例えば？" "具体的には？"
    experience_questions: int = 0  # "〜したことは？"
    meta_questions: int = 0  # AIの機能や制限について


class DialoguePatternDistribution(BaseModel):
    simple_resolution: int = 0  # 質問→回答→了解
    deep_dive: int = 0  # 質問→回答→更なる質問→詳細説明
    correction: int = 0  # 質問→回答→修正依頼→修正回答
    casual_expansion: int = 0  # 質問→回答→感想/経験共有→会話継続
    multi_topic: int = 0  # 複数の異なる話題への展開
    problem_solving: int = 0  # 問題報告→確認→詳細→解決策→実行


class DialoguePatternType(str, Enum):
    SIMPLE_RESOLUTION = "simple_resolution"  # 質問→回答→了解
    DEEP_DIVE = "deep_dive"  # 質問→回答→更なる質問→詳細説明
    CORRECTION = "correction"  # 質問→回答→修正依頼→修正回答
    CASUAL_EXPANSION = "casual_expansion"  # 質問→回答→感想/経験共有→会話継続
    MULTI_TOPIC = "multi_topic"  # 複数の異なる話題への展開
    PROBLEM_SOLVING = "problem_solving"  # 問題報告→確認→詳細→解決策→実行


class ConversationAnalysisQuery(BaseModel):
    topics: List[str] = Field(default_factory=list)
    question_types: QuestionTypeDistribution = Field(
        default_factory=QuestionTypeDistribution
    )
    dialogue_pattern: DialoguePatternType = Field(
        default=DialoguePatternType.SIMPLE_RESOLUTION
    )


class ConversationAnalysis(BaseModel):
    user_characters: int = 0  # ユーザーの文字数の平均
    ai_characters: int = 0  # AIの文字数の平均
    user_ai_question_ratio: Dict[str, int] = Field(
        default_factory=lambda: {"user_questions": 0, "ai_questions": 0}
    )
    topics: List[str] = Field(default_factory=list)
    question_types: QuestionTypeDistribution = Field(
        default_factory=QuestionTypeDistribution
    )
    dialogue_patterns: DialoguePatternDistribution = Field(
        default_factory=DialoguePatternDistribution
    )


class ConversationAnalyses(BaseModel):
    long_conversations: ConversationAnalysis = Field(
        default_factory=ConversationAnalysis,
        description="7ターン以上の長い会話の分析結果",
    )
    short_conversations: ConversationAnalysis = Field(
        default_factory=ConversationAnalysis,
        description="7ターン未満の短い会話の分析結果",
    )
    conversation_length_threshold: int = Field(
        default=7,
        description="長い会話と判定する基準ターン数",
    )


class UnknownReasonDistribution(BaseModel):
    knowledge_limit: int = 0  # 知識の範囲外（時事問題、最新情報など）
    context_misunderstanding: int = 0  # 文脈理解の失敗（指示語不明確、曖昧な質問）
    technical_limitation: int = 0  # 技術的制限（画像生成、外部API要求）
    data_insufficiency: int = 0  # データの不足（必要な情報が不十分）
    ethical_restriction: int = 0  # 倫理的な制限（不適切なリクエスト）


class UnknownResponsePattern(BaseModel):
    complete_unknown: int = 0  # 完全な「わかりません」
    alternative_suggestion: int = 0  # 代替案の提示を含む返答
    additional_info_request: int = 0  # 追加情報を求める返答
    partial_answer: int = 0  # 部分的な回答と制限の説明


class UnknownQuestionAnalysis(BaseModel):
    questions: List[str] = Field(default_factory=list)  # 質問の内容
    unknown_reasons: UnknownReasonDistribution = Field(
        default_factory=UnknownReasonDistribution
    )
    response_patterns: UnknownResponsePattern = Field(
        default_factory=UnknownResponsePattern
    )


class DissatisfactionType(BaseModel):
    response_length_issue: int = 0  # 回答が長すぎる/短すぎる
    complexity_issue: int = 0  # 回答が複雑すぎる/簡単すぎる
    intent_mismatch: int = 0  # 質問の意図と異なる回答
    ambiguous_answer: int = 0  # 曖昧な回答
    impractical_answer: int = 0  # 実用的でない回答


class UserDissatisfactionPattern(BaseModel):
    explicit_complaint: int = 0  # 明確な不満の表明
    question_rephrasing: int = 0  # 質問の言い換え/再質問
    abrupt_termination: int = 0  # 会話の突然の終了
    negative_short_response: int = 0  # 短い否定的な返答
    passive_reaction: int = 0  # その他の消極的な反応


class ImprovementPossibility(BaseModel):
    length_adjustment: int = 0.0  # 回答の長さ調整で改善可能
    detail_adjustment: int = 0  # 説明の詳しさ調整で改善可能
    intent_confirmation: int = 0  # 質問の意図確認で改善可能
    example_addition: int = 0  # 具体例追加で改善可能
    system_enhancement: int = 0  # システムの機能追加が必要


class DissatisfiedConversationAnalysis(BaseModel):
    conversation_pairs: List[str] = Field(
        default_factory=list,
        description="会話の要約のリスト。例：〇〇という質問に△△と回答したが、ユーザの反応がXXだった。",
    )
    dissatisfaction_types: DissatisfactionType = Field(
        default_factory=DissatisfactionType
    )
    user_patterns: UserDissatisfactionPattern = Field(
        default_factory=UserDissatisfactionPattern
    )
    improvement_possibilities: ImprovementPossibility = Field(
        default_factory=ImprovementPossibility
    )
