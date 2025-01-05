import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from twitter_adapter import TwitterAdapter
from supabase_adapter import SupabaseAdapter

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TweetSourceJudgeResult(BaseModel):
    result: str = Field(default="", description="ツイート生成元の判断結果")
    reason: str = Field(default="", description="ツイート生成元の判断理由")


class TweetGenerationResult(BaseModel):
    tweet_text: str = Field(description="生成されたツイート")


class TweetGenerationState(BaseModel):
    # モデルの設定
    model_config = {"arbitrary_types_allowed": True}

    # データ取得関連
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="対話履歴"
    )
    tweet_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="ツイート履歴"
    )
    master_tweet_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="マスターのツイート履歴"
    )

    # 生成関連
    judge_result: TweetSourceJudgeResult = Field(
        default_factory=TweetSourceJudgeResult, description="ツイート生成元の判断結果"
    )
    pre_generated_tweet: str = Field(
        default="", description="1次生成されたツイート（磨き前）"
    )
    generated_tweet: str = Field(default="", description="生成されたツイート")
    is_tweet_polished: bool = Field(
        default=False, description="ツイートが磨きをかけられたかどうか"
    )


def fetch_data_node(state: TweetGenerationState) -> Dict[str, Any]:
    """Supabaseからデータを取得するノード"""
    logger.info("データ取得を開始します...")

    db = SupabaseAdapter()

    messages = db.get_records_by_date_range("local_messages", days=1)
    conversation = []
    for msg in sorted(messages, key=lambda x: x["created_at"]):
        if "role" in msg and "content" in msg:
            conversation.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "created_at": msg["created_at"],
                }
            )

    tweet_history = []
    tweets = db.get_records_by_date_range("tweets", days=3)
    for tweet in tweets:
        tweet_history.append(
            {
                "content": tweet["content"],
                "created_at": tweet["created_at"],
            }
        )

    master_tweets = []
    master_tweet_data = db.get_records_by_date_range("my_tweets", days=1)
    for tweet in master_tweet_data:
        master_tweets.append(
            {
                "content": tweet["text"],
                "created_at": tweet["created_at"],
            }
        )

    return {
        "conversation_history": conversation,
        "tweet_history": tweet_history,
        "master_tweet_history": master_tweets,
    }


def judge_content_node(state: TweetGenerationState) -> Dict[str, Any]:
    """ツイート生成元（会話履歴かマスターのツイート）を判断するノード"""
    logger.info("ツイート生成元の判断を開始します...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt: ChatPromptTemplate = hub.pull("tegnike/tweet_source_judge")
    chain = prompt | llm.with_structured_output(TweetSourceJudgeResult)
    judge_result: TweetSourceJudgeResult = chain.invoke(
        {
            "conversation_history": state.conversation_history,
            "tweet_history": state.tweet_history,
            "master_tweet_history": state.master_tweet_history,
        },
    )
    return {"judge_result": judge_result}


def generate_tweet_node(state: TweetGenerationState) -> Dict[str, Any]:
    """ツイートを生成するノード"""
    logger.info("ツイート生成を開始します...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt: ChatPromptTemplate = hub.pull("tegnike/ai_nikechan_tweet")
    chain = prompt | llm.with_structured_output(TweetGenerationResult)
    tweet_result: TweetGenerationResult = chain.invoke(
        {
            "conversation_history": state.conversation_history,
            "tweet_history": state.tweet_history,
            "master_tweet_history": state.master_tweet_history,
            "judge_result": state.judge_result.result,
            "judge_reason": state.judge_result.reason,
        },
    )
    return {"pre_generated_tweet": tweet_result.tweet_text}


def polish_tweet_node(state: TweetGenerationState) -> Dict[str, Any]:
    """ツイートに磨きをかけるノード"""
    logger.info("ツイートの磨きをかけます...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt: ChatPromptTemplate = hub.pull("tegnike/tweet_polishing")
    chain = prompt | llm.with_structured_output(TweetGenerationResult)
    tweet_result: TweetGenerationResult = chain.invoke(
        {
            "tweet": state.pre_generated_tweet,
        },
    )
    return {"generated_tweet": tweet_result.tweet_text, "is_tweet_polished": True}


class TweetGenerator:
    def __init__(self):
        self.workflow = StateGraph(TweetGenerationState)
        self._build_graph()

    def _build_graph(self):
        # ノードの追加
        self.workflow.add_node("fetch_data", fetch_data_node)
        self.workflow.add_node("judge_content", judge_content_node)
        self.workflow.add_node("generate_tweet", generate_tweet_node)
        self.workflow.add_node("polish_tweet", polish_tweet_node)

        # エントリーポイントの設定
        self.workflow.set_entry_point("fetch_data")

        # エッジの追加
        self.workflow.add_edge("fetch_data", "judge_content")
        self.workflow.add_edge("judge_content", "generate_tweet")
        self.workflow.add_edge("generate_tweet", "polish_tweet")
        self.workflow.add_edge("polish_tweet", END)

    def run(self) -> Optional[str]:
        """グラフを実行する"""
        app = self.workflow.compile()
        initial_state = TweetGenerationState()
        final_state = app.invoke(initial_state)

        if final_state["is_tweet_polished"]:
            graph = app.get_graph()
            graph.draw_png("workflow_graph.png")

            return final_state["generated_tweet"]
        return None


if __name__ == "__main__":
    generator = TweetGenerator()
    tweet_text = generator.run()

    if tweet_text:
        logger.info(f"生成されたツイート:\n{tweet_text}")

        # 実際のツイート投稿（必要に応じてコメントアウトを解除）
        twitter = TwitterAdapter()
        twitter.post_tweet(tweet_text)
