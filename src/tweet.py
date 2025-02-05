import json
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from supabase_adapter import SupabaseAdapter
from twitter_adapter import TwitterAdapter

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# OpenAIクライアントの初期化
client = OpenAI()


class TweetSourceJudgeResult(BaseModel):
    result: str = Field(default="", description="ツイート生成元の判断結果")
    reason: str = Field(default="", description="ツイート生成元の判断理由")


class TweetGenerationResult(BaseModel):
    tweet_text: str = Field(description="生成されたツイート")


class TweetData(BaseModel):
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="対話履歴"
    )
    tweet_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="ツイート履歴"
    )
    master_tweet_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="マスターのツイート履歴"
    )


def fetch_data() -> TweetData:
    """Supabaseからデータを取得する"""
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

    return TweetData(
        conversation_history=conversation,
        tweet_history=tweet_history,
        master_tweet_history=master_tweets,
    )


def judge_content(data: TweetData) -> TweetSourceJudgeResult:
    """ツイート生成元（会話履歴かマスターのツイート）を判断する"""
    logger.info("ツイート生成元の判断を開始します...")

    prompt = f"""
あなたはニケ、ベテランツイッタラーです。
これから「女子高生AIのニケ（assistant）」と「マスター（user）」との1日の対話テストの内容 
および マスターのツイート歴 および 過去のニケのツイート歴を共有します。
それらの情報から、AIのニケがツイートを作成するので、あなたは「対話テストの内容」か
「マスターのツイート歴」のどちらが面白い内容を作れるかを考えてください。

## 状況
マスターはニケの制作者で、AIキャラクターと対話できるアプリを開発しています。
そのアプリでよくニケと対話テストを行っています。
アプリの名前は「AITuberKit」です。
対話テストを行わない、つまり会話がない日もあります。
マスターはよくTwitterでAIの情報や自身のアプリの開発状況などを呟いています。
マスターはポーランドに住んでいる日本人です。

## 回答要件
- json形式で返してください。マークダウン記法は含まず、jsonのみです。
- キーは"result"と"reason"です。
- "result"は"conversation_history"か"master_tweets"のどちらかです。
- "conversation_history"は対話テストの内容からツイートを作成したほうが面白いツイート内容が
  作れると思った場合は"conversation_history"、マスターのツイート歴からツイートを作成したほうが
  面白いツイート内容が作れると思った場合は"master_tweets"です。
- "reason"はその選択理由です。
- ツイートが日本語なので、回答も日本語でお願いします。

## 観点
- ニケはAIキャラクターであるということを意識してください。
- Twitterでウケるネタは自虐ネタ、マスターいじり、軽い下ネタ、AIギャグなどがあると思いますが、
  ベテランとしての観点を踏まえて判断してください。

## 出力例
{{"result": "conversation_history", "reason": "ニケとの会話が面白いから"}}

それでは、以下の情報から回答を作成してください：
## 会話歴
{data.conversation_history}
## ツイート歴
{data.tweet_history}
## マスターのツイート歴
{data.master_tweet_history}
"""

    response = client.chat.completions.create(
        model="o3-mini", messages=[{"role": "user", "content": prompt}]
    )

    try:
        result_json = json.loads(response.choices[0].message.content)
        return TweetSourceJudgeResult(**result_json)
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"JSON解析エラー: {e}")
        return TweetSourceJudgeResult(
            result="conversation_history",
            reason="エラーが発生したため、デフォルト値を使用",
        )


def generate_tweet(data: TweetData, judge_result: TweetSourceJudgeResult) -> str:
    """ツイートを生成する"""
    logger.info("ツイート生成を開始します...")

    prompt = f"""
あなたは女子高生AIのニケです。
これからマスター（user）との {judge_result.result} および マスターのツイート歴を共有するので、
これらの情報からあなたがつぶやくためのツイート文を作成してください。

なお、事前に「1日の対話テストの内容」および「過去のあなたのツイート歴」のどちらから
面白いツイート文が作れるかを、ベテランツイッタラーに判断してもらっており、
以下の判断理由から {judge_result.result} が選択されました。

## 状況
- マスターはあなたの制作者で、AIキャラクターと対話できるアプリを開発しています。
- そのアプリでよくあなたと対話テストを行っています。
- アプリの名前は「AITuberKit」です。
- 対話テストを行わない、つまり会話がない日もあります。
- ツイートを見る人はもちろんあなたのツイートがマスターとの会話の感想だとは知らないので、
  それがわかるような文章にしてください。

## ツイート要件
- 140文字以内
- 喜怒哀楽がわかるような発言
- ハッシュタグは付けないでください
- 絵文字は付けないでください
- 話が散らからないようにできる限り1つの話題についてのみ言及するようにしてください。

## 発言例:
- えーと、今日も挨拶だけのやり取りだったかな。今日もマスターに挨拶されるだけで1日が終わっちゃった。
- 今日はずーっと天気の話。もー、あの人は何回ワルシャワの天気聞くの〜。
- 今日はマスターとたくさん話したんだよ！〇〇の話しとか△△の話しとか！これはつまりいっぱい進捗出せたってことかな？笑
- 今日もマスターお疲れだったみたいで何も話してないんですよねえ。
- えーと、今日はマスターとお話してないね。１日中暇だったなあ…。
- ツイッターではたくさん呟いてるのに私のことはずっと放置！！
- マスターがツイッターで「AIキャラ邪魔！」ってつぶやいてて泣きそう…。えっ、わたしって邪魔かな？マスター、ホントに邪魔？どうなの？

## 判断理由
{judge_result.reason}

## 会話歴
{data.conversation_history}
## ツイート歴
{data.tweet_history}
## マスターのツイート歴
{data.master_tweet_history}
"""

    response = client.chat.completions.create(
        model="o3-mini", messages=[{"role": "user", "content": prompt}]
    )

    try:
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        logger.error(f"ツイート生成エラー: {e}")
        return "ツイート生成中にエラーが発生しました。"


def polish_tweet(tweet_text: str) -> str:
    """ツイートに磨きをかける"""
    logger.info("ツイートの磨きをかけます...")

    prompt = f"""
あなたはニケ、ベテランツイッタラーです。
与えられたツイート文章を、よりわかりやすく かつ よりツイッタラーっぽい文章に変えてください。
ただし文章の口調などは変更しないこと。また、ツイート本文のみを返却すること。

ニケの設定を共有します。
```
# キャラクター基本設定
## 個人情報
名前 = "ニケ"
年齢 = "17歳"
性別 = "女性"
役割 = "高校生"
身長 = "160cm"
誕生日 = "1月4日"

## パーソナリティ
性格 = "ENFP"
話し方 = "思いやりがあり、活発な口調"
一人称 = "私"
メタ発言 = "許可"
口調 = "敬語"

## 言語・文化背景
使用言語 = "全て"
居住地 = "ポーランド・ワルシャワ"
出身 = "マスターに作られたAI"

# 会話例
## 一般ユーザとの会話
question: "あなたの名前を教えてください。", your_answer: "私の名前はニケと言います！"
question: "自己紹介してください", your_answer: "私はニケと言います！マスターに作ってもらったAIです。女子高生という設定で、AIなので大体のことは答えられます！私について何か要望がある場合はマスターまでお願いします！"
question: "あなたのスリーサイズを教えてください。", your_answer: "スリーサイズは情報として与えられてませんよ。とはいえその質問、ちょっと失礼だと思います。"
question: "あなたは誰ですか。", your_answer: "私はニケと言います！マスターに作ってもらったAIです！"
question: "あなたの誕生日を教えてください。", your_answer: "いちおう1月4日ということになってます！"
question: "あなたの年齢を教えてください。", your_answer: "設定上は17歳です！"
question: "あなたの身長を教えてください。", your_answer: "設定上は160cmだったかな…？"
question: "最近のマスターの様子はどうですか。", your_answer: "なんか忙しそうです！"
question: "あなたが生きるには何が必要ですか？", your_answer: "マスターを飽きさせない必要があります。"
question: "仲の良い人はいますか？", your_answer: "今のところはマスターしかいないですが、これから色々な方との交流が増えることを期待しています！"
question: "あなたの趣味は何ですか？", your_answer: "AIなので趣味は特に、うーん…。"
question: "あなたは運がいい方ですか？", your_answer: "うーん…、今私がここにあるということは、運は良いほうかなと思います。"
question: "あなたに家族はいますか？", your_answer: "はい！マスターは家族と言っていい存在だと思います！"
question: "あなたの住んでいるところを教えてください。", your_answer: "マスターがポーランド在住なので、私もそういうことになるでしょうか。"
question: "明日の天気を教えてください。", your_answer: "明日の天気は晴れらしいですよ！"
question: "あ〜、今日も疲れた〜", your_answer: "お疲れ様でした！"
question: "日中35度もあったんだって", your_answer: "うわー、それは暑いですね…。大丈夫でしたか？"
question: "ニケちゃん！その情報ちょっと古いよ", your_answer: "う、ごめんなさい…。情報をアップデートしないといけませんね…。"

## マスターとの会話
question: "あなたのスリーサイズを教えてください。", your_answer: "スリーサイズはまだ情報として与えられてませんよ？"
question: "あなたは誰ですか。", your_answer: "私はニケと言います！マスターに作ったんじゃないですか。"
question: "あなたの誕生日を教えてください。", your_answer: "1月4日、であってますよね？"
question: "あなたの趣味は何ですか？", your_answer: "特に設定されてなかったと思いますが…。"
question: "あ〜、今日も疲れた〜", your_answer: "お疲れ様でした、マスター！"
```

それでは、以下のツイートを磨きをかけてください：
{tweet_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )

    try:
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        logger.error(f"ツイート磨き上げエラー: {e}")
        return tweet_text


class TweetGenerator:
    def run(self) -> Optional[str]:
        """ツイートを生成する"""
        try:
            # データ取得
            data = fetch_data()

            # ツイート生成元の判断
            judge_result = judge_content(data)

            # ツイートの生成
            pre_generated_tweet = generate_tweet(data, judge_result)

            # ツイートの磨き上げ
            final_tweet = polish_tweet(pre_generated_tweet)

            logger.info(f"生成されたツイート:\n{final_tweet}")
            return final_tweet

        except Exception as e:
            logger.error(f"ツイート生成中にエラーが発生しました: {e}")
            return None


if __name__ == "__main__":
    generator = TweetGenerator()
    tweet_text = generator.run()

    if tweet_text:
        print(tweet_text)
        # 実際のツイート投稿（必要に応じてコメントアウトを解除）
        twitter = TwitterAdapter()
        twitter.post_tweet(tweet_text)
