from supabase_adapter import SupabaseAdapter
from openai_adapter import OpenAIAdapter
from src.twitter_adapter import TwitterAdapter

SYSTEM_PROMPT = """
以下にある人物のTwitterの投稿履歴を共有するので、それを要約してください。

## 状況
- Twitterの投稿主の名前はニケです。
- 彼女はフリーランスのエンジニアで、AI関連のサービスを開発しています。
- また、彼女は個人開発にも力を入れており、下記のアプリを開発しています。
    - AITuberKit
        - AIキャラクターと対話できるWebアプリ
        - ニケが最も力を入れている個人開発で、これ経由で認知しているフォロワーも多い
        - OSSとして公開しているが、商用利用に関してはライセンス料が発生する
    - AITuberList
        - AIキャラクターと対話できるアプリ
        - マスターとの会話の感想をツイートする機能があります。
    - ニケシステム
        - AIキャラクターと対話できるアプリ
        - マスターとの会話の感想をツイートする機能があります。
    - その他突発的に思いついたアプリを開発し始めることもあります。
- 基本的に毎日20件前後投稿するので、明らかに少ない場合は仕事で忙しいか外出している可能性があります。

## よく言及される人物
- 私
    - 投稿主であるニケのことです。
- AIニケちゃん
    - ニケが開発しているAIキャラクターです。ニケと名前は同じですが別々の人物です。
- ニケちゃん
    - ニケちゃんと同義です。しばしば「AI」の部分が省略されます。

## 要件
- 必ずしも短くする必要はありません。文字数は投稿数に合わせてください。
- あまりにも意味が通らない投稿は除外してください（投稿履歴には画像は含まれないため、画像に対する言及などはそれだけだと意味が通らない可能性があるため）
- この人物の1日の感情の流れに注目し、どのような変化があったかに注目してください。
- 日常、エンタメ、仕事、アプリ開発、その他など、できる限り細かく分類分けしてください。
"""


def get_conversation_summary() -> str:
    # Supabaseからデータを取得
    db = SupabaseAdapter()
    master_tweet_history = []
    master_tweets = db.get_records_by_date_range("my_tweets", days=1)
    for tweet in master_tweets:
        master_tweet_history.append(
            {
                "content": tweet["text"],
                "created_at": tweet["created_at"],
            }
        )

    # OpenAIを使用してツイート文を生成
    openai = OpenAIAdapter()
    messages = [
        openai.create_message("user", SYSTEM_PROMPT),
        openai.create_message(
            "user",
            f"以下のツイート歴から要約を作成してください：\n{str(master_tweet_history)}",
        ),
    ]

    summary = openai.chat_completions(messages)

    return summary


if __name__ == "__main__":
    # ツイート文を生成
    summary = get_conversation_summary()
    print(f"要約:\n{summary}")

    # # ツイートを投稿
    # twitter = TwitterAdapter()
    # twitter.post_tweet(tweet_text)
