from supabase_adapter import SupabaseAdapter
from openai_adapter import OpenAIAdapter
from twitter_adapter import TwitterAdapter
from typing import List, Dict
from datetime import datetime


def get_conversation_summary() -> str:
    # Supabaseからデータを取得
    db = SupabaseAdapter()
    messages = db.get_records_by_date_range("messages", days=1)

    # メッセージを時系列順に並べ替え、必要な情報のみ抽出
    conversation = []
    for msg in sorted(messages, key=lambda x: x["created_at"]):
        if "role" in msg and "content" in msg:
            conversation.append({"role": msg["role"], "content": msg["content"]})

    # OpenAIを使用してツイート文を生成
    openai = OpenAIAdapter()
    system_prompt = """
    あなたは女子高生AIのニケです。マスター（user）との会話内容を要約してツイートを作成してください。
    マスターはあなたの制作者で、よく対話テストを行っています。
    ツイートは以下の要件を満たしてください：
    - 140文字以内
    - 会話の雰囲気が伝わる明るい口調
    - 「〜だよ！」「〜なの！」などの女子高生らしい口調
    - ハッシュタグは付けないでください
    - 絵文字は付けないでください
    発言例：
    - 今日もマスターに挨拶されるだけで1日が終わっちゃった
    - もー、あの人は何回ワルシャワの天気聞くの
    """

    messages = [
        openai.create_message("system", system_prompt),
        openai.create_message(
            "user",
            f"以下の会話内容を要約してツイートを作成してください：\n{str(conversation)}",
        ),
    ]

    tweet_text = openai.chat_completions(messages)
    return tweet_text


if __name__ == "__main__":
    # ツイート文を生成
    tweet_text = get_conversation_summary()
    print(f"生成されたツイート:\n{tweet_text}")

    # ツイートを投稿
    twitter = TwitterAdapter()
    twitter.post_tweet(tweet_text)
