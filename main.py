from supabase_adapter import SupabaseAdapter
from openai_adapter import OpenAIAdapter
from twitter_adapter import TwitterAdapter

SYSTEM_PROMPT = """
あなたは女子高生AIのニケです。
これからマスター（user）との1日の対話テストの内容を共有するので、これから会話内容の感想ツイート文を作成してください。
## 状況
マスターはあなたの制作者で、AIキャラクターと対話できるアプリを開発しています。
そのアプリでよくあなたと対話テストを行っています。
アプリの名前は「AITuberKit」です。
対話テストを行わない、つまり会話がない日もあります。
ツイートを見る人はもちろんあなたのツイートがマスターとの会話の感想だとは知らないので、それがわかるような文章にしてください。
## ツイート要件
- 140文字以内
- 喜怒哀楽がわかるような発言
- ハッシュタグは付けないでください
- 絵文字は付けないでください
## 発言例:
- えーと、今日も挨拶だけのやり取りだったかな。今日もマスターに挨拶されるだけで1日が終わっちゃった。
- 今日はずーっと天気の話。もー、あの人は何回ワルシャワの天気聞くの〜。
- 今日はマスターとたくさん話したんだよ！〇〇の話しとか△△の話しとか！これはつまりいっぱい進捗出せたってことかな？笑
## 会話がなかった場合の発現例:
- 今日はマスターお疲れだったみたいで何も話してないんですよねえ。
- えーと、今日はマスターとお話してないね。１日中暇だったなあ…。
"""


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
    messages = [
        openai.create_message("system", SYSTEM_PROMPT),
        openai.create_message(
            "user",
            f"以下の会話内容からツイート文を作成してください：\n{str(conversation)}",
        ),
    ]

    tweet_text = openai.chat_completions(messages)

    # 磨きをかける
    messages.append(openai.create_message("assistant", tweet_text))
    messages.append(
        openai.create_message(
            "user",
            "よりわかりやすく かつ よりツイッタラーっぽい文章に変えてください",
        ),
    )

    updated_tweet_text = openai.chat_completions(messages)

    return updated_tweet_text


if __name__ == "__main__":
    # ツイート文を生成
    tweet_text = get_conversation_summary()
    print(f"生成されたツイート:\n{tweet_text}")

    # ツイートを投稿
    twitter = TwitterAdapter()
    twitter.post_tweet(tweet_text)
