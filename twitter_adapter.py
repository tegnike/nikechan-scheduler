import os
import dotenv
import tweepy
from typing import Optional
from supabase_adapter import SupabaseAdapter


class TwitterAdapter:
    def __init__(self):
        # 環境変数の読み込み
        dotenv.load_dotenv()

        # API認証情報の取得
        self.consumer_key = os.getenv("CONSUMER_KEY")
        self.consumer_secret = os.getenv("CONSUMER_SECRET")
        self.bearer_token = os.getenv("BEARER_TOKEN")
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

        # APIクライアントの初期化
        self.client = self._initialize_client()

        # SupabaseAdapterのインスタンスを追加
        self.db = SupabaseAdapter()

    def _initialize_client(self) -> tweepy.Client:
        """Twitter APIクライアントを初期化する"""
        return tweepy.Client(
            consumer_key=self.consumer_key,
            consumer_secret=self.consumer_secret,
            bearer_token=self.bearer_token,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
        )

    def post_tweet(self, text: str) -> Optional[tweepy.Response]:
        """ツイートを投稿する

        Args:
            text (str): 投稿するツイートの本文

        Returns:
            Optional[tweepy.Response]: 投稿成功時はResponseオブジェクト、失敗時はNone
        """
        try:
            response = self.client.create_tweet(text=text)
            print("ツイートの投稿に成功しました！")

            # tweetsテーブルにツイート内容を保存
            tweet_data = { "content": text }
            self.db.insert_record("tweets", tweet_data)
            print("ツイート内容をデータベースに保存しました")

            return response
        except tweepy.TweepError as e:
            print(f"エラーが発生しました: {e}")
            return None


if __name__ == "__main__":
    # TwitterAdapterのテスト
    adapter = TwitterAdapter()
    test_tweet = "テストツイート。マスターがやってるよ。"
    adapter.post_tweet(test_tweet)
