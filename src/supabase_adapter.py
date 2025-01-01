import os
import pprint
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client


class SupabaseAdapter:
    def __init__(self):
        # 環境変数から認証情報を読み込み
        load_dotenv()
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not found in environment variables")

        self.client = create_client(supabase_url, supabase_key)

    def get_all_records(self, table_name: str) -> List[Dict]:
        """テーブルの全レコードを取得"""
        try:
            response = self.client.table(table_name).select("*").execute()
            return response.data
        except Exception as e:
            print(f"Error fetching records: {e}")
            return []

    def get_record_by_id(self, table_name: str, record_id: int) -> Optional[Dict]:
        """指定されたIDのレコードを取得"""
        try:
            response = (
                self.client.table(table_name).select("*").eq("id", record_id).execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error fetching record: {e}")
            return None

    def insert_record(self, table_name: str, data: Dict) -> Optional[Dict]:
        """新規レコードを挿入"""
        try:
            response = self.client.table(table_name).insert(data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error inserting record: {e}")
            return None

    def update_record(
        self, table_name: str, record_id: int, data: Dict
    ) -> Optional[Dict]:
        """レコードを更新"""
        try:
            response = (
                self.client.table(table_name).update(data).eq("id", record_id).execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error updating record: {e}")
            return None

    def delete_record(self, table_name: str, record_id: int) -> bool:
        """レコードを削除"""
        try:
            self.client.table(table_name).delete().eq("id", record_id).execute()
            return True
        except Exception as e:
            print(f"Error deleting record: {e}")
            return False

    def get_records_by_date_range(
        self, table_name: str, end_date: Optional[datetime] = None, days: int = 1
    ) -> List[Dict]:
        """指定期間内のレコードを取得
        Args:
            table_name (str): テーブル名
            end_date (Optional[datetime]): 終了日時（未指定の場合は現在時刻）
            days (int): 遡る日数（デフォルトは1日）
        Returns:
            List[Dict]: 該当するレコードのリスト
        """
        try:
            if end_date is None:
                end_date = datetime.now()

            start_date = end_date - timedelta(days=days)

            response = (
                self.client.table(table_name)
                .select("*")
                .gte("created_at", start_date.isoformat())
                .lte("created_at", end_date.isoformat())
                .execute()
            )
            return response.data
        except Exception as e:
            print(f"Error fetching records by date range: {e}")
            return []


# テスト実行用コード
if __name__ == "__main__":
    # テスト用のテーブル名
    TEST_TABLE = "messages"

    # Supabaseアダプターのインスタンスを作成
    db = SupabaseAdapter()

    # 4. 全レコードの取得をテスト
    print("\n全レコードを取得中...")
    all_records = db.get_all_records(TEST_TABLE)
    print(f"全レコード数: {len(all_records)}")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(all_records)

    # 日付範囲での取得をテスト
    print("\n過去24時間のレコードを取得中...")
    recent_records = db.get_records_by_date_range(TEST_TABLE)
    print(f"該当レコード数: {len(recent_records)}")
    pp.pprint(recent_records)

    # 特定の期間を指定してテスト
    specific_date = datetime(2024, 12, 1)
    print(f"\n{specific_date.date()}から3日間のレコードを取得中...")
    specific_records = db.get_records_by_date_range(
        TEST_TABLE, end_date=specific_date, days=3
    )
    print(f"該当レコード数: {len(specific_records)}")
    pp.pprint(specific_records)
