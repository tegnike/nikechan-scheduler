import os
import pprint
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

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
        self,
        table_name: str,
        identifier_column: str,
        identifier_value: Any,
        data: Dict,
    ) -> Optional[Dict]:
        """指定されたカラムと値に一致するレコードを更新"""
        try:
            response = (
                self.client.table(table_name)
                .update(data)
                .eq(identifier_column, identifier_value)
                .execute()
            )
            # 更新されたレコードが返ってくるか確認 (返ってこない場合もある)
            return response.data[0] if response.data else None
        except Exception as e:
            print(
                f"Error updating record where {identifier_column}={identifier_value}: {e}"
            )
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
        self,
        table_name: str,
        end_date: Optional[datetime] = None,
        days: int = 1,
        start_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """指定期間内のレコードを取得
        Args:
            table_name (str): テーブル名
            end_date (Optional[datetime]): 終了日時
            days (int): 遡る日数（start_dateが指定されていない場合に使用）
            start_date (Optional[datetime]): 開始日時（指定された場合はdaysは無視）
        Returns:
            List[Dict]: 該当するレコードのリスト
        """
        try:
            all_records = []
            page = 0
            page_size = 1000  # Supabaseの1回のクエリでの最大取得件数

            while True:
                query = self.client.table(table_name).select("*")

                if start_date:
                    query = query.gte("created_at", start_date.isoformat())
                else:
                    if end_date is None:
                        end_date = datetime.now(timezone.utc)
                    start_date = end_date - timedelta(days=days)
                    query = query.gte("created_at", start_date.isoformat())

                if end_date:
                    query = query.lt("created_at", end_date.isoformat())

                # ページネーションの設定
                query = query.range(page * page_size, (page + 1) * page_size - 1)

                # データの取得
                response = query.execute()
                current_page_data = response.data

                # データが存在しない場合はループを終了
                if not current_page_data:
                    break

                all_records.extend(current_page_data)

                # 取得したデータが1ページ分に満たない場合はループを終了
                if len(current_page_data) < page_size:
                    break

                page += 1

            return all_records
        except Exception as e:
            print(f"Error fetching records by date range: {e}")
            return []

    def get_record_by_condition(
        self, table_name: str, column: str, value: Any
    ) -> Optional[Dict]:
        """指定されたカラムと値に一致するレコードを取得"""
        try:
            response = (
                self.client.table(table_name).select("*").eq(column, value).execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error fetching record by condition: {e}")
            return None

    def get_records_by_condition(
        self, table_name: str, column: str, value: Any
    ) -> List[Dict]:
        """指定されたカラムと値に一致する全てのレコードを取得"""
        try:
            response = (
                self.client.table(table_name).select("*").eq(column, value).execute()
            )
            return response.data
        except Exception as e:
            print(f"Error fetching records by condition: {e}")
            return []

    def get_records_by_column_values(
        self, table_name: str, column: str, values: List[Any]
    ) -> List[Dict]:
        """指定されたカラムの値リストに一致するレコードを取得"""
        try:
            # Supabaseの `in_` フィルタを使用
            response = (
                self.client.table(table_name).select("*").in_(column, values).execute()
            )
            return response.data
        except Exception as e:
            print(f"Error fetching records by column values: {e}")
            return []

    def upload_to_storage(self, bucket: str, path: str, data: bytes) -> bool:
        """Supabase Storageにファイルをアップロード

        Args:
            bucket (str): バケット名
            path (str): ファイルパス
            data (bytes): アップロードするバイナリデータ

        Returns:
            bool: アップロード成功時はTrue、失敗時はFalse
        """
        try:
            self.client.storage.from_(bucket).upload(path, data)
            return True
        except Exception as e:
            print(f"Error uploading file to storage: {e}")
            return False

    def get_storage_public_url(self, bucket: str, path: str) -> str:
        """Supabase StorageのファイルのURLを取得

        Args:
            bucket (str): バケット名
            path (str): ファイルパス

        Returns:
            str: ファイルの公開URL
        """
        return self.client.storage.from_(bucket).get_public_url(path)

    def upsert_records(
        self, table_name: str, records: List[Dict], on_conflict: str
    ) -> List[Dict]:
        """複数のレコードをUpsert (Insert or Update)

        Args:
            table_name (str): テーブル名
            records (List[Dict]): 挿入または更新するレコードのリスト
            on_conflict (str): コンフリクトが発生した場合に
                               一意性を判断するカラム名

        Returns:
            List[Dict]: Upsertされたレコードのリスト
        """
        try:
            response = (
                self.client.table(table_name)
                .upsert(records, on_conflict=on_conflict)
                .execute()
            )
            return response.data
        except Exception as e:
            print(f"Error upserting records: {e}")
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
