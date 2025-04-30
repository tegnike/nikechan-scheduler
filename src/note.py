import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# srcディレクトリをsys.pathに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# supabase_adapterのインポートは他のimportの後で行う
# linterエラー(module level import not at top of file)を避けるため
try:
    from supabase_adapter import SupabaseAdapter  # type: ignore
except ImportError:
    print("Error: supabase_adapter.py not found or import failed.", file=sys.stderr)
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 定数
NOTE_API_ENDPOINT = "https://note.com/api/v2/creators/nike_cha_n/contents"
NOTE_ARTICLE_KIND = "note"
SUPABASE_TABLE_NAME = "note_articles"
SUPABASE_STORAGE_BUCKET = "note-thumbnails"
NOTE_KEY_COLUMN = "note_key"


def fetch_all_note_articles() -> List[Dict]:
    """note.com APIから全記事を取得する"""
    all_articles = []
    page = 1
    while True:
        try:
            params = {"kind": NOTE_ARTICLE_KIND, "page": page}
            response = requests.get(
                NOTE_API_ENDPOINT,
                params=params,
                timeout=15,  # timeout延長
            )
            response.raise_for_status()  # HTTPエラーチェック
            data = response.json()

            articles = data.get("data", {}).get("contents", [])
            if not articles:
                logging.info(f"ページ {page} に記事が見つかりませんでした。取得完了。")
                break

            all_articles.extend(articles)
            logging.info(f"ページ {page} から {len(articles)} 件の記事を取得しました。")

            # 次のページがあるかどうかの判断
            is_last_page = data.get("data", {}).get("isLastPage", True)
            if is_last_page:
                logging.info("最終ページに到達しました。取得完了。")
                break
            else:
                page += 1

        except requests.exceptions.Timeout:
            logging.warning(
                f"note APIからのデータ取得中にタイムアウトしました (Page: {page})。リトライします..."
            )
            # リトライロジックをここに追加することも可能 (例: time.sleep(5))
            continue  # 同じページでリトライ
        except requests.exceptions.RequestException as e:
            logging.error(
                f"note APIからのデータ取得中にエラーが発生しました (Page: {page}): {e}"
            )
            break
        except Exception as e:
            logging.error(f"予期せぬエラーが発生しました (Page: {page}): {e}")
            break

    logging.info(f"合計 {len(all_articles)} 件の記事を取得しました。")
    return all_articles


def upload_thumbnail(
    supabase_adapter: SupabaseAdapter, image_url: str, article_key: str
) -> Optional[str]:
    """サムネイル画像をSupabase Storageにアップロードし、公開URLを返す"""
    if not image_url:
        logging.warning(f"サムネイルURLが空です (key: {article_key})。スキップします。")
        return None

    try:
        # 画像ファイル名を設定
        parsed_url = urlparse(image_url)
        _, ext = os.path.splitext(parsed_url.path)
        if not ext or ext.lower() not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            # 不明な拡張子や拡張子がない場合は .png を使うか、エラーにする
            logging.warning(
                f"不明なサムネイル拡張子({ext})です。 .png として扱います。 URL: {image_url}"
            )
            ext = ".png"
        file_name = f"{article_key}{ext}"
        storage_path = f"{file_name}"  # バケット直下

        # 画像データをダウンロード
        response = requests.get(image_url, stream=True, timeout=15)  # timeout延長
        response.raise_for_status()
        image_data = response.content

        # Supabase Storageにアップロード (同名ファイルは上書きされる)
        # Storage側でファイルが存在するかチェックして上書きを避けることも可能だが、
        # URLが変わらない限り毎回アップロードしても実害は少ないと判断。
        # 必要であればここで存在チェックを追加。
        success = supabase_adapter.upload_to_storage(
            SUPABASE_STORAGE_BUCKET, storage_path, image_data
        )

        if success:
            public_url = supabase_adapter.get_storage_public_url(
                SUPABASE_STORAGE_BUCKET, storage_path
            )
            logging.info(f"サムネイル画像をアップロード/更新しました: {public_url}")
            return public_url
        else:
            logging.error(f"サムネイル画像のアップロードに失敗しました: {image_url}")
            return None

    except requests.exceptions.Timeout:
        logging.error(
            f"サムネイル画像のダウンロード中にタイムアウトしました ({image_url})。"
        )
        return None
    except requests.exceptions.RequestException as e:
        logging.error(
            f"サムネイル画像のダウンロード中にエラーが発生しました ({image_url}): {e}"
        )
        return None
    except Exception as e:
        logging.error(
            f"サムネイル処理中に予期せぬエラーが発生しました ({image_url}): {e}"
        )
        return None


def prepare_records(
    articles: List[Dict], existing_keys: set[str], supabase_adapter: SupabaseAdapter
) -> Tuple[List[Dict], List[Dict]]:
    """APIデータを新規挿入用と更新用に振り分ける"""
    records_to_insert = []
    records_to_update = []

    required_base_keys = ["key", "status", "likeCount"]
    required_insert_keys = required_base_keys + ["name", "publishAt", "eyecatch"]

    for article in articles:
        note_key = article.get("key")
        if not note_key:
            logging.warning("キー(key)が見つからない記事データをスキップします。")
            continue

        # 更新か挿入かで必須キーを判断
        required_keys = (
            required_base_keys if note_key in existing_keys else required_insert_keys
        )
        if not all(k in article for k in required_keys):
            logging.warning(
                f"必要なキーが不足している記事データをスキップします (key: {note_key})"
            )
            continue

        # 共通データの準備
        status = article["status"]
        like_count = article["likeCount"]

        if note_key in existing_keys:
            # --- 更新レコードの準備 --- (status と like_count のみ)
            records_to_update.append(
                {
                    NOTE_KEY_COLUMN: note_key,
                    "status": status,
                    "like_count": like_count,
                }
            )
        else:
            # --- 新規挿入レコードの準備 --- (サムネイル処理も含む)
            title = article["name"]
            publish_at_str = article["publishAt"]
            eyecatch_url = article["eyecatch"]

            # 日付変換
            try:
                published_at_dt = datetime.fromisoformat(publish_at_str)
                if published_at_dt.tzinfo is None:
                    published_at_dt = published_at_dt.replace(tzinfo=timezone.utc)
                published_at_iso = published_at_dt.isoformat()
            except ValueError:
                logging.warning(
                    f"日付形式の変換に失敗しました。スキップします: "
                    f"{publish_at_str} (key: {note_key})"
                )
                continue

            # サムネイル処理 (新規挿入時のみ)
            thumbnail_url = upload_thumbnail(supabase_adapter, eyecatch_url, note_key)

            records_to_insert.append(
                {
                    NOTE_KEY_COLUMN: note_key,
                    "title": title,
                    "published_at": published_at_iso,
                    "like_count": like_count,
                    "thumbnail_url": thumbnail_url,
                    "status": status,
                }
            )

    return records_to_insert, records_to_update


def main():
    """メイン処理"""
    logging.info("処理を開始します...")

    # 環境変数を読み込み
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        logging.error(
            "Supabaseの認証情報が環境変数に見つかりません。処理を中断します。"
        )
        sys.exit(1)

    # Supabaseアダプターを初期化
    try:
        supabase_adapter = SupabaseAdapter()
    except ValueError as e:
        logging.error(f"Supabaseアダプターの初期化に失敗しました: {e}")
        sys.exit(1)

    # 1. note記事データをAPIから取得
    articles = fetch_all_note_articles()
    if not articles:
        logging.info("取得する記事がありませんでした。処理を終了します。")
        return

    # 2. APIから取得した全記事のnote_keyリストを作成
    api_note_keys = [a["key"] for a in articles if "key" in a]
    if not api_note_keys:
        logging.info("有効なキーを持つ記事がAPI応答にありませんでした。")
        return

    # 3. Supabaseから既存レコードのnote_keyを取得
    logging.info(f"{len(api_note_keys)} 件のキーについて既存レコードを確認します...")
    existing_records = supabase_adapter.get_records_by_column_values(
        SUPABASE_TABLE_NAME, NOTE_KEY_COLUMN, api_note_keys
    )
    existing_keys = {r[NOTE_KEY_COLUMN] for r in existing_records}
    logging.info(f"{len(existing_keys)} 件の既存レコードが見つかりました。")

    # 4. APIデータを新規挿入用と更新用に振り分け、サムネイル処理(新規のみ)
    records_to_insert, records_to_update = prepare_records(
        articles, existing_keys, supabase_adapter
    )

    # 5. Supabaseにデータを挿入 (Insert)
    insert_count = 0
    if records_to_insert:
        logging.info(f"{len(records_to_insert)} 件の新規レコードを挿入します...")
        # insert_records のような一括挿入メソッドがあれば効率的
        # 今回は個別に挿入
        for record in records_to_insert:
            inserted = supabase_adapter.insert_record(SUPABASE_TABLE_NAME, record)
            if inserted:
                insert_count += 1
            else:
                logging.error(
                    f"レコードの挿入に失敗しました: key={record.get(NOTE_KEY_COLUMN)}"
                )
        logging.info(f"{insert_count} 件の新規レコードが挿入されました。")
    else:
        logging.info("挿入する新規レコードはありませんでした。")

    # 6. Supabaseのデータを更新 (Update)
    update_count = 0
    if records_to_update:
        logging.info(f"{len(records_to_update)} 件のレコードを更新します...")
        for record in records_to_update:
            note_key = record[NOTE_KEY_COLUMN]
            update_data = {
                "status": record["status"],
                "like_count": record["like_count"],
            }
            updated = supabase_adapter.update_record(
                SUPABASE_TABLE_NAME, NOTE_KEY_COLUMN, note_key, update_data
            )
            # update_recordは成功してもNoneを返すことがあるため、エラーログがないかで判断
            # if updated is not None: # 正確な成功判定が難しい
            update_count += 1  # ここではエラーが出なければカウント

        logging.info(f"{update_count} 件のレコードが更新されました（試行ベース）。")
    else:
        logging.info("更新するレコードはありませんでした。")

    logging.info("処理が完了しました。")


if __name__ == "__main__":
    main()
