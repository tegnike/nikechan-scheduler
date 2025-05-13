import logging
import os
import re
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
SUPABASE_TABLE_NAME = "articles"
SUPABASE_STORAGE_BUCKET = "note-thumbnails"
IDENTIFIER_COLUMN = "identifier"  # note記事およびZenn記事のキーカラム名

# Zenn関連の定数
ZENN_API_ENDPOINT = "https://zenn.dev/api/articles"
ZENN_USERNAME = "nikechan"
ZENN_ARTICLE_BASE_URL = "https://zenn.dev"
PLATFORM_COLUMN = "platform"


def fetch_all_articles() -> List[Dict]:
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


def fetch_zenn_articles() -> List[Dict]:
    """Zenn APIから指定ユーザーの全記事を取得する"""
    all_articles = []
    page = 1
    while True:
        try:
            params = {
                "username": ZENN_USERNAME,
                "order": "latest",
                "page": page,
            }
            response = requests.get(
                ZENN_API_ENDPOINT,
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            articles_on_page = data.get("articles", [])
            if not articles_on_page:
                logging.info(
                    f"Zenn: ページ {page} に記事が見つかりませんでした。取得完了。"
                )
                break

            all_articles.extend(articles_on_page)
            logging.info(
                f"Zenn: ページ {page} から {len(articles_on_page)} 件の記事を取得しました。"
            )

            if data.get("next_page") is None:
                logging.info("Zenn: 最終ページに到達しました。取得完了。")
                break
            else:
                page = data["next_page"]

        except requests.exceptions.Timeout:
            logging.warning(
                f"Zenn APIからのデータ取得中にタイムアウトしました (Page: {page})。リトライします..."
            )
            continue
        except requests.exceptions.RequestException as e:
            logging.error(
                f"Zenn APIからのデータ取得中にエラーが発生しました (Page: {page}): {e}"
            )
            break
        except Exception as e:
            logging.error(
                f"Zenn記事取得中に予期せぬエラーが発生しました (Page: {page}): {e}"
            )
            break

    logging.info(f"Zenn: 合計 {len(all_articles)} 件の記事を取得しました。")
    return all_articles


def extract_zenn_thumbnail_url_from_html(article_url: str) -> Optional[str]:
    """Zenn記事ページのHTMLからog:imageのURLを抽出する"""
    if not article_url:
        logging.warning("Zenn記事URLが空です。サムネイル抽出をスキップします。")
        return None

    html_content: Optional[str] = None
    try:
        response = requests.get(article_url, timeout=15)
        response.raise_for_status()
        html_content = response.text
    except requests.exceptions.Timeout:
        logging.error(f"Zenn記事ページの取得中にタイムアウトしました ({article_url})。")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(
            f"Zenn記事ページの取得中にエラーが発生しました ({article_url}): {e}"
        )
        return None
    except Exception as e:  # 他の予期せぬエラーキャッチ
        logging.error(f"Zenn記事ページ取得中に予期せぬ汎用エラー ({article_url}): {e}")
        return None

    if html_content is None:  # HTML取得に失敗した場合はここで終了
        return None

    # 正規表現でog:imageのcontentを抽出
    # <meta property="og:image" content="URL_HERE" /> または
    # <meta content="URL_HERE" property="og:image" /> に対応
    match = None
    try:
        match = re.search(
            r"<meta[^>]*property=[\'\"]og:image[\'\"][^>]*content=[\'\"]([^\'\"]+)[\'\"][^>]*>",
            html_content,
            re.IGNORECASE,
        )
        if not match:  # 順番が逆のパターンも考慮
            match = re.search(
                r"<meta[^>]*content=[\'\"]([^\'\"]+)[\'\"][^>]*property=[\'\"]og:image[\'\"][^>]*>",
                html_content,
                re.IGNORECASE,
            )
    except Exception as e:  # 正規表現処理中のエラー
        logging.error(f"ZennサムネイルURLの正規表現処理中にエラー ({article_url}): {e}")
        return None

    if match:
        thumbnail_url = match.group(1)
        logging.info(
            f"ZennサムネイルURL抽出成功: {thumbnail_url} (元URL: {article_url})"
        )
        return thumbnail_url
    else:
        logging.warning(
            f"Zenn記事ページからog:imageが見つかりませんでした: {article_url}"
        )
        return None


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
            logging.warning(
                f"不明なサムネイル拡張子({ext})です。"
                f" .png として扱います。 URL: {image_url}"
            )
            ext = ".png"
        file_name = f"{article_key}{ext}"
        storage_path = f"{file_name}"  # バケット直下

        # 画像データをダウンロード
        response = requests.get(image_url, stream=True, timeout=15)
        response.raise_for_status()
        image_data = response.content

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
    articles: List[Dict],
    existing_keys: set[str],
    supabase_adapter: SupabaseAdapter,
    platform: str,  # "note" または "zenn"
) -> Tuple[List[Dict], List[Dict]]:
    """
    APIデータを新規挿入用と更新用に振り分ける。
    platform引数に基づいてnoteとZennのデータ構造の違いに対応する。
    """
    records_to_insert = []
    records_to_update = []

    for article in articles:
        # --- プラットフォームごとのキーと必須フィールドを設定 ---
        if platform == "note":
            article_identifier_val = article.get("key")
            required_base_keys = ["key", "status", "likeCount"]
            required_insert_keys = required_base_keys + [
                "name",
                "publishAt",
                "eyecatch",
            ]
        elif platform == "zenn":
            article_identifier_val = str(article.get("slug", ""))
            required_base_keys = ["slug", "liked_count"]
            required_insert_keys = required_base_keys + [
                "title",
                "published_at",
                "user",
                "id",
            ]
        else:
            logging.error(f"未知のプラットフォームです: {platform}")
            continue

        if not article_identifier_val:
            logging.warning(
                f"{platform}: slugが見つからない記事データをスキップします。 Article: {article}"
            )
            continue

        is_existing_record = article_identifier_val in existing_keys
        current_required_keys = (
            required_base_keys if is_existing_record else required_insert_keys
        )

        if not all(k in article for k in current_required_keys):
            missing_keys = [k for k in current_required_keys if k not in article]
            logging.warning(
                f"{platform}: 必須キー {missing_keys} が不足。スキップ (slug: {article_identifier_val})"
            )
            continue

        status_value = "public"
        if platform == "note":
            status_value = article.get("status", "unknown")

        like_count = article.get(
            "likeCount" if platform == "note" else "liked_count", 0
        )

        if is_existing_record:
            record_data_to_update = {
                IDENTIFIER_COLUMN: article_identifier_val,
                "status": status_value,
                "like_count": like_count,
                PLATFORM_COLUMN: platform,
            }
            records_to_update.append(record_data_to_update)
        else:
            title = article.get("name" if platform == "note" else "title")
            publish_at_str = article.get(
                "publishAt" if platform == "note" else "published_at"
            )
            published_at_iso = None
            if publish_at_str:
                try:
                    if isinstance(publish_at_str, str):
                        dt_str = publish_at_str.replace("Z", "+00:00")
                        published_at_dt = datetime.fromisoformat(dt_str)
                        if published_at_dt.tzinfo is None:
                            published_at_dt = published_at_dt.replace(
                                tzinfo=timezone.utc
                            )
                        published_at_iso = published_at_dt.isoformat()
                    else:
                        raise ValueError("publish_at is not a string")
                except (ValueError, TypeError) as e:
                    logging.warning(
                        f"{platform}: 日付形式変換失敗。スキップ: "
                        f"{publish_at_str} (slug: {article_identifier_val}): {e}"
                    )
                    continue
            else:
                logging.warning(
                    f"{platform}: 公開日(publish_at)がありません。スキップ (slug: {article_identifier_val})"
                )
                continue

            thumbnail_url_for_db = None
            if platform == "note":
                eyecatch_url = article.get("eyecatch")
                if eyecatch_url:
                    thumbnail_url_for_db = upload_thumbnail(
                        supabase_adapter, eyecatch_url, article_identifier_val
                    )
            elif platform == "zenn":
                slug = article.get("slug")
                zenn_user_info = article.get("user", {})
                zenn_username_from_api = zenn_user_info.get("username")
                if slug and zenn_username_from_api:
                    zenn_article_url = f"{ZENN_ARTICLE_BASE_URL}/{zenn_username_from_api}/articles/{slug}"
                    extracted_thumb_url = extract_zenn_thumbnail_url_from_html(
                        zenn_article_url
                    )
                    if extracted_thumb_url:
                        thumbnail_file_key = f"zenn_{article_identifier_val}"
                        thumbnail_url_for_db = upload_thumbnail(
                            supabase_adapter, extracted_thumb_url, thumbnail_file_key
                        )
                else:
                    logging.warning(
                        f"Zenn: slug ({slug}) または username ({zenn_username_from_api}) が取得できず、"
                        f"サムネイル処理不可 (slug: {article_identifier_val})"
                    )
            records_to_insert.append(
                {
                    IDENTIFIER_COLUMN: article_identifier_val,
                    "title": title,
                    "published_at": published_at_iso,
                    "like_count": like_count,
                    "thumbnail_url": thumbnail_url_for_db,
                    "status": status_value,
                    PLATFORM_COLUMN: platform,
                }
            )
    return records_to_insert, records_to_update


def main():
    """メイン処理"""
    logging.info("処理を開始します...")

    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        logging.error(
            "Supabaseの認証情報が環境変数に見つかりません。処理を中断します。"
        )
        sys.exit(1)

    try:
        supabase_adapter = SupabaseAdapter()
    except ValueError as e:
        logging.error(f"Supabaseアダプターの初期化に失敗しました: {e}")
        sys.exit(1)

    # 1. note記事データをAPIから取得
    note_articles = fetch_all_articles()
    if not note_articles:
        logging.info("取得するnote記事がありませんでした。")
    else:
        logging.info(f"noteから {len(note_articles)} 件の記事を取得しました。")

    # 2. Zenn記事データをAPIから取得
    zenn_articles = fetch_zenn_articles()
    if not zenn_articles:
        logging.info("取得するZenn記事がありませんでした。")
    else:
        logging.info(f"Zennから {len(zenn_articles)} 件の記事を取得しました。")

    if not note_articles and not zenn_articles:
        logging.info(
            "取得する記事がnote、Zenn共に見つかりませんでした。処理を終了します。"
        )
        return

    # 3. APIから取得した全記事のキーリストを作成 (note と zenn)
    api_article_keys = []
    if note_articles:
        api_article_keys.extend([a["key"] for a in note_articles if "key" in a])
    if zenn_articles:
        api_article_keys.extend([str(a["slug"]) for a in zenn_articles if "slug" in a])

    if not api_article_keys:
        logging.info(
            "有効なキーを持つ記事がAPI応答にありませんでした。処理を終了します。"
        )
        return

    # 4. Supabaseから既存レコードのキーを取得 (重複を除外)
    # IDENTIFIER_COLUMN にはnoteのkeyもZennのid(str)も保存される想定
    unique_api_keys = list(set(api_article_keys))
    logging.info(
        f"{len(unique_api_keys)} 件のユニークキーについて既存レコードを確認します..."
    )
    existing_records = supabase_adapter.get_records_by_column_values(
        SUPABASE_TABLE_NAME, IDENTIFIER_COLUMN, unique_api_keys
    )
    # Supabaseから取得したキーも文字列としてセットに格納
    existing_keys = {str(r[IDENTIFIER_COLUMN]) for r in existing_records}
    logging.info(f"{len(existing_keys)} 件の既存レコードが見つかりました。")

    all_records_to_insert = []
    all_records_to_update = []

    # 5. note記事データの準備と振り分け
    if note_articles:
        logging.info("note記事の処理を開始します...")
        note_inserts, note_updates = prepare_records(
            note_articles, existing_keys, supabase_adapter, platform="note"
        )
        all_records_to_insert.extend(note_inserts)
        all_records_to_update.extend(note_updates)
        logging.info(
            f"note記事: {len(note_inserts)}件が新規挿入対象、"
            f"{len(note_updates)}件が更新対象です。"
        )

    # 6. Zenn記事データの準備と振り分け
    if zenn_articles:
        logging.info("Zenn記事の処理を開始します...")
        zenn_inserts, zenn_updates = prepare_records(
            zenn_articles, existing_keys, supabase_adapter, platform="zenn"
        )
        all_records_to_insert.extend(zenn_inserts)
        all_records_to_update.extend(zenn_updates)
        logging.info(
            f"Zenn記事: {len(zenn_inserts)}件が新規挿入対象、"
            f"{len(zenn_updates)}件が更新対象です。"
        )

    # 7. Supabaseにデータを一括挿入 (Insert)
    insert_count = 0
    if all_records_to_insert:
        logging.info(
            f"合計 {len(all_records_to_insert)} 件の新規レコードを挿入します..."
        )
        # SupabaseAdapterに一括挿入メソッドがあればそれを利用するのが望ましい
        # ここでは個別に挿入する前提
        for record in all_records_to_insert:
            inserted = supabase_adapter.insert_record(SUPABASE_TABLE_NAME, record)
            if inserted:
                insert_count += 1
            else:
                logging.error(
                    f"レコード挿入失敗: platform={record.get(PLATFORM_COLUMN)}, "
                    f"slug={record.get(IDENTIFIER_COLUMN)} Data: {record}"
                )
        logging.info(f"{insert_count} 件の新規レコードが挿入されました。")
    else:
        logging.info("挿入する新規レコードはありませんでした。")

    # 8. Supabaseのデータを一括更新 (Update)
    update_count = 0
    if all_records_to_update:
        logging.info(f"合計 {len(all_records_to_update)} 件のレコードを更新します...")
        # SupabaseAdapterに一括更新メソッドがあればそれを利用するのが望ましい
        # ここでは個別に更新する前提
        for record_to_update in all_records_to_update:
            article_key_val = record_to_update[IDENTIFIER_COLUMN]
            # prepare_recordsで更新用データは整形済みのはず
            # IDENTIFIER_COLUMN以外のキーを更新データとして渡す
            update_payload = {
                k: v for k, v in record_to_update.items() if k != IDENTIFIER_COLUMN
            }

            supabase_adapter.update_record(
                SUPABASE_TABLE_NAME, IDENTIFIER_COLUMN, article_key_val, update_payload
            )
            # update_recordが成功したかどうかの明確な戻り値がない場合、
            # エラーログが出なければ成功とみなす運用
            update_count += 1

        logging.info(f"{update_count} 件のレコードが更新されました（試行ベース）。")
    else:
        logging.info("更新するレコードはありませんでした。")

    logging.info("全ての処理が完了しました。")


if __name__ == "__main__":
    main()
