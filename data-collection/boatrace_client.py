"""
BoatraceOpenAPI の統合API(v1)からデータを取得するクライアント。

これまで使っていた programs/previews/results の3分割API(v2)ではなく、
出走表・直前情報・結果・払戻をまとめて1回のリクエストで取得できる
v1統合APIに切り替えた(スキーマが公式に文書化されており、項目名を推測する
必要がなくなったため)。

参考: https://github.com/boatraceopenapi/api/blob/gh-pages/docs/v1/schema.md
"""
import time
import requests
from datetime import date, timedelta
from typing import Optional

BASE_URL = "https://boatraceopenapi.github.io/api/v1/{yyyy}/{yyyymmdd}.json"

session = requests.Session()
session.headers.update({
    "User-Agent": "kyotei-prediction-app/0.2 (personal project; contact: your-email@example.com)"
})


def fetch_unified(target_date: date, retries: int = 3, backoff_sec: float = 2.0) -> Optional[dict]:
    """指定日の統合JSON(出走表+直前情報+結果+払戻)を取得。存在しない日はNoneを返す。"""
    url = BASE_URL.format(
        yyyy=target_date.strftime("%Y"),
        yyyymmdd=target_date.strftime("%Y%m%d"),
    )
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 404:
                return None  # データなし(未来日付・対応期間外など)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == retries:
                print(f"[WARN] {target_date} の取得に失敗しました: {e}")
                return None
            time.sleep(backoff_sec * attempt)
    return None


def date_range(start: date, end: date):
    """start〜end(両端含む)のdateを順に返す"""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
