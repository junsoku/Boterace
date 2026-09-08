"""
BoatraceOpenAPI (https://github.com/BoatraceOpenAPI) から
出走表(programs)・直前情報(previews)・結果(results) のJSONを取得するクライアント。

※ このAPIは非公式の有志プロジェクトです。データの正確性・完全性は保証されないため、
   本番運用では公式サイト(boatrace.jp)の値と定期的に突き合わせることを推奨します。
※ 「ボートレース日和」は三次加工サイトのため直接スクレイピングせず、こちらの
   一次的なオープンデータを使う方針に切り替えています(構造変更に強く、法的にも安全)。
"""
import time
import requests
from datetime import date, timedelta
from typing import Literal, Optional

BASE_URLS = {
    "programs": "https://boatraceopenapi.github.io/programs/v2/{yyyy}/{yyyymmdd}.json",
    "previews": "https://boatraceopenapi.github.io/previews/v2/{yyyy}/{yyyymmdd}.json",
    "results":  "https://boatraceopenapi.github.io/results/v2/{yyyy}/{yyyymmdd}.json",
}

Kind = Literal["programs", "previews", "results"]

session = requests.Session()
session.headers.update({
    "User-Agent": "kyotei-prediction-app/0.1 (personal project; contact: your-email@example.com)"
})


def fetch_json(kind: Kind, target_date: date, retries: int = 3, backoff_sec: float = 2.0) -> Optional[dict]:
    """指定日・指定種別のJSONを取得。存在しない日付は404が返るのでNoneを返す。"""
    url = BASE_URLS[kind].format(
        yyyy=target_date.strftime("%Y"),
        yyyymmdd=target_date.strftime("%Y%m%d"),
    )
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 404:
                return None  # データなし(未来日付・対応期間外など)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == retries:
                print(f"[WARN] {kind} {target_date} の取得に失敗しました: {e}")
                return None
            time.sleep(backoff_sec * attempt)  # 指数的バックオフでリトライ
    return None


def date_range(start: date, end: date):
    """start〜end(両端含む)のdateを順に返す"""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_day(target_date: date) -> dict:
    """1日分の3種類のデータをまとめて取得(呼び出し間隔を空けてサーバー負荷に配慮)"""
    result = {}
    for kind in ("programs", "previews", "results"):
        result[kind] = fetch_json(kind, target_date)
        time.sleep(0.5)  # 連続アクセスを避ける
    return result
