"""
raw_json テーブルに保存されている生データの構造(実際のキー名)を確認するスクリプト。

ingest.py の parse_programs() / parse_previews() / parse_results() で使っている
pick() のキー名候補が、実際のAPIレスポンスと合っているかを確認するために使う。
推測ではなく実データで確認できるので、これを実行した結果を教えてもらえれば
正確に直せる。

使い方:
    python inspect_raw_json.py --db boatrace.db --date 2026-09-08 --kind previews
    (--date省略時は本日、--kind省略時はprograms)
"""
import argparse
import json
import sqlite3
from datetime import date


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (省略時は本日)")
    ap.add_argument("--kind", default="programs", choices=["programs", "previews", "results"])
    args = ap.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    conn = sqlite3.connect(args.db)
    row = conn.execute(
        "SELECT payload FROM raw_json WHERE race_date=? AND kind=?",
        (target_date.isoformat(), args.kind),
    ).fetchone()
    conn.close()

    if not row:
        print(f"データが見つかりません: date={target_date} kind={args.kind}")
        print("ingest.py をまだ実行していないか、その日のデータが未取得の可能性があります。")
        return

    data = json.loads(row[0])

    # トップレベルのキー
    print("=== トップレベルのキー ===")
    if isinstance(data, dict):
        print(list(data.keys()))
        races = data.get("results") or data.get("races") or data.get("programs") or []
    else:
        races = data
        print("(トップレベルはリスト)")

    if not races:
        print("レース配列が見つかりませんでした。")
        return

    # 1レース目のキー
    race = races[0]
    print("\n=== 1レース目のキー ===")
    print(list(race.keys()))

    # レース情報のうち、辞書ではない値(=スカラー値)だけ抜粋表示
    print("\n=== 1レース目の値(一部) ===")
    for k, v in race.items():
        if not isinstance(v, (list, dict)):
            print(f"  {k}: {v!r}")

    # 1艇目のキーと値
    boats = race.get("boats") or race.get("entries") or []
    if boats:
        boat = boats[0]
        print("\n=== 1艇目のキー ===")
        print(list(boat.keys()))
        print("\n=== 1艇目の値(一部) ===")
        for k, v in boat.items():
            if not isinstance(v, (list, dict)):
                print(f"  {k}: {v!r}")
    else:
        print("\nboats/entries 配列が見つかりませんでした。")


if __name__ == "__main__":
    main()
