"""
turnmark/api(非公式、前日までの確定オッズをJSON形式で提供。
https://github.com/turnmark/api)から過去のオッズを取得し、
自分たちのモデルが推奨した3連単の組み合わせ(prediction_log.top_bets_json)が、
実際の市場ではどのくらいのオッズ(=市場が織り込んでいた確率)だったかを比較する。

目的:
    「モデルが高確率と予測した組み合わせが、市場のオッズでも同じように評価されて
    いるのか、それとも市場が気づいていない(過小評価されている=割安な)組み合わせ
    なのか」を検証する。的中率だけで選別する今のA〜D判定は、控除率(約25%)の壁を
    超えられない可能性が高いことが分かっている。もし「モデルの予測確率 > 市場の
    暗黙の確率」となる組み合わせに絞って買った場合の回収率が高ければ、的中率ではなく
    期待値で選別する新しい軸になり得る。

データソースの制約(必ず踏まえること):
    - turnmark/api は前日までのデータしか提供しない(当日のリアルタイムオッズは
      含まれない)。このスクリプトは研究・検証専用であり、当日の本番予想・
      export_today.py の計算には一切使わない。
    - 1日1ファイル(https://turnmark.github.io/api/v1/YYYY/YYYYMMDD.json)で
      全24場・その日の全レース分をまとめて提供している。1日1回更新。
    - 非公式のAPIであり、正確性・完全性は保証されていない(ドキュメントにも明記)。
    - 以前boatrace.dbが100MBを超えてpushできなくなった教訓を踏まえ、取得した
      オッズはDBに保存しない(1レース120通りの3連単オッズを全件保存すると、
      すぐにまた同じ問題が起きる)。自分たちが推奨した組み合わせの分だけ、
      その場で突き合わせて集計する。

使い方:
    python analyze_odds_edge.py --db boatrace.db --start 2026-10-01 --end 2026-10-14
"""
import argparse
import sqlite3
import time
from datetime import date, timedelta

import requests

TURNMARK_BASE_URL = "https://turnmark.github.io/api/v1"
REQUEST_INTERVAL_SEC = 1.0  # 公開GitHub Pagesとはいえ、短時間に連続アクセスしすぎないよう間隔を空ける


def fetch_turnmark_day(d: date) -> dict:
    """指定日のturnmarkデータを取得する。対応期間外・未取得の日付は404になるのでNoneを返す。"""
    url = f"{TURNMARK_BASE_URL}/{d.year:04d}/{d.strftime('%Y%m%d')}.json"
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        print(f"  取得失敗 {d}: {e}")
        return None
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def get_trifecta_odds(day_data: dict, stadium_number: int, race_number: int, combo: str) -> float:
    """day_dataから、指定レース・指定組み合わせ(例 "4-3-1")の3連単オッズを取り出す。
    該当が無ければ(欠場艇絡み・データ欠損等)Noneを返す。"""
    try:
        parts = combo.split("-")
        race = day_data["programs"]["stadiums"][str(stadium_number)]["races"][str(race_number)]
        odds = race["odds"]["trifecta"]
        return odds[parts[0]][parts[1]][parts[2]]
    except (KeyError, TypeError, IndexError):
        return None


def _fetch_our_recommendations(conn: sqlite3.Connection, target_date: str) -> list:
    """その日の、prediction_logの最終スナップショット(=締切前の最後の予想)を
    レースごとに1件ずつ取得する。(top_bet_combo・top_bets_jsonの候補・結果も含む)"""
    import json

    rows = conn.execute(
        """
        SELECT pl.race_id, r.stadium_number, r.race_number, pl.top_bets_json
        FROM prediction_log pl
        JOIN races r ON r.race_id = pl.race_id
        WHERE r.race_date = ?
        """,
        (target_date,),
    ).fetchall()

    results = []
    for race_id, stadium_number, race_number, top_bets_json in rows:
        if not top_bets_json:
            continue
        try:
            candidates = json.loads(top_bets_json)
        except (TypeError, ValueError):
            continue

        result_rows = conn.execute(
            """SELECT e.boat_number, r.arrival_order
               FROM entries e JOIN results r ON r.entry_id = e.entry_id
               WHERE e.race_id = ? ORDER BY e.boat_number""",
            (race_id,),
        ).fetchall()
        if len(result_rows) != 6 or any(row[1] is None for row in result_rows):
            continue  # 結果未確定のレースは検証対象外
        ordered = sorted(result_rows, key=lambda row: row[1])
        actual_combo = "-".join(str(row[0]) for row in ordered[:3])

        results.append({
            "race_id": race_id, "stadium_number": stadium_number, "race_number": race_number,
            "candidates": candidates, "actual_combo": actual_combo,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    # 集計用: モデル予測確率が市場の暗黙確率を上回っていた(=市場で割安だった)候補群と、
    # 逆に下回っていた(=市場で割高だった)候補群で、それぞれ実際の的中率・回収率を比べる。
    STAKE = 100
    edge_group = {"n": 0, "hits": 0, "stake": 0, "payout": 0}       # モデル予測 > 市場の暗黙確率
    no_edge_group = {"n": 0, "hits": 0, "stake": 0, "payout": 0}    # モデル予測 <= 市場の暗黙確率
    no_odds_count = 0
    no_data_days = 0

    d = start
    while d <= end:
        print(f"取得中: {d}")
        day_data = fetch_turnmark_day(d)
        time.sleep(REQUEST_INTERVAL_SEC)
        if day_data is None:
            print(f"  データなし(対応期間外 or 未取得)")
            no_data_days += 1
            d += timedelta(days=1)
            continue

        races = _fetch_our_recommendations(conn, d.isoformat())
        for race in races:
            for cand in race["candidates"]:
                combo = cand.get("combo")
                prob_str = cand.get("prob", "")
                try:
                    model_prob = float(prob_str.rstrip("%")) / 100
                except (ValueError, AttributeError):
                    continue
                if not combo:
                    continue

                market_odds = get_trifecta_odds(day_data, race["stadium_number"], race["race_number"], combo)
                if market_odds is None or market_odds <= 0:
                    no_odds_count += 1
                    continue
                # オッズから市場の暗黙確率を逆算(控除率を無視した簡易版。オッズ1/オッズ)
                market_implied_prob = 1 / market_odds

                hit = combo == race["actual_combo"]
                payout = round(market_odds * STAKE) if hit else 0

                bucket = edge_group if model_prob > market_implied_prob else no_edge_group
                bucket["n"] += 1
                bucket["stake"] += STAKE
                bucket["payout"] += payout
                if hit:
                    bucket["hits"] += 1

        d += timedelta(days=1)

    conn.close()

    print(f"\n対応期間外・データなしの日数: {no_data_days}日")
    print(f"オッズが見つからなかった候補数(欠場艇絡み等): {no_odds_count}件\n")

    for label, bucket in [("モデル予測確率 > 市場の暗黙確率(市場で割安と判断した候補)", edge_group),
                           ("モデル予測確率 <= 市場の暗黙確率(市場並み〜割高と判断した候補)", no_edge_group)]:
        n = bucket["n"]
        if n == 0:
            print(f"{label}: 対象0件")
            continue
        hit_rate = bucket["hits"] / n * 100
        roi = bucket["payout"] / bucket["stake"] * 100 if bucket["stake"] else 0
        print(f"{label}")
        print(f"  件数: {n} / 的中率: {hit_rate:.1f}% / 回収率: {roi:.1f}%\n")

    print("※ 「市場で割安」グループの回収率が「市場並み〜割高」グループより明確に高ければ、"
          "期待値ベースの選別に伸びしろがある可能性が高い。差が無ければ、少なくとも3連単では"
          "市場の方が効率的に織り込んでいる(=このアプローチでは控除率の壁を超えにくい)ことになる。")
    print("※ オッズは控除率(胴元の取り分、約25%)を含んだ値なので、ここでの「市場の暗黙確率」は"
          "正確な確率ではなく簡易的な近似値であることに注意。")


if __name__ == "__main__":
    main()
