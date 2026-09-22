"""
prediction_log(過去の予想ログ)と payouts(実際の払戻金)を突き合わせて、
「どんなパターンの予想が実際に儲かっている/儲かっていないか」を複数の軸で分解する。

analyze_roi.py が「自信あり/なし」の1軸だけだったのに対し、こちらは:
    - 号艇別(◎が1号艇だった時 vs 2号艇だった時 等)
    - 予測確率帯別(◎の勝率予想が高いほど回収率は下がりやすいか)
    - 競艇場別
    - レースグレード別(SG/G1/G2/G3/一般)
の4軸それぞれで、本命3連単1点(毎回100円)の回収率を分解して表示する。

狙い: オッズデータが無くても、「当たっても配当が低くて損しやすいパターン」を
過去の実績から浮かび上がらせ、賭け方のフィルタ(除外ルール)を作る材料にする。

使い方:
    python analyze_ev_patterns.py --db boatrace.db
"""
import argparse
import sqlite3
from collections import defaultdict

STAKE_PER_BET = 100
TRIFECTA_BET_TYPE = "trifecta"  # analyze_roi.pyで自動検出した実際のキー名に合わせて変更可


def _fetch_rows(conn: sqlite3.Connection):
    """prediction_log + races + results + payouts を突き合わせて、分析に必要な列を1行ずつ返す。
    1レースにつきprediction_logの記録が複数(スナップショット)ある場合は最後の1件を使う。
    """
    rows = conn.execute(
        """
        SELECT
            pl.race_id, pl.top_lane, pl.top_pct, pl.top_bet_combo, pl.computed_at,
            r.stadium_number, r.race_grade
        FROM prediction_log pl
        JOIN races r ON r.race_id = pl.race_id
        ORDER BY pl.race_id, pl.computed_at
        """
    ).fetchall()

    # race_idごとに最後のスナップショットだけ残す
    latest_by_race = {}
    for race_id, top_lane, top_pct, top_bet_combo, computed_at, stadium_number, race_grade in rows:
        latest_by_race[race_id] = (top_lane, top_pct, top_bet_combo, stadium_number, race_grade)

    results = []
    for race_id, (top_lane, top_pct, top_bet_combo, stadium_number, race_grade) in latest_by_race.items():
        if top_lane is None or top_pct is None or top_bet_combo is None:
            continue
        result_rows = conn.execute(
            """SELECT e.boat_number, r.arrival_order
               FROM entries e JOIN results r ON r.entry_id = e.entry_id
               WHERE e.race_id = ? ORDER BY e.boat_number""",
            (race_id,),
        ).fetchall()
        if len(result_rows) != 6 or any(row[1] is None for row in result_rows):
            continue  # 結果未確定

        actual_order = sorted(result_rows, key=lambda row: row[1])
        actual_combo = "-".join(str(row[0]) for row in actual_order[:3])

        payout_row = conn.execute(
            f"""SELECT payout_yen FROM payouts
                WHERE race_id=? AND bet_type=? AND combination=? AND payout_yen IS NOT NULL""",
            (race_id, TRIFECTA_BET_TYPE, actual_combo),
        ).fetchone()
        if payout_row is None:
            continue  # 払戻データがまだ無い

        payout_yen = payout_row[0]
        hit = top_bet_combo == actual_combo
        results.append({
            "top_lane": top_lane,
            "top_pct": top_pct,
            "stadium_number": stadium_number,
            "race_grade": race_grade or "一般",
            "hit": hit,
            "payout_yen": payout_yen if hit else 0,
        })
    return results


def _summarize_by(rows: list, key_func, label_func=None):
    """key_func(row)でグルーピングし、件数・的中率・回収率を集計して表示する。"""
    groups = defaultdict(lambda: [0, 0, 0])  # [n, hits, payout_total]
    for row in rows:
        key = key_func(row)
        g = groups[key]
        g[0] += 1
        if row["hit"]:
            g[1] += 1
            g[2] += row["payout_yen"]

    for key in sorted(groups.keys(), key=lambda k: str(k)):
        n, hits, payout_total = groups[key]
        stake = n * STAKE_PER_BET
        roi = payout_total / stake * 100 if stake else 0
        hit_rate = hits / n * 100 if n else 0
        label = label_func(key) if label_func else str(key)
        print(f"  {label:<12}: {n:>5}件 / 的中{hits:>4}件({hit_rate:>5.1f}%) "
              f"/ 賭け金{stake:>8,}円 / 払戻{payout_total:>9,}円 / 回収率{roi:>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    rows = _fetch_rows(conn)
    conn.close()

    print(f"分析対象(払戻データが取れた予想ログ): {len(rows)}件\n")
    if not rows:
        print("対象データがありません。")
        return

    print("■ 号艇別(◎が何号艇だったか)")
    _summarize_by(rows, key_func=lambda r: r["top_lane"], label_func=lambda k: f"{k}号艇")

    print("\n■ 予測確率帯別(◎の予想確率、10%刻み)")
    def pct_bucket(r):
        p = r["top_pct"]
        lower = (int(p) // 10) * 10
        return lower
    _summarize_by(rows, key_func=pct_bucket, label_func=lambda k: f"{k}-{k+9}%")

    print("\n■ レースグレード別")
    _summarize_by(rows, key_func=lambda r: r["race_grade"])

    print("\n■ 競艇場別(stadium_number)")
    _summarize_by(rows, key_func=lambda r: r["stadium_number"], label_func=lambda k: f"場{k}")


if __name__ == "__main__":
    main()
