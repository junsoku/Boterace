"""
prediction_log に記録された race_tier(A〜D、classify_race_tier()の予想時点のスナップショット)ごとに、
実際の的中率・回収率を集計する。

狙い: 「Dは本当に当たらないのか」「Aは本当に信頼できるのか」を、後から実データで検証する。
このスクリプトの結果次第で、将来的に「Dは買い目を出さない(見送り)」に踏み切るかどうかを判断する。

使い方:
    python analyze_tier_accuracy.py --db boatrace.db
"""
import argparse
import sqlite3
from collections import defaultdict

STAKE_PER_BET = 100
TRIFECTA_BET_TYPE = "trifecta"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    rows = conn.execute(
        """
        SELECT pl.race_id, pl.race_tier, pl.top_lane, pl.top_bet_combo, pl.computed_at
        FROM prediction_log pl
        WHERE pl.race_tier IS NOT NULL
        ORDER BY pl.race_id, pl.computed_at
        """
    ).fetchall()

    # race_idごとに最後のスナップショット(締切直前の最終予想)だけを使う
    latest_by_race = {}
    for race_id, tier, top_lane, top_bet_combo, computed_at in rows:
        latest_by_race[race_id] = (tier, top_lane, top_bet_combo)

    # tier -> [n, win_hits, bet_hits, stake, payout_total]
    stats = defaultdict(lambda: [0, 0, 0, 0, 0])

    for race_id, (tier, top_lane, top_bet_combo) in latest_by_race.items():
        result_rows = conn.execute(
            """SELECT e.boat_number, r.arrival_order
               FROM entries e JOIN results r ON r.entry_id = e.entry_id
               WHERE e.race_id = ? ORDER BY e.boat_number""",
            (race_id,),
        ).fetchall()
        if len(result_rows) != 6 or any(row[1] is None for row in result_rows):
            continue  # 結果未確定

        actual_order = sorted(result_rows, key=lambda row: row[1])
        actual_winner = actual_order[0][0]
        actual_combo = "-".join(str(row[0]) for row in actual_order[:3])

        s = stats[tier]
        s[0] += 1
        win_hit = top_lane == actual_winner
        bet_hit = top_bet_combo == actual_combo
        if win_hit:
            s[1] += 1
        if bet_hit:
            s[2] += 1

        payout_row = conn.execute(
            """SELECT payout_yen FROM payouts
               WHERE race_id=? AND bet_type=? AND combination=? AND payout_yen IS NOT NULL""",
            (race_id, TRIFECTA_BET_TYPE, actual_combo),
        ).fetchone()
        if payout_row is not None:
            s[3] += STAKE_PER_BET
            if bet_hit:
                s[4] += payout_row[0]

    conn.close()

    print("ティア別の実績(prediction_logの最終スナップショット・結果確定分のみ)\n")
    for tier in ["A", "B", "C", "D"]:
        n, win_hits, bet_hits, stake, payout_total = stats.get(tier, [0, 0, 0, 0, 0])
        if n == 0:
            print(f"[{tier}] 対象0件")
            continue
        win_rate = win_hits / n * 100
        bet_rate = bet_hits / n * 100
        roi = payout_total / stake * 100 if stake else None
        roi_str = f"{roi:.1f}%" if roi is not None else "N/A(払戻データ無し)"
        print(f"[{tier}] {n:>4}件 / ◎的中率{win_rate:>5.1f}% / 3連単本命的中率{bet_rate:>5.1f}% "
              f"/ 3連単回収率{roi_str}")

    print("\n※ Aが最も的中率・回収率が高く、Dに向かって下がっていれば、"
          "raceTierの判定が実際に機能している証拠になる。")


if __name__ == "__main__":
    main()
