"""
波高(wave_height_cm)・風速(wind_speed_m)の区分ごとに、実際の的中率を集計し、
confidence.pyのROUGH_WATER_WAVE_CM(現在3cm)・ROUGH_WATER_WIND_MS(現在5m)という
固定閾値に、データ上の根拠があるかどうかを検証する。

見るポイント:
    波高・風速が上がるにつれて的中率が下がっていれば、「荒れ水面だと当てにくい」という
    前提が裏付けられる。どのあたりから明確に悪化し始めるかを見て、今の閾値(3cm・5m)が
    妥当か、もっと緩め/厳しめにすべきかを判断する材料にする。
    逆に、波高・風速と的中率にはっきりした関係が見えなければ、固定ルールとして
    1段階格下げする効果が薄い(= 無駄に"自信あり"を減らしているだけ)可能性がある。

使い方:
    python analyze_rough_water.py --db boatrace.db
"""
import argparse
import sqlite3
from collections import defaultdict

STAKE_PER_BET = 100
TRIFECTA_BET_TYPE = "trifecta"

# 波高の区分(cm)。現在のROUGH_WATER_WAVE_CM=3を境界として含むよう細かめに取る。
WAVE_BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 999)]
# 風速の区分(m)。現在のROUGH_WATER_WIND_MS=5を境界として含む。
WIND_BINS = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 999)]


def _bin_label(value, bins):
    for lo, hi in bins:
        if lo <= value < hi:
            return f"{lo}-{hi if hi < 999 else '∞'}"
    return "不明"


def _fetch_race_rows(conn: sqlite3.Connection):
    """prediction_logの最終スナップショットに、レースの波高・風速を結合して返す。"""
    rows = conn.execute(
        """
        SELECT pl.race_id, pl.top_lane, pl.top_bet_combo, pl.computed_at,
               r.wave_height_cm, r.wind_speed_m
        FROM prediction_log pl
        JOIN races r ON r.race_id = pl.race_id
        ORDER BY pl.race_id, pl.computed_at
        """
    ).fetchall()

    latest = {}
    for race_id, top_lane, top_bet_combo, computed_at, wave, wind in rows:
        latest[race_id] = (top_lane, top_bet_combo, wave, wind)
    return latest


def _evaluate(conn: sqlite3.Connection, race_id: int, top_lane, top_bet_combo):
    result_rows = conn.execute(
        """SELECT e.boat_number, r.arrival_order
           FROM entries e JOIN results r ON r.entry_id = e.entry_id
           WHERE e.race_id = ? ORDER BY e.boat_number""",
        (race_id,),
    ).fetchall()
    if len(result_rows) != 6 or any(row[1] is None for row in result_rows):
        return None

    actual_order = sorted(result_rows, key=lambda row: row[1])
    actual_winner = actual_order[0][0]
    actual_top3_set = frozenset(row[0] for row in actual_order[:3])
    actual_combo = "-".join(str(row[0]) for row in actual_order[:3])

    win_hit = top_lane == actual_winner
    bet_hit = top_bet_combo == actual_combo
    top3_set_hit = None
    if top_bet_combo:
        try:
            predicted_top3_set = frozenset(int(x) for x in top_bet_combo.split("-"))
            top3_set_hit = predicted_top3_set == actual_top3_set
        except ValueError:
            pass

    payout_row = conn.execute(
        """SELECT payout_yen FROM payouts
           WHERE race_id=? AND bet_type=? AND combination=? AND payout_yen IS NOT NULL""",
        (race_id, TRIFECTA_BET_TYPE, actual_combo),
    ).fetchone()
    payout_yen = payout_row[0] if payout_row else None

    return {"win_hit": win_hit, "bet_hit": bet_hit, "top3_set_hit": top3_set_hit, "payout_yen": payout_yen}


def _print_table(title: str, stats: dict, bins: list):
    print(f"\n■ {title}")
    for lo, hi in bins:
        label = f"{lo}-{hi if hi < 999 else '∞'}"
        s = stats.get(label)
        if not s or s["n"] == 0:
            print(f"  {label:>8}: 対象0件")
            continue
        n = s["n"]
        win_rate = s["win_hits"] / n * 100
        bet_rate = s["bet_hits"] / n * 100
        top3_rate = (s["top3_hits"] / s["top3_evaluated"] * 100) if s["top3_evaluated"] else None
        roi = s["payout"] / s["stake"] * 100 if s["stake"] else None
        top3_str = f"{top3_rate:.1f}%" if top3_rate is not None else "N/A"
        roi_str = f"{roi:.1f}%" if roi is not None else "N/A"
        print(f"  {label:>8}: {n:>4}件 / ◎的中率{win_rate:>5.1f}% / "
              f"3連単的中率{bet_rate:>5.1f}%(回収率{roi_str}) / 3連複相当{top3_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    latest = _fetch_race_rows(conn)

    def new_stat():
        return {"n": 0, "win_hits": 0, "bet_hits": 0, "top3_hits": 0, "top3_evaluated": 0,
                "stake": 0, "payout": 0}

    wave_stats = defaultdict(new_stat)
    wind_stats = defaultdict(new_stat)
    evaluated = 0

    for race_id, (top_lane, top_bet_combo, wave, wind) in latest.items():
        metrics = _evaluate(conn, race_id, top_lane, top_bet_combo)
        if metrics is None:
            continue
        evaluated += 1

        for value, bins, stats_dict in [(wave, WAVE_BINS, wave_stats), (wind, WIND_BINS, wind_stats)]:
            if value is None:
                continue
            label = _bin_label(value, bins)
            s = stats_dict[label]
            s["n"] += 1
            if metrics["win_hit"]:
                s["win_hits"] += 1
            if metrics["bet_hit"]:
                s["bet_hits"] += 1
            if metrics["top3_set_hit"] is not None:
                s["top3_evaluated"] += 1
                if metrics["top3_set_hit"]:
                    s["top3_hits"] += 1
            if metrics["payout_yen"] is not None:
                s["stake"] += STAKE_PER_BET
                if metrics["bet_hit"]:
                    s["payout"] += metrics["payout_yen"]

    conn.close()

    print(f"対象レース数(結果確定分): {evaluated}件")
    _print_table("波高別(cm) ※現在の格下げ閾値: 3cm以上", wave_stats, WAVE_BINS)
    _print_table("風速別(m) ※現在の格下げ閾値: 5m以上", wind_stats, WIND_BINS)

    print("\n※ 波高・風速が上がるにつれて的中率・回収率が明確に下がっていれば、"
          "ROUGH_WATER_*の閾値は妥当。関係がはっきりしなければ、閾値の見直しや"
          "ルール自体の撤廃を検討する価値がある。")


if __name__ == "__main__":
    main()
