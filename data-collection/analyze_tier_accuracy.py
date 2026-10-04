"""
prediction_log に記録された race_tier(A〜D)ごとに、実際の的中率・回収率を集計する。

今回の拡張で、以下の指標をA/B/C/D別・期間別(初期/中期/直近の3分割)に出せるようにした:
    - レース数
    - ◎単勝的中率
    - 3連単本命(1点)的中率・回収率
    - 推奨3連単候補(通常4点)のいずれかが的中した率
    - 3連複相当的中率(予測トップ3艇の顔ぶれが、実際の1〜3着の顔ぶれと一致した率)
      ※ 推奨3連単の本命(1位候補)に使われている3艇を「予測トップ3艇」とみなしている
        (prediction_logに6艇全員の確率までは保存していないための近似。本命の組み合わせは
        Plackett-Luceで最も確率の高い並びなので、ほぼ確率上位3艇と一致するはず)

狙い: 「Dは本当に当たらないのか」「Aは本当に信頼できるのか」を、複数の指標・複数の期間で
多面的に検証する。1つの指標・1つの期間だけで判断すると、サンプル数の偏りや期間特有の
運に振り回されやすいため。

使い方:
    python analyze_tier_accuracy.py --db boatrace.db
    python analyze_tier_accuracy.py --db boatrace.db --periods 3   # 期間を3分割(既定)
    python analyze_tier_accuracy.py --db boatrace.db --periods 1   # 期間分割なし(全期間一括)
"""
import argparse
import json
import sqlite3
from collections import defaultdict

STAKE_PER_BET = 100
TRIFECTA_BET_TYPE = "trifecta"


def _fetch_latest_predictions(conn: sqlite3.Connection) -> dict:
    """race_idごとに、prediction_logの最後のスナップショット(+レース日付)を返す。"""
    rows = conn.execute(
        """
        SELECT pl.race_id, pl.race_tier, pl.top_lane, pl.top_bet_combo, pl.top_bets_json,
               pl.computed_at, r.race_date
        FROM prediction_log pl
        JOIN races r ON r.race_id = pl.race_id
        WHERE pl.race_tier IS NOT NULL
        ORDER BY pl.race_id, pl.computed_at
        """
    ).fetchall()

    latest = {}
    for race_id, tier, top_lane, top_bet_combo, top_bets_json, computed_at, race_date in rows:
        latest[race_id] = (tier, top_lane, top_bet_combo, top_bets_json, race_date)
    return latest


def _assign_periods(latest: dict, n_periods: int) -> dict:
    """race_dateの昇順でレースをn_periods個の塊(初期→直近)に等分し、race_id -> 期間番号を返す。"""
    if n_periods <= 1:
        return {race_id: 0 for race_id in latest}
    ordered = sorted(latest.keys(), key=lambda rid: (latest[rid][4], rid))
    n = len(ordered)
    period_of = {}
    for i, race_id in enumerate(ordered):
        period_idx = min(int(i * n_periods / n), n_periods - 1)
        period_of[race_id] = period_idx
    return period_of


def _evaluate_race(conn: sqlite3.Connection, race_id: int, top_lane, top_bet_combo, top_bets_json) -> dict:
    """1レース分の各種指標を計算する。結果未確定ならNoneを返す。"""
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

    candidates = []
    if top_bets_json:
        try:
            candidates = [c.get("combo") for c in json.loads(top_bets_json)]
        except (TypeError, ValueError):
            candidates = []
    if not candidates and top_bet_combo:
        candidates = [top_bet_combo]

    win_hit = top_lane == actual_winner
    bet_hit = top_bet_combo == actual_combo
    bet_hit_all = actual_combo in candidates

    top3_set_hit = None
    if top_bet_combo:
        try:
            predicted_top3_set = frozenset(int(x) for x in top_bet_combo.split("-"))
            top3_set_hit = predicted_top3_set == actual_top3_set
        except ValueError:
            top3_set_hit = None

    payout_row = conn.execute(
        """SELECT payout_yen FROM payouts
           WHERE race_id=? AND bet_type=? AND combination=? AND payout_yen IS NOT NULL""",
        (race_id, TRIFECTA_BET_TYPE, actual_combo),
    ).fetchone()
    payout_yen = payout_row[0] if payout_row else None

    return {
        "win_hit": win_hit, "bet_hit": bet_hit, "bet_hit_all": bet_hit_all,
        "top3_set_hit": top3_set_hit, "n_candidates": len(candidates),
        "payout_yen": payout_yen, "bet_hit_payout": payout_yen if bet_hit else 0,
    }


def _new_stat_row():
    return {
        "n": 0, "win_hits": 0, "bet_hits": 0, "bet_hits_all": 0,
        "top3_set_hits": 0, "top3_set_evaluated": 0, "n_candidates_total": 0,
        "stake_single": 0, "payout_single": 0,
    }


def _print_period(label: str, stats_by_tier: dict):
    print(f"\n=== {label} ===")
    for tier in ["A", "B", "C", "D"]:
        s = stats_by_tier.get(tier)
        if not s or s["n"] == 0:
            print(f"[{tier}] 対象0件")
            continue
        n = s["n"]
        win_rate = s["win_hits"] / n * 100
        bet_rate = s["bet_hits"] / n * 100
        bet_rate_all = s["bet_hits_all"] / n * 100
        top3_rate = (s["top3_set_hits"] / s["top3_set_evaluated"] * 100) if s["top3_set_evaluated"] else None
        avg_candidates = s["n_candidates_total"] / n if n else 0
        roi = s["payout_single"] / s["stake_single"] * 100 if s["stake_single"] else None

        top3_str = f"{top3_rate:.1f}%" if top3_rate is not None else "N/A"
        roi_str = f"{roi:.1f}%" if roi is not None else "N/A"
        print(f"[{tier}] {n:>4}件")
        print(f"      ◎単勝的中率              : {win_rate:>5.1f}%")
        print(f"      3連単本命的中率・回収率   : {bet_rate:>5.1f}% / {roi_str}")
        print(f"      推奨候補いずれか的中率     : {bet_rate_all:>5.1f}%(平均{avg_candidates:.1f}点)")
        print(f"      3連複相当的中率            : {top3_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--periods", type=int, default=3,
                     help="レースをrace_date昇順でいくつの期間に分けるか(既定3=初期/中期/直近)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    latest = _fetch_latest_predictions(conn)
    period_of = _assign_periods(latest, args.periods)

    period_labels = {
        1: ["全期間"],
        2: ["前半", "後半"],
        3: ["初期", "中期", "直近"],
    }.get(args.periods, [f"期間{i+1}" for i in range(args.periods)])

    stats_by_period_tier = defaultdict(lambda: defaultdict(_new_stat_row))
    overall_by_tier = defaultdict(_new_stat_row)

    evaluated_count = 0
    for race_id, (tier, top_lane, top_bet_combo, top_bets_json, race_date) in latest.items():
        metrics = _evaluate_race(conn, race_id, top_lane, top_bet_combo, top_bets_json)
        if metrics is None:
            continue
        evaluated_count += 1
        period_idx = period_of[race_id]

        for bucket in (stats_by_period_tier[period_idx][tier], overall_by_tier[tier]):
            bucket["n"] += 1
            if metrics["win_hit"]:
                bucket["win_hits"] += 1
            if metrics["bet_hit"]:
                bucket["bet_hits"] += 1
            if metrics["bet_hit_all"]:
                bucket["bet_hits_all"] += 1
            if metrics["top3_set_hit"] is not None:
                bucket["top3_set_evaluated"] += 1
                if metrics["top3_set_hit"]:
                    bucket["top3_set_hits"] += 1
            bucket["n_candidates_total"] += metrics["n_candidates"]
            if metrics["payout_yen"] is not None:
                bucket["stake_single"] += STAKE_PER_BET
                bucket["payout_single"] += metrics["bet_hit_payout"]

    conn.close()

    print(f"対象レース数(結果確定分): {evaluated_count}件 / 期間分割数: {args.periods}")

    print("\n######## 全期間合計 ########")
    _print_period("全期間", overall_by_tier)

    if args.periods > 1:
        print("\n\n######## 期間別 ########")
        for i in range(args.periods):
            label = period_labels[i] if i < len(period_labels) else f"期間{i+1}"
            _print_period(label, stats_by_period_tier[i])

    print("\n※ Aが最も的中率・回収率が高く、Dに向かって下がっていれば、"
          "raceTierの判定が実際に機能している証拠になる。")
    print("※ 期間を分けて見ることで、特定の期間だけ良い/悪いという運の影響を切り分けやすくなる。"
          "全期間と期間別で傾向が大きくズレる場合は、サンプル数不足かモデル・閾値の変化を疑うこと。")


if __name__ == "__main__":
    main()
