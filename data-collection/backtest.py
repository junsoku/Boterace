"""
結果が確定済みのレースに対して、現在の予想ロジック(export_today.pyのpredict_with_model等)を
そのまま当てはめて予想を再現し、実際の結果と突き合わせて的中率を検証するスクリプト。

位置づけ(history.htmlとの違い):
- これは「今のロジックを過去のレースに当てはめたら、どのくらい当たるか」という
  "モデルの現在の実力チェック"用のツール。実際にその時点で表示していた予想とは異なる。
- 「実際にその時点で表示していた予想」を振り返りたい場合は history.html /
  export_history.py (prediction_logテーブルに基づく本当の履歴)を使うこと。

注意(正直な限界):
- 技術統計(technique_stats)は日々更新されるため、過去のレース当時とは
  条件が変わっている場合がある。とはいえ、今のロジックの妥当性を見る目安にはなる。
- サンプル数が少ないうちは参考程度。目安として最低30〜50レース以上で見たい。
- --model を指定しない場合はヒューリスティック(score_boat)で検証する。本番と同じ条件で
  検証したい場合は、必ず --model data-collection/model.txt のように指定すること。

使い方:
    python backtest.py --db boatrace.db --days 1                       # ヒューリスティックで検証
    python backtest.py --db boatrace.db --days 7 --model model.txt     # MLモデル(ランク学習)で検証
    python backtest.py --db boatrace.db --start 2026-09-01 --end 2026-09-08
"""
import argparse
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from export_today import (
    score_boat, normalize_to_pct, estimate_bets, estimate_bets_ml, predict_with_model,
)
from technique_stats import compute_course_technique_rates, compute_racer_nigashi_rate
from ingest import _migrate_add_missing_columns


def backtest(conn: sqlite3.Connection, start: date, end: date, model=None) -> dict:
    use_ml = model is not None

    races = conn.execute(
        """SELECT race_id, race_date, stadium_number, race_number, race_grade,
                  wave_height_cm, wind_speed_m
           FROM races
           WHERE race_date BETWEEN ? AND ?
           ORDER BY race_date, stadium_number, race_number""",
        (start.isoformat(), end.isoformat()),
    ).fetchall()

    course_stats_cache = {}
    win_hits = 0          # ◎(予想1位)が実際に1着だった回数
    place_hits = 0        # ◎が実際に2着以内だった回数
    trifecta_hits = 0     # 実際の1-2-3着が推奨3連単リストに入っていた回数
    evaluated = 0

    # 回収率計算用(実際の払戻金額が取れたレースのみ集計対象にする)
    stake_single = 0      # 単一の本命3連単だけを100円ずつ買った場合の投資額
    return_single = 0     # 同、払戻額
    stake_multi = 0       # 推奨3連単を全部(最大6点)100円ずつ買った場合の投資額
    return_multi = 0      # 同、払戻額
    payout_data_races = 0 # 払戻データが取れたレース数

    for race_id, race_date_str, stadium_number, race_number, race_grade, wave, wind in races:
        # predict_with_model(MLモデル)が必要とする全特徴量を取得する。
        # ヒューリスティック(score_boat)しか使わない場合でも、同じクエリで揃えておけば
        # --model の有無を後から切り替えても困らない。
        entries = conn.execute(
            """SELECT e.entry_id, e.boat_number, e.racer_registration_number, e.racer_class,
                      e.national_win_rate, e.national_2連率, e.local_win_rate, e.local_2連率,
                      e.motor_2連率, e.boat_hull_2連率, e.average_start_timing,
                      e.flying_count, e.late_count,
                      p.exhibition_time, p.tilt_angle, p.weight_adjustment_kg, p.start_timing_preview,
                      r.arrival_order
               FROM entries e
               JOIN results r ON r.entry_id = e.entry_id
               LEFT JOIN previews p ON p.entry_id = e.entry_id
               WHERE e.race_id = ? ORDER BY e.boat_number""",
            (race_id,),
        ).fetchall()
        # 6艇分の結果が揃っていないレースは検証対象から除く(棄権等で不完全な場合がある)
        if len(entries) != 6 or any(row[-1] is None for row in entries):
            continue

        if stadium_number not in course_stats_cache:
            course_stats_cache[stadium_number] = compute_course_technique_rates(conn, stadium_number)
        course_stats = course_stats_cache[stadium_number]

        exh_values = [row[13] for row in entries if row[13]]
        avg_exh = sum(exh_values) / len(exh_values) if exh_values else None
        race_context = {"wave_height_cm": wave, "wind_speed_m": wind, "race_grade": race_grade,
                         "avg_exhibition_time": avg_exh}

        boat_dicts = []
        for (entry_id, bn, reg_no, racer_class, nat, nat_2r, local, local_2r, motor_2r, hull_2r,
             avg_st, flying, late, exh, tilt_angle, weight_adj, start_timing_prev,
             arrival_order) in entries:
            d = {
                "boat_number": bn, "racer_class": racer_class,
                "national_win_rate": nat, "national_2連率": nat_2r,
                "local_win_rate": local, "local_2連率": local_2r,
                "motor_2連率": motor_2r, "boat_hull_2連率": hull_2r,
                "average_start_timing": avg_st,
                "flying_count": flying, "late_count": late,
                "exhibition_time": exh, "tilt_angle": tilt_angle,
                "weight_adjustment_kg": weight_adj,
                "start_timing_preview": start_timing_prev,
            }
            if bn == 1 and reg_no is not None:
                nigashi = compute_racer_nigashi_rate(conn, reg_no)
                if nigashi:
                    d["nigashi_rate"] = nigashi["nigashi_rate"]
            boat_dicts.append((d, arrival_order))

        scores_by_lane = None
        if use_ml:
            scores = predict_with_model([d for d, _ in boat_dicts], course_stats, model, race_context)
            pcts = normalize_to_pct(scores, use_softmax=True)
            scores_by_lane = {d["boat_number"]: s for (d, _), s in zip(boat_dicts, scores)}
        else:
            scores = [score_boat(d, course_stats, race_context) for d, _ in boat_dicts]
            pcts = normalize_to_pct(scores, use_softmax=True)

        boats_for_bets = [
            {"lane": d["boat_number"], "pct": pct}
            for (d, _), pct in zip(boat_dicts, pcts)
        ]

        if scores_by_lane is not None:
            bet_list = estimate_bets_ml([d for d, _ in boat_dicts], scores_by_lane, top_n=6)
        else:
            bet_list = estimate_bets(boats_for_bets, top_n=6)
        predicted_bets = {b["combo"] for b in bet_list}
        top_bet_combo = bet_list[0]["combo"] if bet_list else None

        # ◎ = 予想確率が最も高い艇
        ranked = sorted(zip(boat_dicts, pcts), key=lambda x: -x[1])
        top_pick_lane = ranked[0][0][0]["boat_number"]
        top_pick_arrival = ranked[0][0][1]

        actual_order = sorted(boat_dicts, key=lambda x: x[1])  # arrival_orderでソート
        actual_combo = "-".join(str(d["boat_number"]) for d, order in actual_order[:3])

        evaluated += 1
        if top_pick_arrival == 1:
            win_hits += 1
        if top_pick_arrival in (1, 2):
            place_hits += 1
        if actual_combo in predicted_bets:
            trifecta_hits += 1

        # 実際の3連単払戻金額を取得(この組み合わせの払戻が見つかった場合のみ回収率の計算対象にする)。
        # payoutsは(race_id, bet_type, combination)でユニークなので、bet_type='trifecta'を
        # 指定しないと同じcombination文字列を持つ他の賭式(3連複等)の払戻を誤って拾う恐れがある。
        payout_row = conn.execute(
            """SELECT payout_yen FROM payouts
               WHERE race_id=? AND bet_type='trifecta' AND combination=? AND payout_yen IS NOT NULL""",
            (race_id, actual_combo),
        ).fetchone()
        if payout_row:
            payout_data_races += 1
            actual_payout = payout_row[0]

            stake_single += 100
            if top_bet_combo == actual_combo:
                return_single += actual_payout

            n_bets = len(bet_list)
            stake_multi += 100 * n_bets
            if actual_combo in predicted_bets:
                return_multi += actual_payout

    def rate(n, d):
        return n / d if d else None

    return {
        "evaluated": evaluated,
        "win_hit_rate": win_hits / evaluated if evaluated else None,
        "place_hit_rate": place_hits / evaluated if evaluated else None,
        "trifecta_hit_rate": trifecta_hits / evaluated if evaluated else None,
        "payout_data_races": payout_data_races,
        "recovery_rate_single": rate(return_single, stake_single),
        "recovery_rate_multi": rate(return_multi, stake_multi),
        "stake_single": stake_single,
        "return_single": return_single,
        "stake_multi": stake_multi,
        "return_multi": return_multi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=1, help="--start/--end を省略した場合、直近何日分を検証するか")
    ap.add_argument("--model", default=None,
                     help="train_model.py で作ったmodel.txt(ランク学習)のパス(省略時はヒューリスティックで検証)。")
    args = ap.parse_args()

    model = None
    if args.model and Path(args.model).exists():
        import lightgbm as lgb
        model = lgb.Booster(model_file=args.model)
        print(f"MLモデル(ランク学習)を読み込みました: {args.model}")
    elif args.model:
        print(f"⚠ 指定されたモデルファイルが見つかりません: {args.model}(ヒューリスティックで検証します)")
    else:
        print("ℹ --model が指定されていないため、ヒューリスティック(score_boat)で検証します。"
              "本番のMLモデルと同じ条件で検証したい場合は --model を指定してください。")

    if args.start and args.end:
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    else:
        end = date.today()
        start = end - timedelta(days=args.days)

    conn = sqlite3.connect(args.db)
    _migrate_add_missing_columns(conn)  # flying_count等の列がまだなければここで追加する
    result = backtest(conn, start, end, model=model)
    conn.close()

    print(f"\n検証期間: {start} 〜 {end}")
    print(f"検証対象レース数: {result['evaluated']}")
    if result["evaluated"] == 0:
        print("結果が確定しているレースが見つかりませんでした。")
        return
    print(f"◎の単勝的中率(1着): {result['win_hit_rate']:.1%}")
    print(f"◎の複勝的中率(2着以内): {result['place_hit_rate']:.1%}")
    print(f"推奨3連単に実際の着順が含まれていた率: {result['trifecta_hit_rate']:.1%}")
    print(f"払戻データが取れたレース数: {result['payout_data_races']}")
    if result["payout_data_races"] == 0:
        print("払戻データがまだ無いため、回収率は計算できませんでした。")
    else:
        if result["recovery_rate_single"] is not None:
            print(f"回収率(本命1点のみ・毎回100円): {result['recovery_rate_single']:.1%} "
                  f"(投資{result['stake_single']}円 → 払戻{result['return_single']}円)")
        if result["recovery_rate_multi"] is not None:
            print(f"回収率(推奨3連単を全点・毎回100円ずつ): {result['recovery_rate_multi']:.1%} "
                  f"(投資{result['stake_multi']}円 → 払戻{result['return_multi']}円)")
    if result["evaluated"] < 30:
        print("⚠ サンプル数がまだ少ないので、参考程度に見てください。")


if __name__ == "__main__":
    main()
