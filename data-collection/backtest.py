"""
結果が確定済みのレースに対して、現在の予想ロジック(export_today.pyのscore_boat等)を
そのまま当てはめて予想を再現し、実際の結果と突き合わせて的中率を検証するスクリプト。

注意(正直な限界):
- 「その時点でどう予想していたか」を保存していないため、今のロジック・今の
  コース別統計(technique_stats)を使って過去のレースを"再予想"している。
  技術統計は日々更新されるため、当時と全く同じ予想には必ずしもならない。
  とはいえ、今のロジックの妥当性を見る目安にはなる。
- サンプル数が少ないうちは参考程度。目安として最低30〜50レース以上で見たい。

使い方:
    python backtest.py --db boatrace.db --days 1        # 直近1日分
    python backtest.py --db boatrace.db --start 2026-09-01 --end 2026-09-08
"""
import argparse
import sqlite3
from datetime import date, timedelta

from export_today import score_boat, normalize_to_pct, estimate_bets
from technique_stats import compute_course_technique_rates, compute_racer_nigashi_rate


def backtest(conn: sqlite3.Connection, start: date, end: date) -> dict:
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

    for race_id, race_date_str, stadium_number, race_number, race_grade, wave, wind in races:
        entries = conn.execute(
            """SELECT e.entry_id, e.boat_number, e.racer_registration_number,
                      e.national_win_rate, e.national_2連率, e.local_win_rate, e.local_2連率,
                      e.motor_2連率, e.boat_hull_2連率, e.flying_count, e.late_count,
                      r.arrival_order
               FROM entries e
               JOIN results r ON r.entry_id = e.entry_id
               WHERE e.race_id = ? ORDER BY e.boat_number""",
            (race_id,),
        ).fetchall()
        # 6艇分の結果が揃っていないレースは検証対象から除く(棄権等で不完全な場合がある)
        if len(entries) != 6 or any(row[-1] is None for row in entries):
            continue

        if stadium_number not in course_stats_cache:
            course_stats_cache[stadium_number] = compute_course_technique_rates(conn, stadium_number)
        course_stats = course_stats_cache[stadium_number]
        race_context = {"wave_height_cm": wave, "wind_speed_m": wind, "race_grade": race_grade}

        boat_dicts = []
        for (entry_id, bn, reg_no, nat, nat_2r, local, local_2r, motor_2r, hull_2r,
             flying, late, arrival_order) in entries:
            d = {
                "boat_number": bn, "national_win_rate": nat, "national_2連率": nat_2r,
                "local_win_rate": local, "local_2連率": local_2r,
                "motor_2連率": motor_2r, "boat_hull_2連率": hull_2r,
                "flying_count": flying, "late_count": late,
            }
            if bn == 1 and reg_no is not None:
                nigashi = compute_racer_nigashi_rate(conn, reg_no)
                if nigashi:
                    d["nigashi_rate"] = nigashi["nigashi_rate"]
            boat_dicts.append((d, arrival_order))

        scores = [score_boat(d, course_stats, race_context) for d, _ in boat_dicts]
        pcts = normalize_to_pct(scores, use_softmax=True)

        boats_for_bets = [
            {"lane": d["boat_number"], "pct": pct}
            for (d, _), pct in zip(boat_dicts, pcts)
        ]
        predicted_bets = {b["combo"] for b in estimate_bets(boats_for_bets, top_n=6)}

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

    return {
        "evaluated": evaluated,
        "win_hit_rate": win_hits / evaluated if evaluated else None,
        "place_hit_rate": place_hits / evaluated if evaluated else None,
        "trifecta_hit_rate": trifecta_hits / evaluated if evaluated else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=1, help="--start/--end を省略した場合、直近何日分を検証するか")
    args = ap.parse_args()

    if args.start and args.end:
        start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    else:
        end = date.today()
        start = end - timedelta(days=args.days)

    conn = sqlite3.connect(args.db)
    result = backtest(conn, start, end)
    conn.close()

    print(f"検証期間: {start} 〜 {end}")
    print(f"検証対象レース数: {result['evaluated']}")
    if result["evaluated"] == 0:
        print("結果が確定しているレースが見つかりませんでした。")
        return
    print(f"◎の単勝的中率(1着): {result['win_hit_rate']:.1%}")
    print(f"◎の複勝的中率(2着以内): {result['place_hit_rate']:.1%}")
    print(f"推奨3連単に実際の着順が含まれていた率: {result['trifecta_hit_rate']:.1%}")
    if result["evaluated"] < 30:
        print("⚠ サンプル数がまだ少ないので、参考程度に見てください。")


if __name__ == "__main__":
    main()
