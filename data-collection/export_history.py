"""
結果が確定済みのレースについて、現在のロジックで予想を再現し、実際の結果と
突き合わせた「履歴」をJSONで出力するスクリプト(history.html用)。

backtest.py が集計値(的中率)だけを出すのに対し、こちらはレース単位の
明細(◎予想 vs 実際の1着、的中/不的中)をそのまま書き出す。

注意: backtest.pyと同様、「今のロジックで過去を再予想」しているため、
その日その時点で実際に表示されていた予想そのものではない(参考値)。

使い方:
    python export_history.py --db boatrace.db --days 14 --out data/history.json
"""
import argparse
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from export_today import score_boat, normalize_to_pct, estimate_bets, MARKS, STADIUM_NAMES
from technique_stats import compute_course_technique_rates, compute_racer_nigashi_rate
from ingest import _migrate_add_missing_columns


def build_history(conn: sqlite3.Connection, start: date, end: date, limit: int) -> dict:
    races = conn.execute(
        """SELECT race_id, race_date, stadium_number, race_number, race_grade,
                  wave_height_cm, wind_speed_m
           FROM races
           WHERE race_date BETWEEN ? AND ?
           ORDER BY race_date DESC, stadium_number, race_number""",
        (start.isoformat(), end.isoformat()),
    ).fetchall()

    course_stats_cache = {}
    history = []
    win_hits = 0
    trifecta_hits = 0
    evaluated = 0
    stake_single = 0
    return_single = 0
    stake_multi = 0
    return_multi = 0

    for race_id, race_date_str, stadium_number, race_number, race_grade, wave, wind in races:
        entries = conn.execute(
            """SELECT e.boat_number, e.racer_name, e.racer_registration_number, e.racer_class,
                      e.national_win_rate, e.national_2連率, e.local_win_rate, e.local_2連率,
                      e.motor_2連率, e.boat_hull_2連率, e.flying_count, e.late_count,
                      p.exhibition_time,
                      r.arrival_order
               FROM entries e
               JOIN results r ON r.entry_id = e.entry_id
               LEFT JOIN previews p ON p.entry_id = e.entry_id
               WHERE e.race_id = ? ORDER BY e.boat_number""",
            (race_id,),
        ).fetchall()
        if len(entries) != 6 or any(row[-1] is None for row in entries):
            continue  # 結果が全艇揃っていないレースは除外

        if stadium_number not in course_stats_cache:
            course_stats_cache[stadium_number] = compute_course_technique_rates(conn, stadium_number)
        course_stats = course_stats_cache[stadium_number]

        exh_values = [row[12] for row in entries if row[12]]
        avg_exh = sum(exh_values) / len(exh_values) if exh_values else None
        race_context = {
            "wave_height_cm": wave, "wind_speed_m": wind, "race_grade": race_grade,
            "avg_exhibition_time": avg_exh,
        }

        boat_dicts = []
        for (bn, name, reg_no, racer_class, nat, nat_2r, local, local_2r, motor_2r, hull_2r,
             flying, late, exh, arrival_order) in entries:
            d = {
                "boat_number": bn, "racer_name": name, "racer_class": racer_class,
                "national_win_rate": nat, "national_2連率": nat_2r,
                "local_win_rate": local, "local_2連率": local_2r,
                "motor_2連率": motor_2r, "boat_hull_2連率": hull_2r,
                "flying_count": flying, "late_count": late, "exhibition_time": exh,
            }
            if bn == 1 and reg_no is not None:
                nigashi = compute_racer_nigashi_rate(conn, reg_no)
                if nigashi:
                    d["nigashi_rate"] = nigashi["nigashi_rate"]
            boat_dicts.append((d, arrival_order))

        scores = [score_boat(d, course_stats, race_context) for d, _ in boat_dicts]
        pcts = normalize_to_pct(scores, use_softmax=True)

        ranked = sorted(zip(boat_dicts, pcts), key=lambda x: -x[1])
        top_pick, top_pct = ranked[0]
        top_pick_d, top_pick_arrival = top_pick

        actual_order = sorted(boat_dicts, key=lambda x: x[1])
        actual_winner_d, _ = actual_order[0]
        actual_combo = "-".join(str(d["boat_number"]) for d, _ in actual_order[:3])

        boats_for_bets = [{"lane": d["boat_number"], "pct": pct} for (d, _), pct in zip(boat_dicts, pcts)]
        bet_list = estimate_bets(boats_for_bets, top_n=6)
        predicted_bets = {b["combo"] for b in bet_list}
        top_bet_combo = bet_list[0]["combo"] if bet_list else None
        trifecta_hit = actual_combo in predicted_bets

        evaluated += 1
        win_hit = top_pick_arrival == 1
        if win_hit:
            win_hits += 1
        if trifecta_hit:
            trifecta_hits += 1

        # 実際の3連単払戻金額(取れた場合のみ回収率の計算対象にする)
        payout_row = conn.execute(
            "SELECT payout_yen FROM payouts WHERE race_id=? AND combination=? AND payout_yen IS NOT NULL",
            (race_id, actual_combo),
        ).fetchone()
        race_payout = payout_row[0] if payout_row else None
        if race_payout is not None:
            stake_single += 100
            if top_bet_combo == actual_combo:
                return_single += race_payout
            n_bets = len(bet_list)
            stake_multi += 100 * n_bets
            if trifecta_hit:
                return_multi += race_payout

        history.append({
            "date": race_date_str,
            "stadium": STADIUM_NAMES.get(stadium_number, f"第{stadium_number}場"),
            "raceNumber": race_number,
            "grade": race_grade,
            "predictedTop": {
                "lane": top_pick_d["boat_number"],
                "name": top_pick_d["racer_name"],
                "pct": top_pct,
            },
            "actualWinner": {
                "lane": actual_winner_d["boat_number"],
                "name": actual_winner_d["racer_name"],
            },
            "winHit": win_hit,
            "trifectaHit": trifecta_hit,
            "actualCombo": actual_combo,
            "actualPayout": race_payout,
            "topBetHit": top_bet_combo == actual_combo,
        })

    history.sort(key=lambda h: (h["date"], h["stadium"], h["raceNumber"]), reverse=True)
    history = history[:limit]

    return {
        "generatedAt": date.today().isoformat(),
        "periodStart": start.isoformat(),
        "periodEnd": end.isoformat(),
        "summary": {
            "evaluated": evaluated,
            "winHitRate": win_hits / evaluated if evaluated else None,
            "trifectaHitRate": trifecta_hits / evaluated if evaluated else None,
            "recoveryRateSingle": (return_single / stake_single) if stake_single else None,
            "recoveryRateMulti": (return_multi / stake_multi) if stake_multi else None,
            "stakeSingle": stake_single,
            "returnSingle": return_single,
        },
        "races": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--limit", type=int, default=200, help="出力する最大レース数(新しい順)")
    ap.add_argument("--out", default="data/history.json")
    args = ap.parse_args()

    end = date.today()
    start = end - timedelta(days=args.days)

    conn = sqlite3.connect(args.db)
    _migrate_add_missing_columns(conn)
    result = build_history(conn, start, end, args.limit)
    conn.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"出力しました: {out_path} ({len(result['races'])}レース分)")


if __name__ == "__main__":
    main()
