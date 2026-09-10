"""
実際に記録された予想ログ(prediction_log)と結果を突き合わせて、履歴JSONを出力する
スクリプト(history.html用)。

重要: これは「今のロジックで過去を再計算」するbacktest.pyとは違い、
export_today.py が実行されるたびに記録している「その時点で実際に出していた予想」
(prediction_log テーブル)を使う。なので本当の意味での「履歴」になる。

ただし、prediction_log は導入時点から記録を開始したものなので、
それより前のレースは履歴に出てこない(記録が無いため)。

使い方:
    python export_history.py --db boatrace.db --days 14 --out data/history.json
"""
import argparse
import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from export_today import STADIUM_NAMES
from ingest import init_db


def _parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            return datetime.fromisoformat(s.replace(" ", "T"))
        except ValueError:
            return None


def choose_snapshot(snapshots: list, close_at: str) -> tuple:
    """締切時刻以前の最後のスナップショットを採用する。無ければ記録上最後のものを使う。"""
    close_dt = _parse_dt(close_at)
    if close_dt:
        candidates = [s for s in snapshots if (_parse_dt(s[0]) or datetime.max) <= close_dt]
        if candidates:
            return candidates[-1]
    return snapshots[-1]


def build_history(conn: sqlite3.Connection, start: date, end: date, limit: int) -> dict:
    races = conn.execute(
        """SELECT race_id, race_date, stadium_number, race_number, race_grade, close_at
           FROM races WHERE race_date BETWEEN ? AND ?
           ORDER BY race_date DESC, stadium_number, race_number""",
        (start.isoformat(), end.isoformat()),
    ).fetchall()

    history = []
    win_hits = 0
    bet_hits = 0
    evaluated = 0
    stake = 0
    payout_return = 0
    stake_all = 0
    payout_return_all = 0

    for race_id, race_date_str, stadium_number, race_number, race_grade, close_at in races:
        snapshots = conn.execute(
            """SELECT computed_at, top_lane, top_pct, top_bet_combo, is_confident, top_bets_json
               FROM prediction_log WHERE race_id = ? ORDER BY computed_at""",
            (race_id,),
        ).fetchall()
        if not snapshots:
            continue  # このレースは記録開始前に終わっていた(または未記録)

        (computed_at, top_lane, top_pct, top_bet_combo,
         is_confident, top_bets_json) = choose_snapshot(snapshots, close_at)

        results = conn.execute(
            """SELECT e.boat_number, e.racer_name, r.arrival_order
               FROM entries e JOIN results r ON r.entry_id = e.entry_id
               WHERE e.race_id = ? ORDER BY e.boat_number""",
            (race_id,),
        ).fetchall()
        if len(results) != 6 or any(row[2] is None for row in results):
            continue  # 結果がまだ確定していない

        actual_order = sorted(results, key=lambda row: row[2])
        actual_winner_lane, actual_winner_name, _ = actual_order[0]
        actual_combo = "-".join(str(row[0]) for row in actual_order[:3])

        top_lane_name = next((name for lane, name, _ in results if lane == top_lane), None)

        win_hit = top_lane == actual_winner_lane
        bet_hit = top_bet_combo == actual_combo

        evaluated += 1
        if win_hit:
            win_hits += 1
        if bet_hit:
            bet_hits += 1

        # 上位候補(通常4件)を、各候補が実際に的中したかどうか付きで展開する。
        # top_bets_json が無い古い記録(この機能を追加する前のレース)は、
        # 従来の本命1件だけにフォールバックする。
        predicted_bet_combos = []
        if top_bets_json:
            try:
                candidates = json.loads(top_bets_json)
            except (TypeError, ValueError):
                candidates = []
            for c in candidates:
                combo = c.get("combo")
                predicted_bet_combos.append({
                    "combo": combo,
                    "prob": c.get("prob"),
                    "hit": combo == actual_combo,
                })
        if not predicted_bet_combos and top_bet_combo:
            predicted_bet_combos.append({
                "combo": top_bet_combo,
                "prob": None,
                "hit": bet_hit,
            })

        payout_row = conn.execute(
            "SELECT payout_yen FROM payouts WHERE race_id=? AND combination=? AND payout_yen IS NOT NULL",
            (race_id, actual_combo),
        ).fetchone()
        race_payout = payout_row[0] if payout_row else None
        if race_payout is not None:
            # 本命1点(100円)の回収率
            stake += 100
            if bet_hit:
                payout_return += race_payout

            # 上位候補すべてを均等(各100円)買いした場合の回収率。
            # 候補のどれか1つでも実際の組み合わせと一致すれば、その点だけ払戻を受け取る。
            n_candidates = len(predicted_bet_combos) if predicted_bet_combos else 1
            stake_all += 100 * n_candidates
            if any(c["hit"] for c in predicted_bet_combos):
                payout_return_all += race_payout

        history.append({
            "date": race_date_str,
            "stadium": STADIUM_NAMES.get(stadium_number, f"第{stadium_number}場"),
            "raceNumber": race_number,
            "grade": race_grade,
            "predictedAt": computed_at,
            "predictedTop": {"lane": top_lane, "name": top_lane_name, "pct": top_pct},
            "predictedBetCombo": top_bet_combo,
            "predictedBetCombos": predicted_bet_combos,
            "wasConfident": bool(is_confident),
            "actualWinner": {"lane": actual_winner_lane, "name": actual_winner_name},
            "actualCombo": actual_combo,
            "winHit": win_hit,
            "betHit": bet_hit,
            "actualPayout": race_payout,
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
            "betHitRate": bet_hits / evaluated if evaluated else None,
            "recoveryRate": (payout_return / stake) if stake else None,
            "stake": stake,
            "payoutReturn": payout_return,
            "recoveryRateAll": (payout_return_all / stake_all) if stake_all else None,
            "stakeAll": stake_all,
            "payoutReturnAll": payout_return_all,
        },
        "races": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--days", type=int, default=90,
                     help="何日分さかのぼって出力するか(history.html側で7/14/30/90日/全期間の切り替えUIがあるため、余裕を持って90日を既定値にしている)")
    ap.add_argument("--limit", type=int, default=1000, help="出力する最大レース数(新しい順)")
    ap.add_argument("--out", default="data/history.json")
    args = ap.parse_args()

    end = date.today()
    start = end - timedelta(days=args.days)

    conn = init_db(args.db)  # prediction_logテーブルが無い古いDBでも確実に用意する
    result = build_history(conn, start, end, args.limit)
    conn.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"出力しました: {out_path} ({len(result['races'])}レース分)")


if __name__ == "__main__":
    main()
