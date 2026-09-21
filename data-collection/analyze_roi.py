"""
prediction_log(過去の予想ログ)と payouts(実際の払戻金)を突き合わせて、
「自信あり」判定だった予想だけに絞った場合の回収率を集計する。

回収率 = 払戻金合計 ÷ 賭け金合計 × 100(%)
  100%を超えていれば黒字、下回っていれば赤字(100円賭けたら平均いくら戻ってきたか)。
  賭け金は1点100円と仮定(公式の最低購入単位)。

集計対象は2種類:
  1. ◎(単勝的中)の自信あり (prediction_log.is_confident)
  2. 推奨3連単(本命1点)の自信あり (prediction_log.bet_is_confident)

payouts.bet_type の実際のキー名(API仕様依存)は環境によって違う可能性があるため、
起動時に DISTINCT な bet_type 一覧を表示する。「単勝」に対応するキーが
候補リストに見つからない場合は、表示されたキー名を見て --win-bet-type で指定すること。

使い方:
    python analyze_roi.py --db boatrace.db
    python analyze_roi.py --db boatrace.db --win-bet-type win --trifecta-bet-type trifecta
"""
import argparse
import sqlite3

# よくあるキー名の候補(実際のDBの値と照合して使う。無ければ --win-bet-type 等で明示指定)
WIN_BET_TYPE_CANDIDATES = ["win", "tansho", "tan"]
TRIFECTA_BET_TYPE_CANDIDATES = ["trifecta", "sanrentan", "3tan"]

STAKE_PER_BET = 100  # 1点あたりの賭け金(円)。公式の最低購入単位に合わせた仮定。


def _detect_bet_type(conn: sqlite3.Connection, candidates: list) -> str | None:
    rows = conn.execute("SELECT DISTINCT bet_type FROM payouts").fetchall()
    existing = {r[0] for r in rows}
    for c in candidates:
        if c in existing:
            return c
    return None


def _lookup_payout(conn: sqlite3.Connection, race_id: int, bet_type: str, combination: str) -> int:
    row = conn.execute(
        "SELECT payout_yen FROM payouts WHERE race_id=? AND bet_type=? AND combination=?",
        (race_id, bet_type, combination),
    ).fetchone()
    return row[0] if row else 0  # 該当が無い(=不的中)なら払戻0円


def _summarize(label: str, n: int, hits: int, total_payout: int):
    if n == 0:
        print(f"{label}: 対象0件(データなし)")
        return
    stake = n * STAKE_PER_BET
    roi = total_payout / stake * 100
    hit_rate = hits / n * 100
    print(f"{label}: {n}件 / 的中{hits}件({hit_rate:.1f}%) / "
          f"賭け金{stake:,}円 / 払戻{total_payout:,}円 / 回収率{roi:.1f}%")


def analyze(conn: sqlite3.Connection, win_bet_type: str, trifecta_bet_type: str):
    logs = conn.execute(
        """SELECT log_id, race_id, top_lane, top_bet_combo, is_confident, bet_is_confident
           FROM prediction_log"""
    ).fetchall()

    # ---- ◎(単勝)の回収率: 自信あり / 自信なし ----
    win_groups = {True: [0, 0, 0], False: [0, 0, 0]}  # [n, hits, total_payout]
    # ---- 推奨3連単(本命)の回収率: 自信あり / 自信なし ----
    bet_groups = {True: [0, 0, 0], False: [0, 0, 0]}

    for log_id, race_id, top_lane, top_bet_combo, is_confident, bet_is_confident in logs:
        if top_lane is not None and is_confident is not None and win_bet_type:
            payout = _lookup_payout(conn, race_id, win_bet_type, str(top_lane))
            g = win_groups[bool(is_confident)]
            g[0] += 1
            g[2] += payout
            if payout > 0:
                g[1] += 1

        if top_bet_combo and bet_is_confident is not None and trifecta_bet_type:
            payout = _lookup_payout(conn, race_id, trifecta_bet_type, top_bet_combo)
            g = bet_groups[bool(bet_is_confident)]
            g[0] += 1
            g[2] += payout
            if payout > 0:
                g[1] += 1

    if win_bet_type:
        print("\n■ ◎(単勝)の回収率")
        _summarize("  自信ありのみ", *win_groups[True])
        _summarize("  自信なし込み全体", *[a + b for a, b in zip(win_groups[True], win_groups[False])])
    else:
        print("\n■ ◎(単勝)の回収率: payoutsにwin(単勝)らしきbet_typeが見つからないためスキップ")

    if trifecta_bet_type:
        print("\n■ 推奨3連単(本命1点)の回収率")
        _summarize("  自信ありのみ", *bet_groups[True])
        _summarize("  自信なし込み全体", *[a + b for a, b in zip(bet_groups[True], bet_groups[False])])
    else:
        print("\n■ 推奨3連単の回収率: payoutsにtrifectaらしきbet_typeが見つからないためスキップ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--win-bet-type", default=None, help="payoutsテーブルの単勝に対応するbet_type名(自動検出できない場合に指定)")
    ap.add_argument("--trifecta-bet-type", default=None, help="payoutsテーブルの3連単に対応するbet_type名(自動検出できない場合に指定)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    existing_types = [r[0] for r in conn.execute("SELECT DISTINCT bet_type FROM payouts").fetchall()]
    print(f"payoutsに存在するbet_type一覧: {existing_types}")

    win_bet_type = args.win_bet_type or _detect_bet_type(conn, WIN_BET_TYPE_CANDIDATES)
    trifecta_bet_type = args.trifecta_bet_type or _detect_bet_type(conn, TRIFECTA_BET_TYPE_CANDIDATES)
    print(f"単勝として使うbet_type: {win_bet_type or '(見つからず)'}")
    print(f"3連単として使うbet_type: {trifecta_bet_type or '(見つからず)'}")

    analyze(conn, win_bet_type, trifecta_bet_type)
    conn.close()


if __name__ == "__main__":
    main()
