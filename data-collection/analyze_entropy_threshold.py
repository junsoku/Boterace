"""
prediction_log.race_entropy(6艇の予測確率分布のエントロピー、0=堅い・1=大混戦)の
実際の分布を可視化し、confidence.pyのRACE_ENTROPY_DOWNGRADE_THRESHOLD(現在0.85)が
妥当な頻度で発動しているか、そして実際に「エントロピーが高いレースは当てにくい」と
言えるのかを検証する。

狙い:
    1. 閾値0.85が、全レースのうちどれくらいの割合で発動しているか
       (想定よりずっと高い頻度で発動していれば、Aがなかなか出ない一因になっている)
    2. エントロピー帯ごとに、実際の◎的中率がどう変わるか
       (本当に「エントロピーが高い=当てにくい」という前提が正しいかを確認する。
       正しくなければ、格下げルール自体の妥当性を見直す必要がある)

使い方:
    python analyze_entropy_threshold.py --db boatrace.db
"""
import argparse
import sqlite3
from collections import defaultdict

ENTROPY_BINS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 1.01)]


def _bin_label(value, bins):
    for lo, hi in bins:
        if lo <= value < hi:
            return f"{lo:.2f}-{hi if hi <= 1.0 else '1.00'}"
    return "不明"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--threshold", type=float, default=0.85,
                     help="検証したい格下げ閾値(既定: confidence.pyの現在値0.85)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        """
        SELECT pl.race_id, pl.race_entropy, pl.top_lane, pl.race_tier
        FROM prediction_log pl
        WHERE pl.race_entropy IS NOT NULL
        """
    ).fetchall()

    print(f"race_entropyが記録されているレース数: {len(rows)}件")
    if not rows:
        print("対象データがありません(race_entropyの記録が始まったばかりの可能性があります)。")
        return

    above_threshold = sum(1 for _, ent, _, _ in rows if ent >= args.threshold)
    print(f"\n閾値{args.threshold}以上(格下げ発動)の件数: {above_threshold}件"
          f"({above_threshold/len(rows)*100:.1f}%)")

    # エントロピー帯ごとの◎的中率を見る(的中判定には結果確定済みのレースだけ使う)
    stats = defaultdict(lambda: {"n": 0, "hits": 0})
    for race_id, entropy, top_lane, tier in rows:
        result_rows = conn.execute(
            """SELECT e.boat_number, r.arrival_order
               FROM entries e JOIN results r ON r.entry_id = e.entry_id
               WHERE e.race_id = ? ORDER BY e.boat_number""",
            (race_id,),
        ).fetchall()
        if len(result_rows) != 6 or any(row[1] is None for row in result_rows):
            continue
        winner = min(result_rows, key=lambda row: row[1])[0]
        label = _bin_label(entropy, ENTROPY_BINS)
        stats[label]["n"] += 1
        if top_lane == winner:
            stats[label]["hits"] += 1

    print(f"\n{'エントロピー帯':<14} {'件数':>6} {'◎的中率':>8}")
    print("-" * 36)
    for lo, hi in ENTROPY_BINS:
        label = f"{lo:.2f}-{hi if hi <= 1.0 else '1.00'}"
        s = stats.get(label)
        if not s or s["n"] == 0:
            print(f"{label:<14} {'0件':>6}")
            continue
        rate = s["hits"] / s["n"] * 100
        print(f"{label:<14} {s['n']:>5}件 {rate:>7.1f}%")

    # 参考: race_tier別にみた格下げ発動率(本当にAを食いつぶしているか)
    tier_counts = defaultdict(lambda: {"n": 0, "above": 0})
    for _, entropy, _, tier in rows:
        if tier is None:
            continue
        tier_counts[tier]["n"] += 1
        if entropy >= args.threshold:
            tier_counts[tier]["above"] += 1

    print(f"\n{'ティア':<6} {'件数':>6} {'うち格下げ閾値以上':>10}")
    print("-" * 36)
    for tier in ["A", "B", "C", "D"]:
        t = tier_counts.get(tier)
        if not t or t["n"] == 0:
            continue
        print(f"{tier:<6} {t['n']:>5}件 {t['above']:>8}件({t['above']/t['n']*100:.1f}%)")

    print("\n※ 閾値以上の発動率が高すぎる(例: 全体の3〜4割を超える)場合、Aが出にくい一因になっている。"
          "エントロピー帯ごとの◎的中率が、エントロピーの上昇とともに明確に下がっていなければ、"
          "そもそも格下げルール自体の前提(混戦=当てにくい)が今のデータでは支持されていないことになる。")


if __name__ == "__main__":
    main()
