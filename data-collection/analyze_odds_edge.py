"""
turnmark/api(非公式、前日までの確定オッズをJSON形式で提供。
https://github.com/turnmark/api)から過去のオッズを取得し、
自分たちのモデルが推奨した3連単の組み合わせ(prediction_log.top_bets_json)が、
実際の市場ではどのくらいのオッズ(=市場が織り込んでいた確率)だったかを比較する。

目的:
    「モデルが高確率と予測した組み合わせが、市場のオッズでも同じように評価されて
    いるのか、それとも市場が気づいていない(過小評価されている=割安な)組み合わせ
    なのか」を検証する。的中率だけで選別する今のA〜D判定は、控除率(約25%)の壁を
    超えられない可能性が高いことが分かっている。もし「モデルの予測確率 > 市場の
    暗黙の確率」となる組み合わせに絞って買った場合の回収率が高ければ、的中率ではなく
    期待値で選別する新しい軸になり得る。

データソースの制約(必ず踏まえること):
    - turnmark/api は前日までのデータしか提供しない(当日のリアルタイムオッズは
      含まれない)。このスクリプトは研究・検証専用であり、当日の本番予想・
      export_today.py の計算には一切使わない。
    - 1日1ファイル(https://turnmark.github.io/api/v1/YYYY/YYYYMMDD.json)で
      全24場・その日の全レース分をまとめて提供している。1日1回更新。
    - 非公式のAPIであり、正確性・完全性は保証されていない(ドキュメントにも明記)。
    - 以前boatrace.dbが100MBを超えてpushできなくなった教訓を踏まえ、取得した
      オッズはDBに保存しない(1レース120通りの3連単オッズを全件保存すると、
      すぐにまた同じ問題が起きる)。自分たちが推奨した組み合わせの分だけ、
      その場で突き合わせて集計する。

出力する集計:
    1. 全候補・期待値>1(モデルが割安と判断)・期待値<=1 の3グループ
       (期待値 = モデルの予測確率 × オッズ。1を超えれば、モデルの見立てでは買う価値がある)
    2. 期待値の区分別(0.5未満〜1.5以上)。期待値が高い区分ほど実際の回収率も高ければ、
       モデルの確率は市場以上の情報を持っている。上がらなければ持っていない。
    3. ティア別(A〜D)と、ティアAだけに絞った場合の集計
    回収率には、ブートストラップ(同じ候補を重複を許して引き直す)による95%の範囲を付ける。
    3連単は少数の高配当で回収率が大きく動くため、点の数字だけで判断しないようにするため。
    ※ 区分や条件を増やして探すほど、偶然良く見える組み合わせは見つかりやすくなる。
      良さそうな区分が見つかっても、別の期間で再現するかを必ず確認すること。

使い方:
    python analyze_odds_edge.py --db boatrace.db --start 2026-10-01 --end 2026-10-14
"""
import argparse
import random
import sqlite3
import time
from datetime import date, timedelta

import requests

TURNMARK_BASE_URL = "https://turnmark.github.io/api/v1"
REQUEST_INTERVAL_SEC = 1.0  # 公開GitHub Pagesとはいえ、短時間に連続アクセスしすぎないよう間隔を空ける
STAKE = 100  # 1候補あたりの購入額(円)
BOOTSTRAP_ROUNDS = 1000
# 期待値(予測確率×オッズ)の区分の下限値。全体用は細かく、件数が少ないティアA用は粗く分ける。
EV_EDGES_FINE = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
EV_EDGES_COARSE = [0.0, 0.75, 1.0]


def fetch_turnmark_day(d: date) -> dict:
    """指定日のturnmarkデータを取得する。対応期間外・未取得の日付は404になるのでNoneを返す。"""
    url = f"{TURNMARK_BASE_URL}/{d.year:04d}/{d.strftime('%Y%m%d')}.json"
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        print(f"  取得失敗 {d}: {e}")
        return None
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def get_trifecta_odds(day_data: dict, stadium_number: int, race_number: int, combo: str) -> float:
    """day_dataから、指定レース・指定組み合わせ(例 "4-3-1")の3連単オッズを取り出す。
    該当が無ければ(欠場艇絡み・データ欠損等)Noneを返す。"""
    try:
        parts = combo.split("-")
        race = day_data["programs"]["stadiums"][str(stadium_number)]["races"][str(race_number)]
        odds = race["odds"]["trifecta"]
        return odds[parts[0]][parts[1]][parts[2]]
    except (KeyError, TypeError, IndexError):
        return None


def _fetch_our_recommendations(conn: sqlite3.Connection, target_date: str) -> list:
    """その日の、prediction_logの最終スナップショット(=締切前の最後の予想)を
    レースごとに1件ずつ取得する。(top_bet_combo・top_bets_jsonの候補・結果も含む)"""
    import json

    rows = conn.execute(
        """
        SELECT pl.race_id, r.stadium_number, r.race_number, pl.top_bets_json, pl.race_tier, pl.computed_at
        FROM prediction_log pl
        JOIN races r ON r.race_id = pl.race_id
        WHERE r.race_date = ?
        ORDER BY pl.race_id, pl.computed_at
        """,
        (target_date,),
    ).fetchall()

    # 万一同じレースに複数のスナップショットが残っていても二重に数えないよう、
    # レースごとに最後の1件だけを使う。
    latest = {}
    for row in rows:
        latest[row[0]] = row

    results = []
    for race_id, stadium_number, race_number, top_bets_json, race_tier, _computed_at in latest.values():
        if not top_bets_json:
            continue
        try:
            candidates = json.loads(top_bets_json)
        except (TypeError, ValueError):
            continue

        result_rows = conn.execute(
            """SELECT e.boat_number, r.arrival_order
               FROM entries e JOIN results r ON r.entry_id = e.entry_id
               WHERE e.race_id = ? ORDER BY e.boat_number""",
            (race_id,),
        ).fetchall()
        if len(result_rows) != 6 or any(row[1] is None for row in result_rows):
            continue  # 結果未確定のレースは検証対象外
        ordered = sorted(result_rows, key=lambda row: row[1])
        actual_combo = "-".join(str(row[0]) for row in ordered[:3])

        results.append({
            "race_id": race_id, "stadium_number": stadium_number, "race_number": race_number,
            "candidates": candidates, "actual_combo": actual_combo, "tier": race_tier,
        })
    return results


def _roi_ci(payouts: list) -> tuple:
    """回収率(%)の95%の範囲を、ブートストラップ(候補を重複を許して引き直す)で求める。
    3連単は少数の高配当で回収率が大きく動くので、点の数字だけでなく範囲も併せて見る。"""
    n = len(payouts)
    if n == 0:
        return (None, None)
    rng = random.Random(0)  # 毎回同じ結果になるよう固定
    means = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = rng.choices(payouts, k=n)
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * BOOTSTRAP_ROUNDS)]
    hi = means[int(0.975 * BOOTSTRAP_ROUNDS) - 1]
    return (lo / STAKE * 100, hi / STAKE * 100)


def _print_row(label: str, group: list):
    """1グループ分の件数・的中率・回収率(95%の範囲つき)を表示し、(件数, 回収率, 下限, 上限)を返す。"""
    n = len(group)
    if n == 0:
        print(f"  {label}: 対象0件")
        return None
    hits = sum(1 for b in group if b["hit"])
    payouts = [b["payout"] for b in group]
    roi = sum(payouts) / (n * STAKE) * 100
    lo, hi = _roi_ci(payouts)
    print(f"  {label}: {n}件 / 的中{hits}件({hits / n * 100:.1f}%) / 回収率{roi:.1f}%(95%範囲 {lo:.1f}〜{hi:.1f}%)")
    return (n, roi, lo, hi)


def _ev_labels(edges: list) -> list:
    labels = []
    for i, e in enumerate(edges):
        if i + 1 < len(edges):
            labels.append(f"期待値{e:.2f}〜{edges[i + 1]:.2f}")
        else:
            labels.append(f"期待値{e:.2f}以上")
    return labels


def _print_ev_table(group: list, edges: list):
    buckets = [[] for _ in edges]
    for b in group:
        idx = 0
        for i, e in enumerate(edges):
            if b["ev"] >= e:
                idx = i
        buckets[idx].append(b)
    for label, bucket in zip(_ev_labels(edges), buckets):
        _print_row(label, bucket)


def _print_verdict(title: str, summary: tuple):
    """回収率の95%の範囲が100%をまたぐかどうかを、事実として一言で示す(自動で採用判断はしない)。"""
    if summary is None:
        return
    _, roi, lo, hi = summary
    if hi < 100:
        print(f"  → {title}: 範囲の上限{hi:.1f}%が100%を下回っています。この期間のデータでは、"
              f"儲かっていた証拠はありません。")
    elif lo > 100:
        print(f"  → {title}: 範囲の下限{lo:.1f}%が100%を上回っています。有利な可能性はありますが、"
              f"別の期間でも再現するかの確認が必要です。")
    else:
        print(f"  → {title}: 範囲が100%をまたいでいます({lo:.1f}〜{hi:.1f}%)。"
              f"有利とも不利とも判断できません。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    # startがendより後だと、ループが1回も回らず「0件」のまま黙って終わってしまう
    # (実際にそうなって原因が分かりにくかった)ため、最初に明示的に止める。
    if start > end:
        raise SystemExit(f"startがendより後になっています(start={start}, end={end})。"
                          f"日付を入れ替えて実行してください。")
    print(f"検証期間: {start} 〜 {end}({(end - start).days + 1}日間)")

    bets = []  # 1候補=1件。ev=モデルの予測確率×オッズ、hit=的中したか、payout=払戻(円)、tier=そのレースのA〜D
    no_odds_count = 0
    no_data_days = 0
    races_found_total = 0   # 自分たちの推奨データ(結果確定済み)が見つかったレース数
    candidates_seen = 0     # そのうち、推奨候補として調べた組み合わせの数
    skipped_bad_candidate = 0  # 確率・組み合わせが読み取れず、オッズ照合まで進めなかった候補数

    d = start
    while d <= end:
        print(f"取得中: {d}")
        day_data = fetch_turnmark_day(d)
        time.sleep(REQUEST_INTERVAL_SEC)
        if day_data is None:
            print(f"  データなし(対応期間外 or 未取得)")
            no_data_days += 1
            d += timedelta(days=1)
            continue

        races = _fetch_our_recommendations(conn, d.isoformat())
        races_found_total += len(races)
        print(f"  自分たちの推奨データ(結果確定済み): {len(races)}レース")
        for race in races:
            for cand in race["candidates"]:
                candidates_seen += 1
                combo = cand.get("combo")
                prob_str = cand.get("prob", "")
                try:
                    model_prob = float(prob_str.rstrip("%")) / 100
                except (ValueError, AttributeError):
                    skipped_bad_candidate += 1
                    continue
                if not combo:
                    skipped_bad_candidate += 1
                    continue

                market_odds = get_trifecta_odds(day_data, race["stadium_number"], race["race_number"], combo)
                if market_odds is None or market_odds <= 0:
                    no_odds_count += 1
                    continue

                hit = combo == race["actual_combo"]
                bets.append({
                    "ev": model_prob * market_odds,
                    "hit": hit,
                    "payout": round(market_odds * STAKE) if hit else 0,
                    "tier": race.get("tier") or "?",
                })

        d += timedelta(days=1)

    conn.close()

    print(f"\n自分たちの推奨データが見つかったレース数: {races_found_total}レース"
          f"(候補{candidates_seen}件、うち確率・組み合わせが読み取れなかったもの{skipped_bad_candidate}件)")
    print(f"対応期間外・データなしの日数: {no_data_days}日")
    print(f"オッズが見つからなかった候補数(欠場艇絡み等): {no_odds_count}件")

    if not bets:
        print("\n集計対象の候補が0件でした。")
        return

    print(f"\n■ 全体(推奨候補を1点{STAKE}円ずつ買った場合)")
    _print_row("全候補", bets)
    ev_pos = [b for b in bets if b["ev"] > 1.0]
    ev_neg = [b for b in bets if b["ev"] <= 1.0]
    pos_summary = _print_row("期待値>1(モデルが割安と判断)", ev_pos)
    _print_row("期待値<=1(市場並み〜割高)", ev_neg)
    _print_verdict("期待値>1の候補", pos_summary)

    print("\n■ 期待値(予測確率×オッズ)の区分別")
    _print_ev_table(bets, EV_EDGES_FINE)

    print("\n■ ティア別(そのティアのレースの推奨候補を買った場合)")
    for tier in ["A", "B", "C", "D"]:
        _print_row(f"ティア{tier}", [b for b in bets if b["tier"] == tier])

    tier_a = [b for b in bets if b["tier"] == "A"]
    print(f"\n■ ティアAのみ(件数が少ないので区分を粗くしています)")
    _print_row("ティアA全候補", tier_a)
    _print_ev_table(tier_a, EV_EDGES_COARSE)
    a_pos = [b for b in tier_a if b["ev"] > 1.0]
    a_summary = _print_row("ティアAかつ期待値>1", a_pos)
    _print_verdict("ティアAかつ期待値>1", a_summary)

    print("\n※ 期待値が高い区分ほど回収率も高くなっていれば、モデルの確率は市場以上の情報を持っている。"
          "上がっていなければ持っていない。")
    print("※ 回収率の95%の範囲が広いほど、その数字は偶然で大きく変わり得る。範囲が100%をまたぐ間は、"
          "有利とも不利とも言えない。")
    print("※ 払戻はturnmarkのオッズ×100円で計算した値で、実際の払戻金とは端数が異なることがある。")


if __name__ == "__main__":
    main()
