"""
DBから本日(または指定日)のレースを取り出し、フロントエンド(HTML/将来のFlutter)が
そのまま読み込めるJSON(today.json)を出力するスクリプト。

暫定スコアリング:
    まだMLモデルがないため、全国勝率・当地勝率・コース(号艇)有利さの加重平均で
    仮のスコアを出している。MLモデルができたら score_boat() だけ差し替えればよい。

使い方:
    python export_today.py --db boatrace.db --date 2026-09-07 --out data/today.json
    (--date省略時は本日日付)
"""
import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

# 号艇(コース)ごとの平均的な有利さの目安(競艇はイン=1号艇が圧倒的に有利という実際の傾向を反映)
COURSE_ADVANTAGE = {1: 1.00, 2: 0.62, 3: 0.50, 4: 0.46, 5: 0.34, 6: 0.28}

MARKS = ["◎", "○", "▲", "△", "×", "注"]

# 競艇24場の公式場番号→場名(全国共通)
STADIUM_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}


def score_boat(entry: dict) -> float:
    """暫定スコアリング関数。MLモデル完成後はここをモデル推論に置き換える。"""
    nat = entry.get("national_win_rate") or 4.0     # データ欠損時は全国平均程度で補完
    local = entry.get("local_win_rate") or nat
    course_adv = COURSE_ADVANTAGE.get(entry.get("boat_number"), 0.3)
    # 全国勝率・当地勝率・コース有利さを重み付けして合成(重みは仮置き)
    return nat * 0.35 + local * 0.35 + course_adv * 10 * 0.30


def scores_to_pct(scores: list) -> list:
    """スコアをsoftmax風に正規化してパーセント表示用の整数配列にする"""
    import math
    exps = [math.exp(s) for s in scores]
    total = sum(exps)
    raw = [e / total * 100 for e in exps]
    # 四捨五入しつつ合計100に丸め込む
    rounded = [round(r) for r in raw]
    diff = 100 - sum(rounded)
    if rounded:
        rounded[0] += diff
    return rounded


def estimate_bets(boats: list, top_n: int = 4) -> list:
    """boats(pct付き)から3連単の期待値っぽい候補を確率の積で簡易生成"""
    ranked = sorted(boats, key=lambda b: -b["pct"])
    combos = []
    top5 = ranked[:5] if len(ranked) >= 5 else ranked
    for i, a in enumerate(top5):
        for b in top5:
            if b is a:
                continue
            for c in top5:
                if c is a or c is b:
                    continue
                p = (a["pct"] / 100) * (b["pct"] / 100) * (c["pct"] / 100)
                combos.append({
                    "combo": f'{a["lane"]}-{b["lane"]}-{c["lane"]}',
                    "raw_prob": p,
                })
    combos.sort(key=lambda x: -x["raw_prob"])
    top = combos[:top_n]
    return [
        {"combo": c["combo"], "prob": f'{c["raw_prob"]*100:.1f}%', "odds": "予想オッズ 未算出"}
        for c in top
    ]


def build_today_json(conn: sqlite3.Connection, target_date: date) -> dict:
    races = conn.execute(
        """SELECT race_id, stadium_number, race_number, race_grade, close_at
           FROM races WHERE race_date = ? ORDER BY stadium_number, race_number""",
        (target_date.isoformat(),),
    ).fetchall()

    stadium_map = {}
    for race_id, stadium_number, race_number, race_grade, close_at in races:
        entries = conn.execute(
            """SELECT boat_number, racer_name, national_win_rate, local_win_rate
               FROM entries WHERE race_id = ? ORDER BY boat_number""",
            (race_id,),
        ).fetchall()
        if not entries:
            continue  # 出走表未取得のレースはスキップ

        boat_dicts = [
            {
                "boat_number": bn,
                "racer_name": name,
                "national_win_rate": nat,
                "local_win_rate": local,
            }
            for bn, name, nat, local in entries
        ]
        scores = [score_boat(b) for b in boat_dicts]
        pcts = scores_to_pct(scores)

        boats_out = []
        for b, pct in zip(boat_dicts, pcts):
            boats_out.append({
                "lane": b["boat_number"],
                "name": b["racer_name"] or f'{b["boat_number"]}号艇選手',
                "pct": pct,
                "natWin": b["national_win_rate"],
                "localWin": b["local_win_rate"],
            })
        # 予想順にソートして印を付与
        boats_ranked = sorted(boats_out, key=lambda x: -x["pct"])
        for i, b in enumerate(boats_ranked):
            b["mark"] = MARKS[i] if i < len(MARKS) else ""

        race_out = {
            "number": race_number,
            "close": close_at,
            "grade": race_grade,
            "boats": sorted(boats_ranked, key=lambda x: x["lane"]),  # 表示は号艇順
            "boats_ranked": boats_ranked,                            # 予想順(印付き)
            "bets": estimate_bets(boats_out),
        }

        stadium_name = STADIUM_NAMES.get(stadium_number, f"第{stadium_number}場")
        stadium_map.setdefault(stadium_name, {"name": stadium_name, "stadium_number": stadium_number, "grade": None, "races": []})
        stadium_map[stadium_name]["races"].append(race_out)

    return {
        "date": target_date.isoformat(),
        "generated_at": date.today().isoformat(),
        "stadiums": list(stadium_map.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (省略時は本日)")
    ap.add_argument("--out", default="data/today.json")
    args = ap.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    conn = sqlite3.connect(args.db)
    result = build_today_json(conn, target_date)
    conn.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"出力しました: {out_path} ({len(result['stadiums'])}場)")


if __name__ == "__main__":
    main()
