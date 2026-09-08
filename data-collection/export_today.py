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

from technique_stats import compute_course_technique_rates, compute_racer_nigashi_rate, course_advantage_score
from build_features import FEATURE_COLS

# 号艇(コース)ごとの平均的な有利さの目安(競艇はイン=1号艇が圧倒的に有利という実際の傾向を反映)
# ※ technique_stats.compute_course_technique_rates() が使えるならそちらを優先し、
#   この定数は最終フォールバックとしてのみ使う。
COURSE_ADVANTAGE = {1: 1.00, 2: 0.62, 3: 0.50, 4: 0.46, 5: 0.34, 6: 0.28}

MARKS = ["◎", "○", "▲", "△", "×", "注"]

# 競艇24場の公式場番号→場名(全国共通)
STADIUM_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}


def score_boat(entry: dict, course_stats: dict) -> float:
    """予想スコアリング関数(暫定・ヒューリスティック)。
    MLモデル(model.txt)が用意されていれば predict_with_model() の方が優先して使われる。
    全国勝率・当地勝率に加え、場ごとのコース別決まり手統計(逃げ率・まくり率等)から
    導いた「コースの勝ちやすさ」を加味する。1号艇については個人の逃し率が分かれば
    さらに調整する。
    """
    nat = entry.get("national_win_rate") or 4.0     # データ欠損時は全国平均程度で補完
    local = entry.get("local_win_rate") or nat
    course = entry.get("boat_number")

    course_adv = course_advantage_score(course_stats, course) if course_stats else COURSE_ADVANTAGE.get(course, 0.3) * 10

    if course == 1 and entry.get("nigashi_rate") is not None:
        # 逃し率(先マイを取っても差される/まくられる率)が高い選手ほど1コース優位を割り引く
        course_adv *= max(0.3, 1 - entry["nigashi_rate"])

    return nat * 0.35 + local * 0.35 + course_adv * 0.30


def predict_with_model(entries: list, course_stats: dict, model) -> list:
    """train_model.py で学習したLightGBMモデルで各艇の勝利確率を予測する。
    build_features.py の FEATURE_COLS と同じ並び・同じ特徴量になるよう揃えている。
    """
    rows = []
    for e in entries:
        course_win_rate = course_stats.get(e["boat_number"], {}).get("win_rate") if course_stats else None
        rows.append({
            "boat_number": e["boat_number"],
            "national_win_rate": e.get("national_win_rate") or 4.0,
            "local_win_rate": e.get("local_win_rate") or 4.0,
            "motor_2連率": e.get("motor_2連率") or 30.0,
            "boat_hull_2連率": e.get("boat_hull_2連率") or 30.0,
            "average_start_timing": e.get("average_start_timing") or 0.17,
            "course_win_rate": course_win_rate if course_win_rate is not None else 0.2,
        })
    import pandas as pd
    X = pd.DataFrame(rows)[FEATURE_COLS]
    return list(model.predict(X))


def normalize_to_pct(values: list, use_softmax: bool) -> list:
    """スコア(またはモデルが出した確率)をパーセント表示用の整数配列(合計100)にする。
    ヒューリスティックスコアは値のスケールがまちまちなのでsoftmaxで正規化し、
    モデルの予測確率(0〜1)はすでに比較可能な値なので単純合計で正規化する。
    """
    import math
    if use_softmax:
        weights = [math.exp(v) for v in values]
    else:
        weights = [max(v, 1e-6) for v in values]
    total = sum(weights)
    raw = [w / total * 100 for w in weights]
    rounded = [round(r) for r in raw]
    diff = 100 - sum(rounded)
    if rounded:
        rounded[0] += diff
    return rounded


def estimate_bets(boats: list, top_n: int = 4) -> list:
    """
    Plackett-Luceモデルで3連単(1着→2着→3着)の確率を推定する。
    P(a→b→c) = P(aが1着) × P(bが2着|aを除いた中で) × P(cが3着|a,bを除いた中で)
    強さには各艇の予想勝率(pct)をそのまま使う近似(完全に厳密ではないが実用上十分)。
    ※ 掛け算の順序を無視していた旧実装のバグを修正(順序を入れ替えても同じ値になり、
      本来上位に来るはずの組み合わせが漏れることがあった)。
    """
    strengths = {b["lane"]: max(b["pct"], 0.1) for b in boats}  # 0%だと計算できないので下駄を履かせる
    lanes = list(strengths.keys())
    total = sum(strengths.values())

    combos = []
    for a in lanes:
        p_a = strengths[a] / total
        remaining_after_a = {l: s for l, s in strengths.items() if l != a}
        total_after_a = sum(remaining_after_a.values())
        for b in remaining_after_a:
            p_b = remaining_after_a[b] / total_after_a
            remaining_after_ab = {l: s for l, s in remaining_after_a.items() if l != b}
            total_after_ab = sum(remaining_after_ab.values())
            for c in remaining_after_ab:
                p_c = remaining_after_ab[c] / total_after_ab
                combos.append({
                    "combo": f"{a}-{b}-{c}",
                    "raw_prob": p_a * p_b * p_c,
                })

    combos.sort(key=lambda x: -x["raw_prob"])
    top = combos[:top_n]
    return [
        {"combo": c["combo"], "prob": f'{c["raw_prob"]*100:.1f}%'}
        for c in top
    ]


def build_today_json(conn: sqlite3.Connection, target_date: date, model=None) -> dict:
    races = conn.execute(
        """SELECT race_id, stadium_number, race_number, race_grade, close_at
           FROM races WHERE race_date = ? ORDER BY stadium_number, race_number""",
        (target_date.isoformat(),),
    ).fetchall()

    stadium_map = {}
    course_stats_cache = {}  # stadium_number -> compute_course_technique_rates() の結果(場ごとに1回だけ計算)

    for race_id, stadium_number, race_number, race_grade, close_at in races:
        entries = conn.execute(
            """SELECT boat_number, racer_name, racer_registration_number,
                      national_win_rate, local_win_rate, motor_2連率, boat_hull_2連率,
                      average_start_timing
               FROM entries WHERE race_id = ? ORDER BY boat_number""",
            (race_id,),
        ).fetchall()
        if not entries:
            continue  # 出走表未取得のレースはスキップ

        if stadium_number not in course_stats_cache:
            course_stats_cache[stadium_number] = compute_course_technique_rates(conn, stadium_number)
        course_stats = course_stats_cache[stadium_number]

        boat_dicts = []
        for bn, name, reg_no, nat, local, motor_2r, hull_2r, avg_st in entries:
            d = {
                "boat_number": bn,
                "racer_name": name,
                "national_win_rate": nat,
                "local_win_rate": local,
                "motor_2連率": motor_2r,
                "boat_hull_2連率": hull_2r,
                "average_start_timing": avg_st,
            }
            if bn == 1 and reg_no is not None:
                nigashi = compute_racer_nigashi_rate(conn, reg_no)
                if nigashi:
                    d["nigashi_rate"] = nigashi["nigashi_rate"]
            boat_dicts.append(d)

        if model is not None:
            scores = predict_with_model(boat_dicts, course_stats, model)
            pcts = normalize_to_pct(scores, use_softmax=False)
        else:
            scores = [score_boat(b, course_stats) for b in boat_dicts]
            pcts = normalize_to_pct(scores, use_softmax=True)

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
        "using_ml_model": model is not None,
        "stadiums": list(stadium_map.values()),
        "course_stats_by_stadium": {
            str(sn): {
                str(c): {"win_rate": round(s[c]["win_rate"], 3), "is_fallback": s[c]["is_fallback"], "sample_size": s[c]["sample_size"]}
                for c in range(1, 7)
            }
            for sn, s in course_stats_cache.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (省略時は本日)")
    ap.add_argument("--out", default="data/today.json")
    ap.add_argument("--model", default=None, help="train_model.py で作ったmodel.txtのパス(省略時はヒューリスティックで予想)")
    args = ap.parse_args()

    model = None
    if args.model and Path(args.model).exists():
        import lightgbm as lgb
        model = lgb.Booster(model_file=args.model)
        print(f"MLモデルを読み込みました: {args.model}")
    elif args.model:
        print(f"⚠ 指定されたモデルファイルが見つかりません: {args.model}(ヒューリスティックで続行します)")

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    conn = sqlite3.connect(args.db)
    result = build_today_json(conn, target_date, model=model)
    conn.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"出力しました: {out_path} ({len(result['stadiums'])}場)")


if __name__ == "__main__":
    main()
