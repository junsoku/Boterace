"""
過去の結果が確定済みのレースについて、「◎(予想1位)の確率が何%だった時、
実際にどのくらいの割合で当たっていたか」を集計するモジュール。

例えば「◎の予想確率が40%台だったレースは、過去に実際65%当たっている」といった
実績が分かれば、今日のレースでも同じ確率帯なら信頼して良い、という判断ができる。
これはオッズが無くても作れる「当たりやすさの実績値」。

単勝(◎)用に加えて、推奨3連単(本命1点)用の確信度も同じ考え方で計算する
(compute_bet_confidence_calibration / lookup_bet_confidence)。

※ 注意: この過去再計算は score_boat()(ヒューリスティック)を使っており、
  本番で実際に使っているMLモデル(train_model.py / predict_with_model)とは
  厳密には別のロジック。確率の値がわずかにズレる可能性がある。

使い方(export_today.py内から呼び出す想定):
    from confidence import compute_confidence_calibration, lookup_confidence
    from confidence import compute_bet_confidence_calibration, lookup_bet_confidence
    calibration = compute_confidence_calibration(conn)
    conf = lookup_confidence(calibration, top_pick_pct)
    # conf = {"bucket": "40-49%", "hit_rate": 0.65, "sample_size": 42, "is_confident": True}

    bet_calibration = compute_bet_confidence_calibration(conn)
    bet_conf = lookup_bet_confidence(bet_calibration, top_bet_prob_pct)
    # bet_conf = {"bucket": "10-14%", "hit_rate": 0.22, "sample_size": 35, "is_confident": True}
"""
import sqlite3
from datetime import date, timedelta
from typing import Optional

from export_today import (
    score_boat, normalize_to_pct, estimate_bets, estimate_bets_ml,
    estimate_exacta_bets, estimate_exacta_bets_ml, predict_with_model,
)
from technique_stats import (
    compute_course_technique_rates, compute_racer_nigashi_rate, compute_racer_recent_form,
)

# ---- ◎(単勝)用の確信度設定 ----
# 予想確率を10%刻みでグループ化する(サンプルが集まりやすいよう粗めの区切り)
BUCKET_SIZE = 10
MIN_SAMPLE_FOR_STAR = 20   # このサンプル数未満のバケットは信頼度を判定しない
CONFIDENCE_THRESHOLD = 0.60  # このバケットの過去的中率がこれ以上なら「星」を付ける

# ---- 推奨3連単(本命1点)用の確信度設定 ----
# 3連単の推定確率は◎の単勝確率よりずっと低いレンジ(数%〜20%程度)に収まりやすいので、
# バケットを細かく(5%刻み)取る。
BET_BUCKET_SIZE = 5
MIN_SAMPLE_FOR_BET_STAR = 20
# 3連単(6艇中120通り)のランダムな的中率は1/120≈0.83%。そこから大きく上振れしていることを
# 「自信あり」の目安にする。
# ※ 実データで診断した結果(diagnose_confidence.py)、Plackett-Luce計算の性質上
#   1つの組み合わせの予測確率が10%を超えること自体がほぼ無く、0.15だと理論上到達困難だった。
#   実測では「予測確率5-9%」帯の実際の的中率が12.0%(2026-09時点、n=576)だったため、
#   そこを拾えるよう0.10に調整。データが増えたら再度diagnose_confidence.pyで見直すこと。
BET_CONFIDENCE_THRESHOLD = 0.10

# ---- 推奨2連単(本命1点)用の確信度設定 ----
# 2連単(6艇中30通り)は3連単より的中しやすい(ランダムなら1/30≈3.3%)ので、
# バケットは3連単よりやや粗めに、しきい値も高めに取る。
EXACTA_BUCKET_SIZE = 5
MIN_SAMPLE_FOR_EXACTA_STAR = 20
EXACTA_CONFIDENCE_THRESHOLD = 0.30


def _bucket_label(pct: int) -> str:
    lower = (pct // BUCKET_SIZE) * BUCKET_SIZE
    upper = lower + BUCKET_SIZE - 1
    return f"{lower}-{upper}%"


def _bet_bucket_label(pct: float) -> str:
    lower = (int(pct) // BET_BUCKET_SIZE) * BET_BUCKET_SIZE
    upper = lower + BET_BUCKET_SIZE - 1
    return f"{lower}-{upper}%"


def _exacta_bucket_label(pct: float) -> str:
    lower = (int(pct) // EXACTA_BUCKET_SIZE) * EXACTA_BUCKET_SIZE
    upper = lower + EXACTA_BUCKET_SIZE - 1
    return f"{lower}-{upper}%"


def _iter_calibration_races(conn: sqlite3.Connection, lookback_days: int, model=None):
    """過去lookback_days日分の、結果が確定しているレースを1件ずつ再予想しながら返す共通処理。
    compute_confidence_calibration / compute_bet_confidence_calibration の両方から使う。

    model が渡されていれば、本番(export_today.py)と同じ predict_with_model()(ランク学習MLモデル)
    で再予想する。model が None の場合のみ、旧来のヒューリスティック(score_boat)にフォールバックする。
    確信度の実績値は「本番で実際に使っている予想ロジック」と揃っていて初めて意味を持つため、
    model があるなら必ずそちらを使うべき。

    yield: (boats[{"lane":.., "pct":..}], actual_order[boat_numberを着順順に並べたリスト],
            boat_dicts, scores_by_lane または None)
    """
    end = date.today()
    start = end - timedelta(days=lookback_days)

    races = conn.execute(
        """SELECT race_id, race_date, stadium_number, race_grade, wave_height_cm, wind_speed_m,
                  temperature_c, water_temperature_c
           FROM races WHERE race_date BETWEEN ? AND ?""",
        (start.isoformat(), end.isoformat()),
    ).fetchall()

    course_stats_cache = {}

    for race_id, race_date, stadium_number, race_grade, wave, wind, temperature_c, water_temperature_c in races:
        entries = conn.execute(
            """SELECT e.boat_number, e.racer_registration_number, e.racer_class,
                      e.national_win_rate, e.national_2連率, e.local_win_rate, e.local_2連率,
                      e.motor_2連率, e.boat_hull_2連率, e.average_start_timing,
                      e.flying_count, e.late_count,
                      p.exhibition_time, p.tilt_angle, p.weight_adjustment_kg,
                      p.start_timing_preview, p.start_course, r.arrival_order
               FROM entries e
               JOIN results r ON r.entry_id = e.entry_id
               LEFT JOIN previews p ON p.entry_id = e.entry_id
               WHERE e.race_id = ? ORDER BY e.boat_number""",
            (race_id,),
        ).fetchall()
        if len(entries) != 6 or any(row[-1] is None for row in entries):
            continue

        if stadium_number not in course_stats_cache:
            course_stats_cache[stadium_number] = compute_course_technique_rates(conn, stadium_number)
        course_stats = course_stats_cache[stadium_number]

        exh_values = [row[12] for row in entries if row[12]]
        avg_exh = sum(exh_values) / len(exh_values) if exh_values else None
        race_context = {"wave_height_cm": wave, "wind_speed_m": wind, "race_grade": race_grade,
                         "avg_exhibition_time": avg_exh,
                         "temperature_c": temperature_c, "water_temperature_c": water_temperature_c}

        boat_dicts = []
        for (bn, reg_no, racer_class, nat, nat_2r, local, local_2r, motor_2r, hull_2r, avg_st,
             flying, late, exh, tilt_angle, weight_adj, start_timing_prev, start_course,
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
                "start_course": start_course,
            }
            if reg_no is not None:
                # そのレースの日付より前の結果だけを使う(build_features.pyと同じ、リーク防止)
                form = compute_racer_recent_form(conn, reg_no, before_date=race_date)
                if form:
                    d["racer_recent_avg_order"] = form["avg_arrival_order"]
                    d["racer_recent_win_rate"] = form["recent_win_rate"]
            if bn == 1 and reg_no is not None:
                nigashi = compute_racer_nigashi_rate(conn, reg_no)
                if nigashi:
                    d["nigashi_rate"] = nigashi["nigashi_rate"]
            boat_dicts.append((d, arrival_order))

        scores_by_lane = None
        if model is not None:
            scores = predict_with_model([d for d, _ in boat_dicts], course_stats, model, race_context)
            pcts = normalize_to_pct(scores, use_softmax=True)
            scores_by_lane = {d["boat_number"]: s for (d, _), s in zip(boat_dicts, scores)}
        else:
            scores = [score_boat(d, course_stats, race_context) for d, _ in boat_dicts]
            pcts = normalize_to_pct(scores, use_softmax=True)

        boats = [{"lane": d["boat_number"], "pct": pct} for (d, _), pct in zip(boat_dicts, pcts)]
        actual_order = [d["boat_number"] for (d, arrival) in
                         sorted(boat_dicts, key=lambda x: x[1])]

        yield boats, actual_order, [d for d, _ in boat_dicts], scores_by_lane


def compute_confidence_calibration(conn: sqlite3.Connection, lookback_days: int = 90, model=None) -> dict:
    """
    過去lookback_days日分の結果確定レースを、現在のロジック(modelがあればMLモデル)で再予想し、
    ◎の予想確率帯ごとの実際の的中率を集計する。
    戻り値: {"30-39%": {"hit_rate":0.55,"sample_size":18}, ...}
    """
    buckets = {}  # label -> [hits, total]

    for boats, actual_order, _boat_dicts, _scores_by_lane in _iter_calibration_races(conn, lookback_days, model):
        ranked = sorted(boats, key=lambda b: -b["pct"])
        top_boat = ranked[0]
        top_pct = top_boat["pct"]

        label = _bucket_label(top_pct)
        if label not in buckets:
            buckets[label] = [0, 0]
        buckets[label][1] += 1
        if actual_order and actual_order[0] == top_boat["lane"]:
            buckets[label][0] += 1

    return {
        label: {"hit_rate": hits / total if total else None, "sample_size": total}
        for label, (hits, total) in buckets.items()
    }


def compute_bet_confidence_calibration(conn: sqlite3.Connection, lookback_days: int = 90, model=None) -> dict:
    """
    過去lookback_days日分の結果確定レースを、現在のロジック(modelがあればMLモデル)で再予想し、
    推奨3連単(本命1点)の推定確率帯ごとに、実際にその組み合わせが的中していた割合を集計する。
    戻り値: {"10-14%": {"hit_rate":0.22,"sample_size":35}, ...}

    model があれば estimate_bets_ml()(ランク学習モデル1つのスコアによるPlackett-Luce計算)
    で3連単を推定する。無ければ旧来の estimate_bets()(1着確率の按分)にフォールバックする。
    """
    buckets = {}  # label -> [hits, total]

    for boats, actual_order, boat_dicts, scores_by_lane in _iter_calibration_races(conn, lookback_days, model):
        if scores_by_lane is not None:
            bets = estimate_bets_ml(boat_dicts, scores_by_lane, top_n=1)
        else:
            bets = estimate_bets(boats, top_n=1)
        if not bets:
            continue
        top_bet = bets[0]
        try:
            prob_val = float(top_bet["prob"].rstrip("%"))
        except (ValueError, AttributeError):
            continue

        label = _bet_bucket_label(prob_val)
        if label not in buckets:
            buckets[label] = [0, 0]
        buckets[label][1] += 1

        actual_combo = "-".join(str(n) for n in actual_order[:3])
        if top_bet["combo"] == actual_combo:
            buckets[label][0] += 1

    return {
        label: {"hit_rate": hits / total if total else None, "sample_size": total}
        for label, (hits, total) in buckets.items()
    }


def compute_exacta_confidence_calibration(conn: sqlite3.Connection, lookback_days: int = 90, model=None) -> dict:
    """
    過去lookback_days日分の結果確定レースを、現在のロジック(modelがあればMLモデル)で再予想し、
    推奨2連単(本命1点)の推定確率帯ごとに、実際にその組み合わせが的中していた割合を集計する。
    戻り値: {"15-19%": {"hit_rate":0.35,"sample_size":40}, ...}

    model があれば estimate_exacta_bets_ml()(ランク学習モデルのスコアによるPlackett-Luce計算)
    で推定する。無ければ旧来の estimate_exacta_bets()(1着確率の按分)にフォールバックする。
    """
    buckets = {}  # label -> [hits, total]

    for boats, actual_order, boat_dicts, scores_by_lane in _iter_calibration_races(conn, lookback_days, model):
        if scores_by_lane is not None:
            bets = estimate_exacta_bets_ml(boat_dicts, scores_by_lane, top_n=1)
        else:
            bets = estimate_exacta_bets(boats, top_n=1)
        if not bets:
            continue
        top_bet = bets[0]
        try:
            prob_val = float(top_bet["prob"].rstrip("%"))
        except (ValueError, AttributeError):
            continue

        label = _exacta_bucket_label(prob_val)
        if label not in buckets:
            buckets[label] = [0, 0]
        buckets[label][1] += 1

        actual_combo = "-".join(str(n) for n in actual_order[:2])
        if top_bet["combo"] == actual_combo:
            buckets[label][0] += 1

    return {
        label: {"hit_rate": hits / total if total else None, "sample_size": total}
        for label, (hits, total) in buckets.items()
    }


def lookup_confidence(calibration: dict, top_pick_pct: int) -> dict:
    """今日の予想の◎確率から、対応するバケットの実績を引いて信頼度を判定する"""
    label = _bucket_label(top_pick_pct)
    stat = calibration.get(label)
    if not stat or stat["sample_size"] < MIN_SAMPLE_FOR_STAR:
        return {"bucket": label, "hit_rate": stat["hit_rate"] if stat else None,
                "sample_size": stat["sample_size"] if stat else 0, "is_confident": False}
    return {
        "bucket": label,
        "hit_rate": stat["hit_rate"],
        "sample_size": stat["sample_size"],
        "is_confident": stat["hit_rate"] >= CONFIDENCE_THRESHOLD,
    }


def lookup_bet_confidence(calibration: dict, top_bet_prob_pct: float) -> dict:
    """今日の推奨3連単(本命)の推定確率から、対応するバケットの実績を引いて信頼度を判定する"""
    label = _bet_bucket_label(top_bet_prob_pct)
    stat = calibration.get(label)
    if not stat or stat["sample_size"] < MIN_SAMPLE_FOR_BET_STAR:
        return {"bucket": label, "hit_rate": stat["hit_rate"] if stat else None,
                "sample_size": stat["sample_size"] if stat else 0, "is_confident": False}
    return {
        "bucket": label,
        "hit_rate": stat["hit_rate"],
        "sample_size": stat["sample_size"],
        "is_confident": stat["hit_rate"] >= BET_CONFIDENCE_THRESHOLD,
    }


def lookup_exacta_confidence(calibration: dict, top_exacta_prob_pct: float) -> dict:
    """今日の推奨2連単(本命)の推定確率から、対応するバケットの実績を引いて信頼度を判定する"""
    label = _exacta_bucket_label(top_exacta_prob_pct)
    stat = calibration.get(label)
    if not stat or stat["sample_size"] < MIN_SAMPLE_FOR_EXACTA_STAR:
        return {"bucket": label, "hit_rate": stat["hit_rate"] if stat else None,
                "sample_size": stat["sample_size"] if stat else 0, "is_confident": False}
    return {
        "bucket": label,
        "hit_rate": stat["hit_rate"],
        "sample_size": stat["sample_size"],
        "is_confident": stat["hit_rate"] >= EXACTA_CONFIDENCE_THRESHOLD,
    }


# ---- レース単位の信頼度ティア(A/B/C/D) ----
# 「このレース自体、予想が当てやすいか」を事前に判定し、Dは見送りの目安にする。
# 単勝・2連単・3連単、それぞれの実績データ(予測確率帯ごとの過去の的中率)で個別に判定し、
# 一番厳しい(悪い)ものを採用する。「単勝は堅いが2着・3着が大混戦」のようなレースを
# 単勝だけで見た判定(=★自信ありバッジと同じ情報)にせず、正しくC/D寄りに倒すため。
# 「勘」や手作りの閾値ではなく、実際に的中率を最優先する方針に沿って、
# 「過去、この確率帯だった時に本当にどれくらい当たっていたか」を軸に判定する。

# 単勝は◎の予測確率がそのまま50〜60%台まで乗りやすいので、実績的中率の基準もそれに合わせて高め。
# 各グレード間の間隔を広めに取り、A〜Dの実際の的中率差がはっきり出るようにしている
# (間隔が狭いと、複数軸の加重平均を通した後で差がぼやけてしまうため)。
RACE_TIER_WIN_THRESHOLDS = [
    ("A", 0.65),
    ("B", 0.50),
    ("C", 0.35),
]
# 2連単(30通り)は単勝より当てにくいが3連単よりは当てやすいので、間に基準を置く。
# (diagnose_confidence.pyの実測で15-19%帯が実績的中率32.5%だったことを踏まえた目安値)
RACE_TIER_EXACTA_THRESHOLDS = [
    ("A", 0.40),
    ("B", 0.28),
    ("C", 0.18),
]
# 3連単(120通り)は最も当てにくく、予測確率自体が10%を超えることも稀
# (diagnose_confidence.pyの実測で5-9%帯が実績的中率12.0%だったことを踏まえた目安値)
RACE_TIER_BET_THRESHOLDS = [
    ("A", 0.18),
    ("B", 0.12),
    ("C", 0.06),
]
RACE_TIER_MIN_SAMPLE = 20  # これ未満のサンプルしかないバケットは実績を信頼せず、その軸の判定はスキップする

# 実績データが少ない(まだ判定できない)場合の暫定フォールバック(単勝軸のみ)で使う
# gap(1位・2位の予測確率差)の基準点。classify_race_tier内でfallback_anchorsとして直接使用。
# 1位と2位の確率差が8pt以上ならC境界、20pt以上ならB境界相当(A相当は暫定判定では出さない)。
# 実績が溜まればcalibrationベースの本判定に置き換わる。

# 荒れ水面(波高・風速がこの値以上)は、判定に関わらず1段階格下げする
# (score_boat()の「荒れ水面だと1号艇が不利になりやすい」というロジックと同じ考え方)。
ROUGH_WATER_WAVE_CM = 3
ROUGH_WATER_WIND_MS = 5
_TIER_ORDER = ["A", "B", "C", "D"]

# 単勝・2連単・3連単それぞれの軸の重み(重み付き平均のウェイト)。
# 3連単は実際に購入する対象(推奨3連単)なので、他の2軸よりやや重めに配分している。
RACE_TIER_WEIGHTS = {"win": 0.3, "exacta": 0.3, "bet": 0.4}

# 下限ルール: 加重平均で他の軸に助けられても、いずれかの軸の実績的中率がここまで低ければ
# 「ほぼ運任せと変わらない」とみなし、他がどれだけ強くても問答無用でDにする。
# Dを「見送りの目安」として厳格に保つための安全弁(比較として、6艇均等なら単勝16.7%、
# 2連単は30通りで3.3%、3連単は120通りで0.83%が完全にランダムな場合の的中率)。
RACE_TIER_FLOOR = {"win": 0.20, "exacta": 0.05, "bet": 0.02}


def _downgrade_tier(tier: str) -> str:
    idx = _TIER_ORDER.index(tier)
    return _TIER_ORDER[min(idx + 1, len(_TIER_ORDER) - 1)]


def _continuous_score(value: float, anchors: list) -> float:
    """
    (基準値, スコア)の組(昇順)を区分線形で補間し、valueに対応する連続的なスコアを返す。
    例えば単勝の実績的中率が「Aの基準(60%)をギリギリ超えた61%」なのか
    「90%と余裕で超えている」のかで、スコアに差を付けるために使う。
    これをやらずに先にA/B/C/Dの4段階へ丸めてから平均すると、結局
    「自信ありバッジが何個点灯しているか」を数えているのとほぼ同じになってしまうため。
    range外は両端の値でクリップする。
    """
    if value <= anchors[0][0]:
        return anchors[0][1]
    if value >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return y1
            frac = (value - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return anchors[-1][1]  # 到達しないはずだが念のため


def _classify_by_calibration(calibration: dict, pct: float, thresholds: list,
                              bucket_fn=_bucket_label) -> Optional[dict]:
    """calibration(予測確率帯ごとの実績的中率)から、指定した閾値でA〜Dを判定する
    (人が読む理由テキスト用。最終ティアの計算自体は連続スコアの方を使う)。
    サンプル不足で判定できなければNoneを返す(呼び出し側でフォールバックに回す)。"""
    label = bucket_fn(pct)
    stat = calibration.get(label)
    if not stat or stat["sample_size"] < RACE_TIER_MIN_SAMPLE or stat["hit_rate"] is None:
        return None
    tier = "D"
    for t, threshold in thresholds:
        if stat["hit_rate"] >= threshold:
            tier = t
            break
    return {"tier": tier, "label": label, "hit_rate": stat["hit_rate"], "sample_size": stat["sample_size"]}


def _anchors_from_thresholds(thresholds: list, max_score: float = 4.0) -> list:
    """[("A",a),("B",b),("C",c)]の閾値リストから、_continuous_score用の
    (基準値, スコア)アンカー列を作る。D=1点を原点(0)に置き、C→2点、B→3点、A→max_scoreとする。"""
    by_tier = dict(thresholds)
    scores = {"C": 2.0, "B": 3.0, "A": max_score}
    anchors = [(0.0, 1.0)]
    for t in ("C", "B", "A"):
        if t in by_tier:
            anchors.append((by_tier[t], scores[t]))
    return anchors


def _score_to_tier(score: float) -> str:
    if score >= 3.5:
        return "A"
    if score >= 2.5:
        return "B"
    if score >= 1.5:
        return "C"
    return "D"


def classify_race_tier(calibration: dict, top1_pct: float, second_pct: float = None,
                        wave_height_cm: float = None, wind_speed_m: float = None,
                        exacta_calibration: dict = None, top_exacta_prob_pct: float = None,
                        bet_calibration: dict = None, top_bet_prob_pct: float = None) -> dict:
    """
    レース単位の信頼度をA(高信頼度)〜D(荒れ要素が強い/見送り)で判定する。

    単勝・2連単・3連単、それぞれの実績的中率を(A/B/C/Dの4段階に丸めるのではなく)
    _continuous_score()で連続的なスコア(1〜4点)に変換してから、RACE_TIER_WEIGHTSで
    重み付き平均を取り、最後に1回だけ最終ティアへ丸める。
    先に4段階へ丸めてから平均すると、結局「自信ありバッジが何個点灯しているか」を
    数えているのとほぼ同じになってしまう(「ギリギリ基準超え」も「大幅に基準超え」も
    同じ点数になるため)ので、それを避けるために連続スコアのまま合成している。

    ただし、加重平均だけだと「1つの軸がほぼ運任せレベルまで弱くても、他の軸が
    強ければDを回避できてしまう」ため、Dを「見送りの目安」として厳格に保つ安全弁として、
    いずれかの軸の実績的中率がRACE_TIER_FLOORを下回っていれば、平均計算を待たず
    問答無用でDにする(下限ルール)。

    単勝は実績データが不十分な場合、1位・2位の予測確率差で暫定判定する(2連単・3連単は
    暫定判定を持たず、単にその軸を平均から除外する。下限ルールも実績データがある軸にしか
    適用されない)。
    最後に、荒れ水面(波・風がROUGH_WATER_*以上)なら1段階格下げする。

    戻り値: {"tier": "A", "reason": "...", "basis": "calibration" or "fallback"}
    """
    floor_triggered = False
    win_result = _classify_by_calibration(calibration, top1_pct, RACE_TIER_WIN_THRESHOLDS)
    if win_result:
        win_score = _continuous_score(win_result["hit_rate"], _anchors_from_thresholds(RACE_TIER_WIN_THRESHOLDS))
        reasons = [f"単勝{win_result['label']}帯の実績的中率{win_result['hit_rate']:.1%}(n={win_result['sample_size']})"]
        basis = "calibration"
        if win_result["hit_rate"] < RACE_TIER_FLOOR["win"]:
            floor_triggered = True
    else:
        gap = (top1_pct - second_pct) if second_pct is not None else 0
        # フォールバックはA相当を出さない(実績データに基づかない暫定判定のため上限3点=B相当まで)
        fallback_anchors = [(0.0, 1.0), (8.0, 2.0), (20.0, 3.0)]  # gap=8pt→C境界、20pt→B境界
        win_score = _continuous_score(gap, fallback_anchors)
        reasons = [f"単勝の実績データ不足のため暫定判定: 1位-2位の確率差{gap:.0f}pt"]
        basis = "fallback"

    weighted_sum = win_score * RACE_TIER_WEIGHTS["win"]
    weight_total = RACE_TIER_WEIGHTS["win"]

    if exacta_calibration is not None and top_exacta_prob_pct is not None:
        exacta_result = _classify_by_calibration(exacta_calibration, top_exacta_prob_pct,
                                                   RACE_TIER_EXACTA_THRESHOLDS, bucket_fn=_exacta_bucket_label)
        if exacta_result:
            exacta_score = _continuous_score(exacta_result["hit_rate"],
                                              _anchors_from_thresholds(RACE_TIER_EXACTA_THRESHOLDS))
            weighted_sum += exacta_score * RACE_TIER_WEIGHTS["exacta"]
            weight_total += RACE_TIER_WEIGHTS["exacta"]
            reasons.append(f"2連単{exacta_result['label']}帯の実績的中率{exacta_result['hit_rate']:.1%}"
                            f"(n={exacta_result['sample_size']})")
            if exacta_result["hit_rate"] < RACE_TIER_FLOOR["exacta"]:
                floor_triggered = True

    if bet_calibration is not None and top_bet_prob_pct is not None:
        bet_result = _classify_by_calibration(bet_calibration, top_bet_prob_pct,
                                               RACE_TIER_BET_THRESHOLDS, bucket_fn=_bet_bucket_label)
        if bet_result:
            bet_score = _continuous_score(bet_result["hit_rate"],
                                           _anchors_from_thresholds(RACE_TIER_BET_THRESHOLDS))
            weighted_sum += bet_score * RACE_TIER_WEIGHTS["bet"]
            weight_total += RACE_TIER_WEIGHTS["bet"]
            reasons.append(f"3連単{bet_result['label']}帯の実績的中率{bet_result['hit_rate']:.1%}"
                            f"(n={bet_result['sample_size']})")
            if bet_result["hit_rate"] < RACE_TIER_FLOOR["bet"]:
                floor_triggered = True

    avg_score = weighted_sum / weight_total if weight_total else win_score
    tier = _score_to_tier(avg_score)

    reason = " / ".join(reasons)

    if floor_triggered and tier != "D":
        tier = "D"
        reason += " / いずれかの軸がほぼ運任せレベル(下限ルール)のためD"

    rough_water = (wave_height_cm is not None and wave_height_cm >= ROUGH_WATER_WAVE_CM) or \
                  (wind_speed_m is not None and wind_speed_m >= ROUGH_WATER_WIND_MS)
    if rough_water and tier != "D":
        tier = _downgrade_tier(tier)
        reason += " / 荒れ水面のため1段階格下げ"

    return {"tier": tier, "reason": reason, "basis": basis}
