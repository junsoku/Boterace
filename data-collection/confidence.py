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

from export_today import score_boat, normalize_to_pct, estimate_bets, predict_with_model
from technique_stats import compute_course_technique_rates, compute_racer_nigashi_rate

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
# 「自信あり」の目安にする。実績が溜まってきたら調整して良い。
BET_CONFIDENCE_THRESHOLD = 0.15


def _bucket_label(pct: int) -> str:
    lower = (pct // BUCKET_SIZE) * BUCKET_SIZE
    upper = lower + BUCKET_SIZE - 1
    return f"{lower}-{upper}%"


def _bet_bucket_label(pct: float) -> str:
    lower = (int(pct) // BET_BUCKET_SIZE) * BET_BUCKET_SIZE
    upper = lower + BET_BUCKET_SIZE - 1
    return f"{lower}-{upper}%"


def _iter_calibration_races(conn: sqlite3.Connection, lookback_days: int, model=None):
    """過去lookback_days日分の、結果が確定しているレースを1件ずつ再予想しながり返す共通処理。
    compute_confidence_calibration / compute_bet_confidence_calibration の両方から使う。

    model が渡されていれば、本番(export_today.py)と同じ predict_with_model()(MLモデル)で
    再予想する。model が None の場合のみ、旧来のヒューリスティック(score_boat)にフォールバックする。
    確信度の実績値は「本番で実際に使っている予想ロジック」と揃っていて初めて意味を持つため、
    model があるなら必ずそちらを使うべき。

    yield: (boats[{"lane":.., "pct":..}], actual_order[boat_numberを着順順に並べたリスト])
    """
    end = date.today()
    start = end - timedelta(days=lookback_days)

    races = conn.execute(
        """SELECT race_id, stadium_number, race_grade, wave_height_cm, wind_speed_m
           FROM races WHERE race_date BETWEEN ? AND ?""",
        (start.isoformat(), end.isoformat()),
    ).fetchall()

    course_stats_cache = {}

    for race_id, stadium_number, race_grade, wave, wind in races:
        entries = conn.execute(
            """SELECT e.boat_number, e.racer_registration_number, e.racer_class,
                      e.national_win_rate, e.national_2連率, e.local_win_rate, e.local_2連率,
                      e.motor_2連率, e.boat_hull_2連率, e.average_start_timing,
                      e.flying_count, e.late_count,
                      p.exhibition_time, p.tilt_angle, r.arrival_order
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
                         "avg_exhibition_time": avg_exh}

        boat_dicts = []
        for (bn, reg_no, racer_class, nat, nat_2r, local, local_2r, motor_2r, hull_2r, avg_st,
             flying, late, exh, tilt_angle, arrival_order) in entries:
            d = {
                "boat_number": bn, "racer_class": racer_class,
                "national_win_rate": nat, "national_2連率": nat_2r,
                "local_win_rate": local, "local_2連率": local_2r,
                "motor_2連率": motor_2r, "boat_hull_2連率": hull_2r,
                "average_start_timing": avg_st,
                "flying_count": flying, "late_count": late,
                "exhibition_time": exh, "tilt_angle": tilt_angle,
            }
            if bn == 1 and reg_no is not None:
                nigashi = compute_racer_nigashi_rate(conn, reg_no)
                if nigashi:
                    d["nigashi_rate"] = nigashi["nigashi_rate"]
            boat_dicts.append((d, arrival_order))

        if model is not None:
            scores = predict_with_model([d for d, _ in boat_dicts], course_stats, model, race_context)
            pcts = normalize_to_pct(scores, use_softmax=False)
        else:
            scores = [score_boat(d, course_stats, race_context) for d, _ in boat_dicts]
            pcts = normalize_to_pct(scores, use_softmax=True)

        boats = [{"lane": d["boat_number"], "pct": pct} for (d, _), pct in zip(boat_dicts, pcts)]
        actual_order = [d["boat_number"] for (d, arrival) in
                         sorted(boat_dicts, key=lambda x: x[1])]

        yield boats, actual_order


def compute_confidence_calibration(conn: sqlite3.Connection, lookback_days: int = 90, model=None) -> dict:
    """
    過去lookback_days日分の結果確定レースを、現在のロジック(modelがあればMLモデル)で再予想し、
    ◎の予想確率帯ごとの実際の的中率を集計する。
    戻り値: {"30-39%": {"hit_rate":0.55,"sample_size":18}, ...}
    """
    buckets = {}  # label -> [hits, total]

    for boats, actual_order in _iter_calibration_races(conn, lookback_days, model):
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
    """
    buckets = {}  # label -> [hits, total]

    for boats, actual_order in _iter_calibration_races(conn, lookback_days, model):
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
