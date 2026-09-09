"""
過去の結果が確定済みのレースについて、「◎(予想1位)の確率が何%だった時、
実際にどのくらいの割合で当たっていたか」を集計するモジュール。

例えば「◎の予想確率が40%台だったレースは、過去に実際65%当たっている」といった
実績が分かれば、今日のレースでも同じ確率帯なら信頼して良い、という判断ができる。
これはオッズが無くても作れる「当たりやすさの実績値」。

使い方(export_today.py内から呼び出す想定):
    from confidence import compute_confidence_calibration, lookup_confidence
    calibration = compute_confidence_calibration(conn)
    conf = lookup_confidence(calibration, top_pick_pct)
    # conf = {"bucket": "40-49%", "hit_rate": 0.65, "sample_size": 42, "is_confident": True}
"""
import sqlite3
from datetime import date, timedelta

from export_today import score_boat, normalize_to_pct
from technique_stats import compute_course_technique_rates, compute_racer_nigashi_rate

# 予想確率を10%刻みでグループ化する(サンプルが集まりやすいよう粗めの区切り)
BUCKET_SIZE = 10
MIN_SAMPLE_FOR_STAR = 20   # このサンプル数未満のバケットは信頼度を判定しない
CONFIDENCE_THRESHOLD = 0.60  # このバケットの過去的中率がこれ以上なら「星」を付ける


def _bucket_label(pct: int) -> str:
    lower = (pct // BUCKET_SIZE) * BUCKET_SIZE
    upper = lower + BUCKET_SIZE - 1
    return f"{lower}-{upper}%"


def compute_confidence_calibration(conn: sqlite3.Connection, lookback_days: int = 90) -> dict:
    """
    過去lookback_days日分の結果確定レースを、現在のロジックで再予想し、
    ◎の予想確率帯ごとの実際の的中率を集計する。
    戻り値: {"30-39%": {"hit_rate":0.55,"sample_size":18}, ...}
    """
    end = date.today()
    start = end - timedelta(days=lookback_days)

    races = conn.execute(
        """SELECT race_id, stadium_number, race_grade, wave_height_cm, wind_speed_m
           FROM races WHERE race_date BETWEEN ? AND ?""",
        (start.isoformat(), end.isoformat()),
    ).fetchall()

    course_stats_cache = {}
    buckets = {}  # label -> [hits, total]

    for race_id, stadium_number, race_grade, wave, wind in races:
        entries = conn.execute(
            """SELECT e.boat_number, e.racer_registration_number, e.racer_class,
                      e.national_win_rate, e.national_2連率, e.local_win_rate, e.local_2連率,
                      e.motor_2連率, e.boat_hull_2連率, e.flying_count, e.late_count,
                      p.exhibition_time, r.arrival_order
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

        exh_values = [row[11] for row in entries if row[11]]
        avg_exh = sum(exh_values) / len(exh_values) if exh_values else None
        race_context = {"wave_height_cm": wave, "wind_speed_m": wind, "race_grade": race_grade,
                         "avg_exhibition_time": avg_exh}

        boat_dicts = []
        for (bn, reg_no, racer_class, nat, nat_2r, local, local_2r, motor_2r, hull_2r,
             flying, late, exh, arrival_order) in entries:
            d = {
                "boat_number": bn, "racer_class": racer_class,
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
        (top_d, top_arrival), top_pct = ranked[0]

        label = _bucket_label(top_pct)
        if label not in buckets:
            buckets[label] = [0, 0]
        buckets[label][1] += 1
        if top_arrival == 1:
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
