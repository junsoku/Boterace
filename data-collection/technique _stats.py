"""
蓄積した results テーブル(実際の着順・進入コース・決まり手)から、
場ごと・コース別の決まり手統計(逃げ率・まくり率・差し率・まくり差し率・逃し率)を計算する。

「ボートレース日和」のようなサイトが表示している指標と同種のものだが、
自前のデータ(ingest.pyで蓄積したresults)から算出するため、スクレイピング不要。

※ 注意: ある程度のレース数(理想は数千レース単位)が蓄積されないと統計として不安定。
   データが少ない場合は STADIUM_DEFAULT_COURSE_STATS の一般的な目安値にフォールバックする。
   この目安値はボートレース全体の大まかな傾向であり、正確な公式統計ではないため、
   実データが十分に溜まったら必ず compute_course_technique_rates() の結果に置き換えること。
"""
import sqlite3
from collections import defaultdict
from typing import Optional

# 決まり手のカテゴリ(結果データに入りうる代表的な値)
TECHNIQUES = ["逃げ", "差し", "まくり", "まくり差し", "抜き", "恵まれ"]

# 場ごとの実データが不十分な場合のフォールバック値(全国平均的な大まかな傾向の目安)
# コース番号 -> {決まり手: 発生率(そのコースが絡んだ勝利のうちの割合ではなく、全レースに対する勝率)}
STADIUM_DEFAULT_COURSE_STATS = {
    1: {"win_rate": 0.55, "逃げ": 0.47, "差し": 0.03, "まくり": 0.02, "その他": 0.03},
    2: {"win_rate": 0.14, "逃げ": 0.00, "差し": 0.10, "まくり": 0.02, "その他": 0.02},
    3: {"win_rate": 0.12, "逃げ": 0.00, "差し": 0.05, "まくり": 0.05, "その他": 0.02},
    4: {"win_rate": 0.11, "逃げ": 0.00, "差し": 0.03, "まくり": 0.06, "その他": 0.02},
    5: {"win_rate": 0.05, "逃げ": 0.00, "差し": 0.01, "まくり": 0.03, "その他": 0.01},
    6: {"win_rate": 0.03, "逃げ": 0.00, "差し": 0.01, "まくり": 0.01, "その他": 0.01},
}
MIN_SAMPLE_SIZE = 200  # このレース数未満ならフォールバック値を使う


def compute_course_technique_rates(conn: sqlite3.Connection, stadium_number: Optional[int] = None) -> dict:
    """
    コース別(1〜6)の勝率・決まり手内訳を集計する。
    stadium_number を指定するとその場だけ、Noneなら全場合算。
    戻り値: {course: {"win_rate":.., "逃げ":.., "差し":.., ... , "sample_size":n}}
    """
    where = "WHERE r.actual_course IS NOT NULL"
    params = []
    if stadium_number is not None:
        where += " AND races.stadium_number = ?"
        params.append(stadium_number)

    rows = conn.execute(
        f"""
        SELECT r.actual_course, r.arrival_order, r.winning_technique
        FROM results r
        JOIN entries e ON e.entry_id = r.entry_id
        JOIN races ON races.race_id = e.race_id
        {where}
        """,
        params,
    ).fetchall()

    total_by_course = defaultdict(int)
    win_by_course = defaultdict(int)
    technique_win_by_course = defaultdict(lambda: defaultdict(int))

    for course, arrival_order, technique in rows:
        if course is None:
            continue
        total_by_course[course] += 1
        if arrival_order == 1:
            win_by_course[course] += 1
            key = technique if technique in TECHNIQUES else "その他"
            technique_win_by_course[course][key] += 1

    result = {}
    for course in range(1, 7):
        n = total_by_course.get(course, 0)
        if n < MIN_SAMPLE_SIZE:
            result[course] = dict(STADIUM_DEFAULT_COURSE_STATS[course], sample_size=n, is_fallback=True)
            continue
        wins = win_by_course.get(course, 0)
        stat = {"win_rate": wins / n, "sample_size": n, "is_fallback": False}
        for tech in TECHNIQUES:
            stat[tech] = technique_win_by_course[course].get(tech, 0) / n
        result[course] = stat
    return result


def compute_racer_nigashi_rate(conn: sqlite3.Connection, racer_registration_number: int) -> Optional[dict]:
    """
    特定選手の「逃し率」(1コースからスタートしたのに勝ちきれなかった率)を計算する。
    十分なサンプルがなければ None を返す(呼び出し側で場の平均にフォールバックする想定)。
    """
    rows = conn.execute(
        """
        SELECT r.arrival_order
        FROM results r
        JOIN entries e ON e.entry_id = r.entry_id
        WHERE e.racer_registration_number = ? AND r.actual_course = 1
        """,
        (racer_registration_number,),
    ).fetchall()

    n = len(rows)
    if n < 20:  # 個人統計は場の統計よりサンプルが集まりにくいので閾値を下げる
        return None
    wins = sum(1 for (arrival,) in rows if arrival == 1)
    return {"nigashi_rate": 1 - (wins / n), "sample_size": n}


def course_advantage_score(course_stats: dict, course: int) -> float:
    """
    course_technique_rates() の結果から、そのコースの「勝ちやすさ」を単一スコアに圧縮する。
    score_boat() 側で他の指標と合成しやすいよう 0〜10 程度のスケールにする。
    """
    stat = course_stats.get(course)
    if not stat:
        return 3.0
    return stat["win_rate"] * 10
