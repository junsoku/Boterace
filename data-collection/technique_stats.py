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

RECENT_FORM_N_RACES = 10   # 「直近の調子」として遡る走数の既定値
RECENT_FORM_MIN_SAMPLE = 5  # これ未満の走数しか無ければ「調子」は計算しない(不安定なため)


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


def fetch_racer_history(conn: sqlite3.Connection, racer_registration_number: int) -> list:
    """
    選手の全レース結果を (race_date, race_id, arrival_order) のリストで、日付昇順に返す。
    build_features.py が同じ選手について何度も呼び出す(1選手が複数レースに出走するため)ことを
    想定し、DBへの問い合わせは選手ごとに1回だけで済むようにしている
    (呼び出し側でrecent_form_from_history()に渡して都度スライスする使い方を想定)。
    """
    rows = conn.execute(
        """
        SELECT races.race_date, races.race_id, r.arrival_order
        FROM results r
        JOIN entries e ON e.entry_id = r.entry_id
        JOIN races ON races.race_id = e.race_id
        WHERE e.racer_registration_number = ? AND r.arrival_order IS NOT NULL
        ORDER BY races.race_date, races.race_id
        """,
        (racer_registration_number,),
    ).fetchall()
    return rows


def recent_form_from_history(history: list, before_date: str, before_race_id: Optional[int] = None,
                              n_races: int = RECENT_FORM_N_RACES) -> Optional[dict]:
    """
    fetch_racer_history() で取得済みの履歴から、「before_date(・before_race_id)より前」の
    直近n_races走だけを切り出して平均着順・勝率を計算する。

    未来のレースの結果が紛れ込まないよう、race_dateで厳密に区切っている
    (同日開催の複数レースまでは区別していない簡易実装。同日中の先着順までは見ていないため、
    ごく僅かに同日レース分の情報が前後する可能性があるが、実運用上の影響は小さい)。
    十分なサンプルがなければ None を返す。
    """
    if before_race_id is not None:
        past = [h for h in history if (h[0], h[1]) < (before_date, before_race_id)]
    else:
        past = [h for h in history if h[0] < before_date]

    if len(past) < RECENT_FORM_MIN_SAMPLE:
        return None
    recent = past[-n_races:]
    n = len(recent)
    avg_order = sum(row[2] for row in recent) / n
    win_rate = sum(1 for row in recent if row[2] == 1) / n
    return {"avg_arrival_order": avg_order, "recent_win_rate": win_rate, "sample_size": n}


def compute_racer_recent_form(conn: sqlite3.Connection, racer_registration_number: int,
                               before_date: Optional[str] = None,
                               n_races: int = RECENT_FORM_N_RACES) -> Optional[dict]:
    """
    選手の直近n_races走(before_dateより前。Noneなら全期間の最新n_races走)の
    平均着順・勝率を1回のクエリで計算する。

    export_today.py・confidence.py のように「1レースにつき選手6人分」程度の呼び出し頻度なら
    このままで十分軽い。build_features.py のように同じ選手を何百行にもわたって扱う場合は、
    fetch_racer_history() + recent_form_from_history() の組み合わせ(選手ごとに1クエリ)を使うこと。
    """
    where = "WHERE e.racer_registration_number = ? AND r.arrival_order IS NOT NULL"
    params = [racer_registration_number]
    if before_date is not None:
        where += " AND races.race_date < ?"
        params.append(before_date)

    rows = conn.execute(
        f"""
        SELECT r.arrival_order
        FROM results r
        JOIN entries e ON e.entry_id = r.entry_id
        JOIN races ON races.race_id = e.race_id
        {where}
        ORDER BY races.race_date DESC, races.race_id DESC
        LIMIT ?
        """,
        params + [n_races],
    ).fetchall()

    n = len(rows)
    if n < RECENT_FORM_MIN_SAMPLE:
        return None
    avg_order = sum(row[0] for row in rows) / n
    win_rate = sum(1 for row in rows if row[0] == 1) / n
    return {"avg_arrival_order": avg_order, "recent_win_rate": win_rate, "sample_size": n}
