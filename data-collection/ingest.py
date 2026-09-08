"""
BoatraceOpenAPIから出走表・直前情報・結果を取得し、SQLiteに保存するメインスクリプト。

使い方:
    python ingest.py --start 2026-08-01 --end 2026-08-31 --db boatrace.db

設計方針:
    1. 生JSONは必ず raw_json テーブルにそのまま保存する
       (パースのキー名が実際のAPI仕様とズレていても、後から再パースできるようにするため)
    2. 正規化テーブル(races/entries/previews/results)への変換はベストエフォート。
       このスクリプトを書いた時点ではAPIレスポンスを実機取得して検証できていないため、
       フィールド名は BoatraceOpenAPI の一般的な命名慣習からの推測が含まれます。
       初回実行時は必ず --inspect オプションでJSON構造を確認し、
       parse_programs() 等の pick() 呼び出しのキー名を実データに合わせて調整してください。
"""
import argparse
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from boatrace_client import fetch_day, date_range


def pick(d: dict, *candidates: str, default: Any = None) -> Any:
    """複数の想定キー名から最初に見つかった値を返す(API側の命名揺れに対応)"""
    for key in candidates:
        if key in d:
            return d[key]
    return default


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    schema = Path(__file__).parent / "schema.sql"
    conn.executescript(schema.read_text(encoding="utf-8"))
    conn.commit()
    return conn


def save_raw(conn: sqlite3.Connection, race_date: date, kind: str, payload: Optional[dict]):
    if payload is None:
        return
    conn.execute(
        """INSERT INTO raw_json (race_date, kind, fetched_at, payload)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(race_date, kind) DO UPDATE SET
             fetched_at=excluded.fetched_at, payload=excluded.payload""",
        (race_date.isoformat(), kind, datetime.now().isoformat(), json.dumps(payload, ensure_ascii=False)),
    )


def parse_programs(conn: sqlite3.Connection, race_date: date, payload: Optional[dict]):
    """出走表JSON -> races / entries テーブルへ格納"""
    if not payload:
        return
    races = pick(payload, "results", "programs", "races", default=[])
    for race in races:
        stadium_number = pick(race, "race_stadium_number", "stadium_number")
        race_number = pick(race, "race_number")
        if stadium_number is None or race_number is None:
            continue
        cur = conn.execute(
            """INSERT INTO races (race_date, stadium_number, race_number, race_title,
                                   race_grade, close_at, distance_m)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(race_date, stadium_number, race_number) DO UPDATE SET
                 race_title=excluded.race_title, race_grade=excluded.race_grade,
                 close_at=excluded.close_at, distance_m=excluded.distance_m""",
            (
                race_date.isoformat(), stadium_number, race_number,
                pick(race, "race_title", "title"),
                pick(race, "race_grade_number", "race_grade"),
                pick(race, "race_closed_at", "close_at"),
                pick(race, "race_distance", "distance_m", default=1800),
            ),
        )
        race_id = conn.execute(
            "SELECT race_id FROM races WHERE race_date=? AND stadium_number=? AND race_number=?",
            (race_date.isoformat(), stadium_number, race_number),
        ).fetchone()[0]

        boats = pick(race, "boats", "entries", default=[])
        for boat in boats:
            boat_number = pick(boat, "racer_boat_number", "boat_number")
            if boat_number is None:
                continue
            conn.execute(
                """INSERT INTO entries (race_id, boat_number, racer_registration_number,
                        racer_name, racer_branch, racer_class, racer_age, racer_weight_kg,
                        national_win_rate, national_2連率, local_win_rate, local_2連率,
                        motor_number, motor_2連率, boat_hull_number, boat_hull_2連率,
                        average_start_timing)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(race_id, boat_number) DO NOTHING""",
                (
                    race_id, boat_number,
                    pick(boat, "racer_number", "racer_registration_number"),
                    pick(boat, "racer_name"),
                    pick(boat, "racer_branch_number", "racer_branch"),
                    pick(boat, "racer_class_number", "racer_class"),
                    pick(boat, "racer_age"),
                    pick(boat, "racer_weight"),
                    pick(boat, "racer_national_top_1_percent", "national_win_rate"),
                    pick(boat, "racer_national_top_2_percent", "national_2連率"),
                    pick(boat, "racer_local_top_1_percent", "local_win_rate"),
                    pick(boat, "racer_local_top_2_percent", "local_2連率"),
                    pick(boat, "racer_assigned_motor_number", "motor_number"),
                    pick(boat, "racer_assigned_motor_top_2_percent", "motor_2連率"),
                    pick(boat, "racer_assigned_boat_number", "boat_hull_number"),
                    pick(boat, "racer_assigned_boat_top_2_percent", "boat_hull_2連率"),
                    pick(boat, "racer_average_start_timing", "average_start_timing"),
                ),
            )
    conn.commit()


def get_entry_id(conn: sqlite3.Connection, race_date: date, stadium_number, race_number, boat_number) -> Optional[int]:
    row = conn.execute(
        """SELECT e.entry_id FROM entries e
           JOIN races r ON r.race_id = e.race_id
           WHERE r.race_date=? AND r.stadium_number=? AND r.race_number=? AND e.boat_number=?""",
        (race_date.isoformat(), stadium_number, race_number, boat_number),
    ).fetchone()
    return row[0] if row else None


def parse_results(conn: sqlite3.Connection, race_date: date, payload: Optional[dict]):
    """結果JSON -> results テーブルへ格納(先にentriesが存在している必要あり)"""
    if not payload:
        return
    races = pick(payload, "results", "races", default=[])
    for race in races:
        stadium_number = pick(race, "race_stadium_number", "stadium_number")
        race_number = pick(race, "race_number")
        boats = pick(race, "boats", "entries", default=[])
        for boat in boats:
            boat_number = pick(boat, "racer_boat_number", "boat_number")
            entry_id = get_entry_id(conn, race_date, stadium_number, race_number, boat_number)
            if entry_id is None:
                continue  # 出走表が先に取り込まれていない場合はスキップ
            conn.execute(
                """INSERT INTO results (entry_id, arrival_order, actual_course,
                        actual_start_timing, race_time, winning_technique, remarks)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(entry_id) DO UPDATE SET
                     arrival_order=excluded.arrival_order""",
                (
                    entry_id,
                    pick(boat, "racer_place_number", "arrival_order"),
                    pick(boat, "racer_course_number", "actual_course"),
                    pick(boat, "racer_start_timing", "actual_start_timing"),
                    pick(boat, "race_time"),
                    pick(boat, "winning_technique"),
                    pick(boat, "remarks"),
                ),
            )
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--db", default="boatrace.db")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    conn = init_db(args.db)

    for d in date_range(start, end):
        print(f"取得中: {d}")
        day_data = fetch_day(d)
        save_raw(conn, d, "programs", day_data["programs"])
        save_raw(conn, d, "previews", day_data["previews"])
        save_raw(conn, d, "results", day_data["results"])
        conn.commit()

        parse_programs(conn, d, day_data["programs"])
        parse_results(conn, d, day_data["results"])
        # previewsのパースはprogramsと同様のパターンで追加可能(必要になったら実装)

    conn.close()
    print("完了しました。")


if __name__ == "__main__":
    main()
