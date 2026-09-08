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
    _migrate_add_missing_columns(conn)
    return conn


def _migrate_add_missing_columns(conn: sqlite3.Connection):
    """schema.sqlに列を追加した後、既存DBにも反映させるための簡易マイグレーション。
    CREATE TABLE IF NOT EXISTS は既存テーブルへの列追加はしてくれないため、
    ALTER TABLE ADD COLUMN を個別に試し、「既にある」エラーは無視する。
    """
    migrations = [
        ("entries", "flying_count", "INTEGER"),
        ("entries", "late_count", "INTEGER"),
    ]
    for table, column, coltype in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise  # 列名重複以外のエラーは想定外なので伝播させる


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
                        average_start_timing, flying_count, late_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    pick(boat, "racer_flying_count", "racer_boat_flying_count", "flying_count"),
                    pick(boat, "racer_late_count", "racer_boat_late_count", "late_count"),
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


def parse_previews(conn: sqlite3.Connection, race_date: date, payload: Optional[dict]):
    """直前情報JSON -> previews テーブル(選手ごと) + races テーブルの天候欄へ格納"""
    if not payload:
        return
    races = pick(payload, "results", "races", default=[])
    for race in races:
        stadium_number = pick(race, "race_stadium_number", "stadium_number")
        race_number = pick(race, "race_number")

        # 天候はレース単位(6艇共通)の情報なので、races テーブル側を更新する
        conn.execute(
            """UPDATE races SET
                 weather=?, wind_direction=?, wind_speed_m=?, wave_height_cm=?,
                 temperature_c=?, water_temperature_c=?
               WHERE race_date=? AND stadium_number=? AND race_number=?""",
            (
                pick(race, "race_weather_condition", "weather"),
                pick(race, "race_wind_direction_number", "wind_direction"),
                pick(race, "race_wind_velocity", "wind_speed_m"),
                pick(race, "race_wave_height", "wave_height_cm"),
                pick(race, "race_temperature", "temperature_c"),
                pick(race, "race_water_temperature", "water_temperature_c"),
                race_date.isoformat(), stadium_number, race_number,
            ),
        )

        boats = pick(race, "boats", "entries", default=[])
        for boat in boats:
            boat_number = pick(boat, "racer_boat_number", "boat_number")
            entry_id = get_entry_id(conn, race_date, stadium_number, race_number, boat_number)
            if entry_id is None:
                continue  # 出走表が先に取り込まれていない場合はスキップ
            conn.execute(
                """INSERT INTO previews (entry_id, exhibition_time, tilt_angle,
                        weight_adjustment_kg, start_course, start_timing_preview,
                        parts_exchanged)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(entry_id) DO UPDATE SET
                     exhibition_time=excluded.exhibition_time,
                     tilt_angle=excluded.tilt_angle,
                     start_course=excluded.start_course,
                     start_timing_preview=excluded.start_timing_preview""",
                (
                    entry_id,
                    pick(boat, "racer_exhibition_time", "exhibition_time"),
                    pick(boat, "racer_tilt", "tilt_angle"),
                    pick(boat, "racer_weight_adjustment", "weight_adjustment_kg"),
                    pick(boat, "racer_course_number", "start_course"),
                    pick(boat, "racer_start_timing", "start_timing_preview"),
                    pick(boat, "racer_parts_exchanged", "parts_exchanged"),
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
        parse_previews(conn, d, day_data["previews"])
        parse_results(conn, d, day_data["results"])

    conn.close()
    print("完了しました。")


if __name__ == "__main__":
    main()
