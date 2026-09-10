"""
BoatraceOpenAPI 統合API(v1)から出走表・直前情報・結果・払戻を取得し、SQLiteに保存する
メインスクリプト。

使い方:
    python ingest.py --start 2026-08-01 --end 2026-08-31 --db boatrace.db

設計方針:
    1. 生JSONは必ず raw_json テーブルにそのまま保存する(再パースできるようにするため)
    2. 項目名は公式スキーマ文書(下記)で確認済みの正確なものを使っている(推測ではない)
       https://github.com/boatraceopenapi/api/blob/gh-pages/docs/v1/schema.md
    3. `_source` 付きの項目(級別・天候・グレードなど)は、コード化された数値ではなく
       スクレイピング元の人間可読な文字列(例: "A1"、"晴")をそのまま使う。
"""
import argparse
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from boatrace_client import fetch_unified, date_range

# 決まり手番号 -> 名称(公式サイトの表記に準拠。技術情報が数値コードのみのため変換に使用)
TECHNIQUE_NAMES = {1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり差し", 5: "抜き", 6: "恵まれ"}

# 級別番号 -> 名称。rank_number_source(スクレイピング元の文字列)が取れない場合のフォールバック変換
RANK_NAMES = {1: "A1", 2: "A2", 3: "B1", 4: "B2"}


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    schema = Path(__file__).parent / "schema.sql"
    conn.executescript(schema.read_text(encoding="utf-8"))
    conn.commit()
    _migrate_add_missing_columns(conn)
    return conn


def _migrate_add_missing_columns(conn: sqlite3.Connection):
    """schema.sqlに列を追加した後、既存DBにも反映させるための簡易マイグレーション。"""
    migrations = [
        ("entries", "flying_count", "INTEGER"),
        ("entries", "late_count", "INTEGER"),
        ("prediction_log", "top_bets_json", "TEXT"),
        ("prediction_log", "bet_is_confident", "INTEGER"),
    ]
    for table, column, coltype in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise

    # payoutsの重複防止(30分おきの再取り込みで同じ払戻行が増殖しないようにする)
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_payouts_unique ON payouts(race_id, bet_type, combination)"
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"[WARN] payouts一意インデックス作成に失敗(既存データに重複がある可能性): {e}")


def save_raw(conn: sqlite3.Connection, race_date: date, payload: Optional[dict]):
    if payload is None:
        return
    conn.execute(
        """INSERT INTO raw_json (race_date, kind, fetched_at, payload)
           VALUES (?, 'unified', ?, ?)
           ON CONFLICT(race_date, kind) DO UPDATE SET
             fetched_at=excluded.fetched_at, payload=excluded.payload""",
        (race_date.isoformat(), datetime.now().isoformat(), json.dumps(payload, ensure_ascii=False)),
    )


def get_entry_id(conn: sqlite3.Connection, race_id: int, boat_number: int) -> Optional[int]:
    row = conn.execute(
        "SELECT entry_id FROM entries WHERE race_id=? AND boat_number=?",
        (race_id, boat_number),
    ).fetchone()
    return row[0] if row else None


def parse_unified(conn: sqlite3.Connection, race_date: date, payload: Optional[dict]):
    """統合JSON(programs.stadiums.{場}.races.{R} 直下に出走表+preview+result)を全テーブルへ格納"""
    if not payload:
        return
    stadiums = (payload.get("programs") or {}).get("stadiums") or {}

    for stadium_str, stadium_obj in stadiums.items():
        stadium_number = int(stadium_str)
        races = (stadium_obj or {}).get("races") or {}

        for race_str, race in races.items():
            race_number = int(race_str)
            preview = race.get("preview") or {}
            result = race.get("result") or {}

            weather_source = preview.get("weather_number_source") or result.get("weather_number_source")
            wind_speed = preview.get("wind_speed") or result.get("wind_speed")
            wind_dir = preview.get("wind_direction_number") or result.get("wind_direction_number")
            wave_height = preview.get("wave_height") or result.get("wave_height")
            air_temp = preview.get("air_temperature") or result.get("air_temperature")
            water_temp = preview.get("water_temperature") or result.get("water_temperature")

            conn.execute(
                """INSERT INTO races (race_date, stadium_number, race_number, race_title,
                                       race_grade, close_at, distance_m,
                                       weather, wind_direction, wind_speed_m, wave_height_cm,
                                       temperature_c, water_temperature_c)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(race_date, stadium_number, race_number) DO UPDATE SET
                     race_title=excluded.race_title, race_grade=excluded.race_grade,
                     close_at=excluded.close_at, distance_m=excluded.distance_m,
                     weather=excluded.weather, wind_direction=excluded.wind_direction,
                     wind_speed_m=excluded.wind_speed_m, wave_height_cm=excluded.wave_height_cm,
                     temperature_c=excluded.temperature_c, water_temperature_c=excluded.water_temperature_c""",
                (
                    race_date.isoformat(), stadium_number, race_number,
                    race.get("title"),
                    race.get("grade_number_source") or race.get("grade_number"),
                    race.get("closed_at"),
                    race.get("distance") or 1800,
                    weather_source, wind_dir, wind_speed, wave_height, air_temp, water_temp,
                ),
            )
            race_id = conn.execute(
                "SELECT race_id FROM races WHERE race_date=? AND stadium_number=? AND race_number=?",
                (race_date.isoformat(), stadium_number, race_number),
            ).fetchone()[0]

            racers = race.get("racers") or {}
            for entry_str, racer in racers.items():
                boat_number = racer.get("entry_number") or int(entry_str)
                conn.execute(
                    """INSERT INTO entries (race_id, boat_number, racer_registration_number,
                            racer_name, racer_branch, racer_class, racer_age, racer_weight_kg,
                            national_win_rate, national_2連率, local_win_rate, local_2連率,
                            motor_number, motor_2連率, boat_hull_number, boat_hull_2連率,
                            average_start_timing, flying_count, late_count)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(race_id, boat_number) DO UPDATE SET
                         racer_class=excluded.racer_class,
                         national_win_rate=excluded.national_win_rate,
                         national_2連率=excluded.national_2連率,
                         local_win_rate=excluded.local_win_rate,
                         local_2連率=excluded.local_2連率,
                         motor_2連率=excluded.motor_2連率,
                         boat_hull_2連率=excluded.boat_hull_2連率,
                         flying_count=excluded.flying_count,
                         late_count=excluded.late_count""",
                    (
                        race_id, boat_number,
                        racer.get("number"),
                        racer.get("name"),
                        racer.get("branch_number_source") or racer.get("branch_number"),
                        racer.get("rank_number_source") or RANK_NAMES.get(racer.get("rank_number"), racer.get("rank_number")),
                        racer.get("age"),
                        racer.get("weight"),
                        racer.get("national_win_rate"),
                        racer.get("national_top_2_percent"),
                        racer.get("local_win_rate"),
                        racer.get("local_top_2_percent"),
                        racer.get("motor_number"),
                        racer.get("motor_top_2_percent"),
                        racer.get("boat_number"),
                        racer.get("boat_top_2_percent"),
                        racer.get("average_start_timing"),
                        racer.get("flying_count"),
                        racer.get("late_count"),
                    ),
                )

            preview_racers = preview.get("racers") or {}
            for entry_str, p in preview_racers.items():
                boat_number = p.get("entry_number") or int(entry_str)
                entry_id = get_entry_id(conn, race_id, boat_number)
                if entry_id is None:
                    continue
                conn.execute(
                    """INSERT INTO previews (entry_id, exhibition_time, tilt_angle,
                            weight_adjustment_kg, start_course, start_timing_preview)
                       VALUES (?,?,?,?,?,?)
                       ON CONFLICT(entry_id) DO UPDATE SET
                         exhibition_time=excluded.exhibition_time,
                         tilt_angle=excluded.tilt_angle,
                         weight_adjustment_kg=excluded.weight_adjustment_kg,
                         start_course=excluded.start_course,
                         start_timing_preview=excluded.start_timing_preview""",
                    (
                        entry_id,
                        p.get("exhibition_time"),
                        p.get("tilt_adjustment"),
                        p.get("weight_adjustment"),
                        p.get("course_number"),
                        p.get("start_timing"),
                    ),
                )

            result_racers = result.get("racers") or {}
            technique_name = TECHNIQUE_NAMES.get(result.get("technique_number"))
            for entry_str, r in result_racers.items():
                boat_number = r.get("entry_number") or int(entry_str)
                entry_id = get_entry_id(conn, race_id, boat_number)
                if entry_id is None:
                    continue
                place = r.get("place_number")
                conn.execute(
                    """INSERT INTO results (entry_id, arrival_order, actual_course,
                            actual_start_timing, winning_technique)
                       VALUES (?,?,?,?,?)
                       ON CONFLICT(entry_id) DO UPDATE SET
                         arrival_order=excluded.arrival_order,
                         actual_course=excluded.actual_course,
                         actual_start_timing=excluded.actual_start_timing,
                         winning_technique=excluded.winning_technique""",
                    (
                        entry_id, place,
                        r.get("course_number"),
                        r.get("start_timing"),
                        technique_name if place == 1 else None,
                    ),
                )

            payouts = result.get("payouts") or {}
            for bet_type, items in payouts.items():
                for item in items or []:
                    conn.execute(
                        """INSERT INTO payouts (race_id, bet_type, combination, payout_yen)
                           VALUES (?,?,?,?)
                           ON CONFLICT(race_id, bet_type, combination) DO UPDATE SET
                             payout_yen=excluded.payout_yen""",
                        (race_id, bet_type, item.get("combination"), item.get("amount")),
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
        payload = fetch_unified(d)
        save_raw(conn, d, payload)
        conn.commit()
        parse_unified(conn, d, payload)

    conn.close()
    print("完了しました。")


if __name__ == "__main__":
    main()
