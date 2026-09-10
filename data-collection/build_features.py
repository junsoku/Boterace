"""
蓄積したDB(ingest.pyで作ったboatrace.db)から、機械学習用の特徴量テーブルを作る。

1行 = 1艇(entry)。目的変数 is_winner は「そのレースで1着だったか(0/1)」。
学習は train_model.py で行う。

使い方:
    python build_features.py --db boatrace.db --out features.csv
"""
import argparse
import sqlite3

import pandas as pd

from technique_stats import compute_course_technique_rates

# racer_class(文字列 "A1"/"A2"/"B1"/"B2") を数値化するためのマップ。
# 数値が大きいほど上位級(LightGBMに渡すための単純な序列エンコーディング)。
RACER_CLASS_RANK = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

FEATURE_COLS = [
    "boat_number",
    "national_win_rate",
    "local_win_rate",
    "motor_2連率",
    "boat_hull_2連率",
    "average_start_timing",
    "course_win_rate",
    # ここから追加分
    "racer_class_rank",
    "national_2連率",
    "local_2連率",
    "flying_count",
    "late_count",
    "exhibition_time",
    "tilt_angle",
    "wind_speed_m",
    "wave_height_cm",
]


def build_features(conn: sqlite3.Connection) -> tuple[pd.DataFrame, list]:
    query = """
        SELECT
            races.race_id, races.race_date, races.stadium_number, races.race_number,
            races.wind_speed_m, races.wave_height_cm,
            e.entry_id, e.boat_number, e.racer_registration_number, e.racer_name,
            e.racer_class,
            e.national_win_rate, e.national_2連率,
            e.local_win_rate, e.local_2連率,
            e.motor_2連率, e.boat_hull_2連率, e.average_start_timing,
            e.flying_count, e.late_count,
            p.exhibition_time, p.tilt_angle,
            r.arrival_order
        FROM entries e
        JOIN races ON races.race_id = e.race_id
        LEFT JOIN previews p ON p.entry_id = e.entry_id
        LEFT JOIN results r ON r.entry_id = e.entry_id
    """
    df = pd.read_sql_query(query, conn)
    df = df[df["arrival_order"].notna()].copy()  # 結果が確定しているレースのみ学習に使う

    if df.empty:
        return df, FEATURE_COLS

    # 場ごとのコース別勝率をキャッシュして結合(technique_stats.py の集計を利用)
    course_stats_cache = {
        int(sn): compute_course_technique_rates(conn, int(sn))
        for sn in df["stadium_number"].unique()
    }

    def course_win_rate(row):
        stats = course_stats_cache.get(int(row["stadium_number"]))
        if not stats:
            return None
        return stats.get(int(row["boat_number"]), {}).get("win_rate")

    df["course_win_rate"] = df.apply(course_win_rate, axis=1)
    df["is_winner"] = (df["arrival_order"] == 1).astype(int)

    # 級別(文字列)を序列の数値に変換。未知の値/欠損は後段の中央値補完に任せる。
    df["racer_class_rank"] = df["racer_class"].map(RACER_CLASS_RANK)

    # 欠損値は列の中央値で補完(学習を止めないための簡易対応。件数が増えたら要見直し)
    for col in FEATURE_COLS:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    return df, FEATURE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--out", default="features.csv")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    df, feature_cols = build_features(conn)
    conn.close()

    n_races = df["race_id"].nunique() if not df.empty else 0
    print(f"結果が確定しているレース数: {n_races}")
    if n_races < 300:
        print("⚠ レース数がまだ少ないです(目安は300レース以上)。学習してもモデルの精度はまだ不安定です。")

    df.to_csv(args.out, index=False)
    print(f"出力しました: {args.out} ({len(df)}行 / {len(feature_cols)}特徴量)")


if __name__ == "__main__":
    main()
