"""
蓄積したDB(ingest.pyで作ったboatrace.db)から、機械学習用の特徴量テーブルを作る。

1行 = 1艇(entry)。is_winner/is_second/is_thirdの3列を持たせているが、学習自体は
train_model.py 側でランク学習(lambdarank)の1モデルとして行う(着順をそのまま
relevanceとして使うため)。is_second/is_thirdは過去の3段階モデル方式の名残りで、
現在は使っていない(将来また使う可能性があるため残してある)。

特徴量のうちコース別統計(course_win_rate等)・選手の直近の調子(racer_recent_*)は、
「そのレースより前の結果だけ」を使うようにしている(未来の情報が紛れ込むリークを防ぐため)。

使い方:
    python build_features.py --db boatrace.db --out features.csv
"""
import argparse
import sqlite3

import pandas as pd

from technique_stats import (
    compute_course_technique_rates, fetch_racer_history, recent_form_from_history,
)

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
    # コース別の決まり手発生率(technique_stats.py の compute_course_technique_rates を利用)
    "course_nige_rate",
    "course_sashi_rate",
    "course_makuri_rate",
    # ここから直前情報の取りこぼし分
    "weight_adjustment_kg",
    "start_timing_preview",
    # ここから気温・水温(races由来、追加漏れしていた分)
    "temperature_c",
    "water_temperature_c",
    # ここから選手の直近の調子(直近10走の平均着順・勝率。未来の結果が混ざらないよう
    # そのレースの日付より前の結果だけを使っている。詳細はtechnique_stats.py参照)
    "racer_recent_avg_order",
    "racer_recent_win_rate",
    # 展示での進入コース(号艇番号とズレることがある実測値。previews.start_course)
    "start_course",
]


def build_features(conn: sqlite3.Connection) -> tuple[pd.DataFrame, list]:
    query = """
        SELECT
            races.race_id, races.race_date, races.stadium_number, races.race_number,
            races.wind_speed_m, races.wave_height_cm,
            races.temperature_c, races.water_temperature_c,
            e.entry_id, e.boat_number, e.racer_registration_number, e.racer_name,
            e.racer_class,
            e.national_win_rate, e.national_2連率,
            e.local_win_rate, e.local_2連率,
            e.motor_2連率, e.boat_hull_2連率, e.average_start_timing,
            e.flying_count, e.late_count,
            p.exhibition_time, p.tilt_angle,
            p.weight_adjustment_kg, p.start_timing_preview, p.start_course,
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

    # 場ごとのコース別勝率を結合(technique_stats.py の集計を利用)。
    # 必ず「そのレースより前の結果だけ」を使う(before_date・before_race_id)。
    # これを省略して場全体の集計(未来の結果も含む)を使うと、学習データに未来の情報が
    # 紛れ込むリークになり、交差検証の的中率が実際の運用より楽観的に出てしまう。
    # レース単位(同じレースの6艇は同じ統計を使う)でキャッシュし、DB問い合わせを
    # レース数程度に抑える(行数分=艇数分まで増やす必要はないため)。
    course_stats_cache = {}

    def course_stats_for_race(stadium_number, race_date, race_id):
        key = (stadium_number, race_date, race_id)
        if key not in course_stats_cache:
            course_stats_cache[key] = compute_course_technique_rates(
                conn, stadium_number, before_date=race_date, before_race_id=race_id
            )
        return course_stats_cache[key]

    def course_stat_lookup(row, key):
        stats = course_stats_for_race(int(row["stadium_number"]), row["race_date"], int(row["race_id"]))
        if not stats:
            return None
        return stats.get(int(row["boat_number"]), {}).get(key)

    df["course_win_rate"] = df.apply(lambda row: course_stat_lookup(row, "win_rate"), axis=1)
    df["course_nige_rate"] = df.apply(lambda row: course_stat_lookup(row, "逃げ"), axis=1)
    df["course_sashi_rate"] = df.apply(lambda row: course_stat_lookup(row, "差し"), axis=1)
    df["course_makuri_rate"] = df.apply(lambda row: course_stat_lookup(row, "まくり"), axis=1)

    # 選手の直近の調子: 選手ごとに全履歴を1回だけ取得してキャッシュし(DB問い合わせを選手数分に抑える)、
    # 各行では「そのレースの日付より前」の直近10走だけを切り出す(未来の結果が混ざるリークを防ぐため)。
    racer_history_cache = {}

    def recent_form_lookup(row, key):
        reg_no = row["racer_registration_number"]
        if pd.isna(reg_no):
            return None
        reg_no = int(reg_no)
        if reg_no not in racer_history_cache:
            racer_history_cache[reg_no] = fetch_racer_history(conn, reg_no)
        form = recent_form_from_history(
            racer_history_cache[reg_no], row["race_date"], before_race_id=int(row["race_id"])
        )
        return form.get(key) if form else None

    df["racer_recent_avg_order"] = df.apply(lambda row: recent_form_lookup(row, "avg_arrival_order"), axis=1)
    df["racer_recent_win_rate"] = df.apply(lambda row: recent_form_lookup(row, "recent_win_rate"), axis=1)

    df["is_winner"] = (df["arrival_order"] == 1).astype(int)
    df["is_second"] = (df["arrival_order"] == 2).astype(int)  # 2着モデル用(1着だった艇を除いた中で学習)
    df["is_third"] = (df["arrival_order"] == 3).astype(int)   # 3着モデル用(1・2着だった艇を除いた中で学習)

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
