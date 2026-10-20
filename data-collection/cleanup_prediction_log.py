"""
prediction_log に溜まった過去の重複スナップショット(export_today.pyの修正前は、締切前の
30分おきの実行のたびにINSERTしていたため、1レースあたり数十行溜まっていた)を、
1レース1行(export_history.pyのchoose_snapshot()と同じ基準=締切時刻以前の最後の
スナップショット)にまとめ、残りを削除する。

今後export_today.py側は1レース1行(UPDATE方式)に直っているので、これは既存の
過去データを一度だけ掃除するための使い捨てスクリプト。

安全のため、既定では --dry-run 相当の確認表示のみ行い、実際に削除するには
--execute を明示的に指定する必要がある。

使い方:
    python cleanup_prediction_log.py --db boatrace.db                # 件数を確認するだけ(削除しない)
    python cleanup_prediction_log.py --db boatrace.db --execute      # 実際に削除する
"""
import argparse
import sqlite3

from export_history import choose_snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--execute", action="store_true",
                     help="実際に削除を実行する(指定しなければ件数確認のみで削除しない)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    close_at_by_race = dict(conn.execute("SELECT race_id, close_at FROM races").fetchall())

    rows = conn.execute(
        "SELECT log_id, race_id, computed_at FROM prediction_log ORDER BY race_id, computed_at"
    ).fetchall()

    by_race = {}
    for log_id, race_id, computed_at in rows:
        by_race.setdefault(race_id, []).append((computed_at, log_id))

    total_before = len(rows)
    to_delete = []
    multi_snapshot_races = 0

    for race_id, snapshots in by_race.items():
        if len(snapshots) <= 1:
            continue
        multi_snapshot_races += 1
        close_at = close_at_by_race.get(race_id)
        kept_computed_at, kept_log_id = choose_snapshot(snapshots, close_at)
        for computed_at, log_id in snapshots:
            if log_id != kept_log_id:
                to_delete.append(log_id)

    print(f"対象レース数: {len(by_race)}件(うち複数スナップショットあり: {multi_snapshot_races}件)")
    print(f"prediction_log 削除前の総行数: {total_before}行")
    print(f"削除予定: {len(to_delete)}行 → 削除後の想定行数: {total_before - len(to_delete)}行")

    if not args.execute:
        print("\n--executeを指定していないため、確認のみで削除は行っていません。"
              "この件数で問題なければ --execute を付けて再実行してください。")
        conn.close()
        return

    if not to_delete:
        print("\n削除対象がありませんでした。")
        conn.close()
        return

    conn.executemany("DELETE FROM prediction_log WHERE log_id=?", [(lid,) for lid in to_delete])
    conn.commit()
    print(f"\n{len(to_delete)}行削除しました。VACUUM実行中(ファイルサイズを実際に縮小します)...")
    conn.execute("VACUUM")
    conn.commit()

    import os
    size_mb = os.path.getsize(args.db) / 1024 / 1024
    print(f"完了しました。DBファイルサイズ: {size_mb:.1f}MB")

    conn.close()


if __name__ == "__main__":
    main()
