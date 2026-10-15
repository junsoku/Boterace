"""
boatrace.db のテーブル(・インデックス)ごとの実際のディスク使用量を可視化する。

GitHubの通常git(非LFS)のファイルサイズ上限(100MB)に近づいている問題の原因切り分け用。
raw_jsonの間引き(ingest.pyの_prune_raw_json)だけでは不十分だった場合に、
他にどのテーブルが肥大化しているかを特定するために使う。

SQLiteのdbstat仮想テーブル(ページ単位の実使用量を見られる標準機能)を使うので、
行数ベースの概算ではなく、実際のディスク上のバイト数に近い値が分かる。

使い方:
    python analyze_db_size.py --db boatrace.db
"""
import argparse
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    try:
        rows = conn.execute(
            """
            SELECT s.name, COALESCE(m.type, '(index)') as obj_type, SUM(s.pgsize) as bytes
            FROM dbstat s
            LEFT JOIN sqlite_master m ON m.name = s.name
            GROUP BY s.name
            ORDER BY bytes DESC
            """
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"dbstat仮想テーブルが使えませんでした({e})。")
        print("この環境のSQLiteビルドがdbstatに対応していない可能性があります。")
        print("代わりに行数だけ出します(バイト数の目安にはなりません)。\n")
        rows = None

    import os
    total_bytes = os.path.getsize(args.db)
    print(f"DBファイル全体: {total_bytes / 1024 / 1024:.1f}MB\n")

    if rows:
        print(f"{'テーブル/インデックス':<30} {'種別':<8} {'サイズ(MB)':>10} {'割合':>7}")
        print("-" * 60)
        for name, obj_type, bytes_ in rows:
            if bytes_ is None or name is None:
                continue
            mb = bytes_ / 1024 / 1024
            pct = bytes_ / total_bytes * 100 if total_bytes else 0
            print(f"{name:<30} {obj_type:<8} {mb:>10.2f} {pct:>6.1f}%")

    print("\n■ 参考: 主要テーブルの行数")
    for table in ["raw_json", "races", "entries", "previews", "results", "payouts", "prediction_log"]:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<16}: {n:>8,}行")
        except sqlite3.OperationalError:
            print(f"  {table:<16}: (テーブルが存在しません)")

    conn.close()

    print("\n※ サイズの大きいテーブルから優先的に削減策を検討する。"
          "raw_json以外が大きい場合は、古いpayouts/resultsの集約・アーカイブや、"
          "不要なインデックスの見直しなどを検討する。")


if __name__ == "__main__":
    main()
