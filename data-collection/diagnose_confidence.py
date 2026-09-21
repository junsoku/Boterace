"""
「自信あり」がなぜ出にくいのかを診断するためのスクリプト。
compute_bet_confidence_calibration() / compute_exacta_confidence_calibration() /
compute_confidence_calibration() が実際にどんなバケット(確率帯)を作り、
それぞれのサンプル数・的中率がどうなっているかをそのまま表示する。

見るポイント:
    - サンプル数が MIN_SAMPLE_FOR_*_STAR 未満のバケットばかりなら → データ不足が原因
      (自信あり/なし以前に、統計的に判定できるだけのサンプルがまだ溜まっていない)
    - サンプル数は足りているのに的中率が閾値未満なら → 閾値(BET_CONFIDENCE_THRESHOLD等)が
      実態に対して高すぎる可能性がある

使い方:
    python diagnose_confidence.py --db boatrace.db --model model.txt
    (--model省略時はヒューリスティックscore_boat()で再計算)
"""
import argparse
import sqlite3

import confidence as conf


def _print_table(title: str, calibration: dict, min_sample: int, threshold: float):
    print(f"\n■ {title}")
    if not calibration:
        print("  (該当レースが1件もありませんでした)")
        return
    total = sum(v["sample_size"] for v in calibration.values())
    print(f"  対象レース合計: {total}件 / バケット数: {len(calibration)}")
    for label in sorted(calibration.keys(), key=lambda l: int(l.split("-")[0])):
        stat = calibration[label]
        n = stat["sample_size"]
        hr = stat["hit_rate"]
        hr_str = f"{hr:.1%}" if hr is not None else "N/A"
        enough = "○" if n >= min_sample else "✕(サンプル不足)"
        over_threshold = "○" if (hr is not None and hr >= threshold) else "✕"
        is_confident = "★自信あり" if (n >= min_sample and hr is not None and hr >= threshold) else ""
        print(f"  {label:>8}: {n:>4}件 / 的中率{hr_str:>7} "
              f"/ サンプル充足{enough} / 閾値({threshold:.0%})超え{over_threshold} {is_confident}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--model", default=None, help="model.txtのパス(省略時はヒューリスティックで計算)")
    ap.add_argument("--lookback-days", type=int, default=90)
    args = ap.parse_args()

    model = None
    if args.model:
        import lightgbm as lgb
        model = lgb.Booster(model_file=args.model)
        print(f"MLモデルを使用: {args.model}")
    else:
        print("ヒューリスティック(score_boat)を使用(--model未指定)")

    conn = sqlite3.connect(args.db)

    win_calib = conf.compute_confidence_calibration(conn, lookback_days=args.lookback_days, model=model)
    _print_table("◎(単勝)の確信度バケット", win_calib, conf.MIN_SAMPLE_FOR_STAR, conf.CONFIDENCE_THRESHOLD)

    bet_calib = conf.compute_bet_confidence_calibration(conn, lookback_days=args.lookback_days, model=model)
    _print_table("推奨3連単(本命)の確信度バケット", bet_calib, conf.MIN_SAMPLE_FOR_BET_STAR, conf.BET_CONFIDENCE_THRESHOLD)

    exacta_calib = conf.compute_exacta_confidence_calibration(conn, lookback_days=args.lookback_days, model=model)
    _print_table("推奨2連単(本命)の確信度バケット", exacta_calib, conf.MIN_SAMPLE_FOR_EXACTA_STAR, conf.EXACTA_CONFIDENCE_THRESHOLD)

    conn.close()


if __name__ == "__main__":
    main()
