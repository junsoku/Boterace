"""
モデルA(基準。通常は現行のFEATURE_COLS全部)とモデルB(--drop-featureで指定した
特徴量を抜いたもの)を、train_model.py と全く同じ手順(GroupKFold 5分割、lambdarank)で
学習し、各foldの検証用レース(学習に使っていないレース)に対して、ablation_test.pyより
ずっと詳しい指標一式で比較する。

ablation_test.py は単勝のNDCGしか見ていなかったため、「単勝は良くなったが3連単は
悪化した」のような副作用を見逃す可能性があった。本スクリプトは以下を両方の設定について
算出することで、的中率が総合的に改善したかを確認できるようにする:

    - ◎本命1着的中率(win hit rate)
    - 3連単本命的中率(推奨1位の組み合わせの的中率)
    - 推奨候補いずれかの的中率(通常上位4点のいずれかが当たった率)
    - 3連複相当的中率(予測トップ3艇の顔ぶれが実際の上位3着と一致した率)
    - 購入額・払戻額・回収率(実際のpayoutsデータを使用。オッズを前提にしない)
    - レース数・95%信頼区間(Wilson score interval。小サンプルでの誤判断を防ぐ)

model.txtは一切保存・上書きしない(本番モデルには影響しない使い捨ての実験)。
精度が改善したかどうかの最終判断は、このスクリプトの出力を見て人が行うこと
(自動で本番モデルを置き換える仕組みではない)。

使い方:
    # 基準(全特徴量)だけを評価して現状把握する
    python compare_models.py --features features.csv --db boatrace.db

    # 基準 vs start_courceを抜いた場合、を比較する
    python compare_models.py --features features.csv --db boatrace.db --drop-feature start_course
"""
import argparse
import math
import sqlite3

import numpy as np
import pandas as pd

from build_features import FEATURE_COLS as ALL_FEATURE_COLS
from export_today import normalize_to_pct, estimate_bets_ml
from train_model import N_FOLDS, _add_relevance, _group_sizes

STAKE_PER_BET = 100
TOP_N_CANDIDATES = 4


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple:
    """二項比率(的中率)の95%信頼区間(Wilson score interval)。
    小サンプルでの単純な正規近似より安定しており、nが小さい時にも使える。"""
    if n == 0:
        return (0.0, 0.0)
    phat = hits / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def evaluate_race_predictions(conn: sqlite3.Connection, test_df: pd.DataFrame, preds: np.ndarray) -> list:
    """1fold分の検証レースについて、レースごとの的中結果・払戻を計算する。"""
    df = test_df.copy()
    df["pred_score"] = preds
    race_results = []

    for race_id, group in df.groupby("race_id"):
        group = group.sort_values("boat_number")
        boat_numbers = group["boat_number"].astype(int).tolist()
        scores = group["pred_score"].tolist()
        if len(boat_numbers) != 6:
            continue  # 6艇揃っていないレースは対象外

        pcts = normalize_to_pct(scores, use_softmax=True)
        boats_ranked = sorted(zip(boat_numbers, pcts), key=lambda x: -x[1])
        top_lane = boats_ranked[0][0]
        predicted_top3_set = frozenset(x[0] for x in boats_ranked[:3])

        scores_by_lane = {bn: s for bn, s in zip(boat_numbers, scores)}
        bets = estimate_bets_ml([], scores_by_lane, top_n=TOP_N_CANDIDATES)
        top_bet_combo = bets[0]["combo"] if bets else None
        candidate_combos = {b["combo"] for b in bets}

        result_rows = conn.execute(
            """SELECT e.boat_number, r.arrival_order
               FROM entries e JOIN results r ON r.entry_id = e.entry_id
               WHERE e.race_id = ? ORDER BY e.boat_number""",
            (int(race_id),),
        ).fetchall()
        if len(result_rows) != 6 or any(row[1] is None for row in result_rows):
            continue  # 結果未確定のレースは対象外

        ordered = sorted(result_rows, key=lambda row: row[1])
        actual_winner = ordered[0][0]
        actual_top3_set = frozenset(row[0] for row in ordered[:3])
        actual_combo = "-".join(str(row[0]) for row in ordered[:3])

        payout_row = conn.execute(
            """SELECT payout_yen FROM payouts
               WHERE race_id=? AND bet_type='trifecta' AND combination=? AND payout_yen IS NOT NULL""",
            (int(race_id), actual_combo),
        ).fetchone()
        payout_yen = payout_row[0] if payout_row else None
        bet_hit = top_bet_combo == actual_combo

        race_results.append({
            "win_hit": top_lane == actual_winner,
            "bet_hit": bet_hit,
            "bet_hit_any": actual_combo in candidate_combos,
            "top3_set_hit": predicted_top3_set == actual_top3_set,
            "payout_yen": payout_yen,
            "bet_hit_payout": payout_yen if bet_hit else 0,
        })
    return race_results


def run_cv_and_evaluate(df: pd.DataFrame, feature_cols: list, conn: sqlite3.Connection, label: str) -> list:
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    df = df.dropna(subset=feature_cols + ["arrival_order", "race_id"]).reset_index(drop=True)
    df = _add_relevance(df)

    n_races = df["race_id"].nunique()
    n_folds = min(N_FOLDS, n_races)
    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3],
        "verbosity": -1, "learning_rate": 0.05, "num_leaves": 15,
    }

    gkf = GroupKFold(n_splits=n_folds)
    all_race_results = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["race_id"])):
        train_df = df.iloc[train_idx].sort_values("race_id").reset_index(drop=True)
        test_df = df.iloc[test_idx].sort_values("race_id").reset_index(drop=True)

        train_set = lgb.Dataset(train_df[feature_cols], label=train_df["relevance"], group=_group_sizes(train_df))
        valid_set = lgb.Dataset(test_df[feature_cols], label=test_df["relevance"], group=_group_sizes(test_df),
                                 reference=train_set)
        model = lgb.train(
            params, train_set, num_boost_round=500, valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        preds = model.predict(test_df[feature_cols])
        race_results = evaluate_race_predictions(conn, test_df, preds)
        all_race_results.extend(race_results)
        print(f"  [{label}] fold {fold_i+1}/{n_folds} 完了({len(race_results)}レース評価)")

    return all_race_results


def summarize(race_results: list, label: str) -> dict:
    n = len(race_results)
    win_hits = sum(1 for r in race_results if r["win_hit"])
    bet_hits = sum(1 for r in race_results if r["bet_hit"])
    bet_hits_any = sum(1 for r in race_results if r["bet_hit_any"])
    top3_hits = sum(1 for r in race_results if r["top3_set_hit"])
    stake = sum(STAKE_PER_BET for r in race_results if r["payout_yen"] is not None)
    payout = sum(r["bet_hit_payout"] for r in race_results if r["payout_yen"] is not None)
    roi = payout / stake * 100 if stake else None

    summary = {"label": label, "n": n}
    for key, hits in [("win", win_hits), ("bet", bet_hits), ("bet_any", bet_hits_any), ("top3", top3_hits)]:
        rate = hits / n if n else 0.0
        lo, hi = wilson_interval(hits, n)
        summary[key] = {"hits": hits, "rate": rate, "ci": (lo, hi)}
    summary["roi"] = roi
    summary["stake"] = stake
    summary["payout"] = payout
    return summary


def print_summary(summary: dict):
    n = summary["n"]
    print(f"\n[{summary['label']}] 検証レース数: {n}件")
    for key, jp in [("win", "◎本命1着的中率"), ("bet", "3連単本命的中率"),
                    ("bet_any", f"推奨候補(上位{TOP_N_CANDIDATES}点)いずれかの的中率"),
                    ("top3", "3連複相当的中率")]:
        s = summary[key]
        lo, hi = s["ci"]
        print(f"  {jp:<32}: {s['rate']*100:5.1f}%  (95%信頼区間 {lo*100:4.1f}% 〜 {hi*100:4.1f}%、{s['hits']}/{n}件)")
    roi_str = f"{summary['roi']:.1f}%" if summary["roi"] is not None else "N/A"
    print(f"  {'3連単本命 購入額/払戻額/回収率':<32}: {summary['stake']}円 / {summary['payout']}円 / {roi_str}")
    print("  ※ 回収率は払戻データがある範囲のみの集計。サンプルが少ないと大きくブレるので、"
          "点推定として参考程度に見ること(上記に信頼区間は出していない)。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--db", default="boatrace.db")
    ap.add_argument("--drop-feature", default=None,
                     help="指定すると、全特徴量(基準)とこの特徴量を抜いたものを比較する。"
                          "省略すると基準だけを評価する。")
    args = ap.parse_args()

    try:
        import lightgbm as lgb  # noqa: F401
        from sklearn.model_selection import GroupKFold  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"lightgbm / scikit-learn が見つかりません: {e}")

    if args.drop_feature is not None and args.drop_feature not in ALL_FEATURE_COLS:
        raise SystemExit(f"'{args.drop_feature}' はFEATURE_COLSに存在しません。"
                          f"存在する特徴量: {ALL_FEATURE_COLS}")

    df = pd.read_csv(args.features)
    conn = sqlite3.connect(args.db)

    print("=" * 70)
    print(f"[A: 基準(全{len(ALL_FEATURE_COLS)}特徴量)] 学習・評価中...")
    results_a = run_cv_and_evaluate(df, ALL_FEATURE_COLS, conn, "A: 基準")
    summary_a = summarize(results_a, f"A: 基準(全{len(ALL_FEATURE_COLS)}特徴量)")
    print_summary(summary_a)

    if args.drop_feature:
        print("\n" + "=" * 70)
        cols_b = [c for c in ALL_FEATURE_COLS if c != args.drop_feature]
        print(f"[B: {args.drop_feature}抜き] 学習・評価中...")
        results_b = run_cv_and_evaluate(df, cols_b, conn, f"B: {args.drop_feature}抜き")
        summary_b = summarize(results_b, f"B: {args.drop_feature}抜き")
        print_summary(summary_b)

        print("\n" + "=" * 70)
        print("[A vs B 差分]")
        for key, jp in [("win", "◎本命1着的中率"), ("bet", "3連単本命的中率"),
                        ("bet_any", "推奨候補いずれかの的中率"), ("top3", "3連複相当的中率")]:
            diff = (summary_b[key]["rate"] - summary_a[key]["rate"]) * 100
            print(f"  {jp:<28}: {diff:+5.1f}pt")
        if summary_a["roi"] is not None and summary_b["roi"] is not None:
            print(f"  {'回収率':<28}: {summary_b['roi']-summary_a['roi']:+5.1f}pt")

        # A・Bどちらが優れているかの自動判定はしない(提案書の制約通り)。
        # 目安として、Bの的中率系指標がAの信頼区間の下限を下回っていないかだけ示す。
        print("\n[参考] Bの各指標がAの信頼区間の範囲内に収まっているか(外れていれば明確な差がある可能性):")
        for key, jp in [("win", "◎本命1着的中率"), ("bet", "3連単本命的中率"),
                        ("bet_any", "推奨候補いずれかの的中率"), ("top3", "3連複相当的中率")]:
            a_lo, a_hi = summary_a[key]["ci"]
            b_rate = summary_b[key]["rate"]
            in_range = a_lo <= b_rate <= a_hi
            print(f"  {jp:<28}: {'Aの信頼区間内(明確な差なし)' if in_range else 'Aの信頼区間外(差がある可能性)'}")

    print("\n※ 採用するかどうかはこの出力を見て判断すること。自動でmodel.txtを置き換える仕組みではない。")
    conn.close()


if __name__ == "__main__":
    main()
