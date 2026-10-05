"""
「ある特徴量が実際にモデルの精度に効いているか」を確かめるための一回限りの実験スクリプト
(元々は「boat_number(号艇)が効きすぎているのでは?」の検証用に作ったが、--drop-featureで
任意の特徴量を指定できるよう一般化してある。例: start_courseを追加した効果を測る)。

train_model.py と全く同じ手順(GroupKFold 5分割、lambdarank)で、
  (A) 通常通り全特徴量で学習
  (B) FEATURE_COLSから指定した特徴量(--drop-feature、既定はboat_number)だけを抜いて学習
の2パターンを回し、的中率(ndcg@1)がどれくら落ちるか/落ちないかを比較する。
model.txtは保存しない(本番モデルには一切影響しない使い捨ての実験)。

狙い:
    その特徴量と相関の強い別の特徴量が、重要度を"横取り"しているだけなら、(B)でも
    的中率はあまり落ちず、代わりに相関する特徴量の重要度が上がるはず。
    逆に(B)で的中率が大きく落ちるなら、その特徴量固有の情報(他の特徴量では
    代替できない情報)がちゃんと効いていたということ。

使い方:
    python ablation_test.py --features features.csv
    python ablation_test.py --features features.csv --drop-feature start_course
"""
import argparse

import numpy as np
import pandas as pd

from build_features import FEATURE_COLS as ALL_FEATURE_COLS
from train_model import N_FOLDS, MAX_RELEVANCE, _add_relevance, _group_sizes, _hit_rate


def run_cv(df: pd.DataFrame, feature_cols: list, lgb, GroupKFold, label: str):
    df = df.dropna(subset=feature_cols + ["arrival_order", "race_id"]).reset_index(drop=True)
    df = _add_relevance(df)

    n_races = df["race_id"].nunique()
    n_folds = min(N_FOLDS, n_races)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 15,
    }

    gkf = GroupKFold(n_splits=n_folds)
    fold_hit_rates = []
    last_model = None

    for train_idx, test_idx in gkf.split(df, groups=df["race_id"]):
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
        fold_hit_rates.append(_hit_rate(test_df, preds))
        last_model = model

    mean_hit = float(np.mean(fold_hit_rates))
    std_hit = float(np.std(fold_hit_rates))
    print(f"[{label}] 的中率={mean_hit:.1%}(±{std_hit*100:.1f}pt) / 使用特徴量数={len(feature_cols)}")

    # 参考: このパターンでの特徴量重要度トップ5(全データで学習し直したモデルではなく、
    # 最後のfoldのモデルによる簡易参考値。本番のtrain_model.pyの重要度とは別集計)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": last_model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance["gain_pct"] = (importance["gain"] / importance["gain"].sum() * 100).round(1)
    print(f"[{label}] 特徴量重要度(参考・最終foldのみ、上位5件):")
    for _, row in importance.head(5).iterrows():
        print(f"    {row['feature']:<22} {row['gain_pct']:>5.1f}%")

    return mean_hit, std_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--drop-feature", default="boat_number",
                     help="FEATURE_COLSから除外して比較する特徴量名(既定: boat_number)")
    args = ap.parse_args()

    if args.drop_feature not in ALL_FEATURE_COLS:
        raise SystemExit(f"'{args.drop_feature}' はFEATURE_COLSに存在しません。"
                          f"存在する特徴量: {ALL_FEATURE_COLS}")

    try:
        import lightgbm as lgb
        from sklearn.model_selection import GroupKFold
    except ImportError as e:
        raise SystemExit(f"lightgbm / scikit-learn が見つかりません: {e}")

    df = pd.read_csv(args.features)

    print("=" * 60)
    hit_a, std_a = run_cv(df, ALL_FEATURE_COLS, lgb, GroupKFold, f"A: 全特徴量({args.drop_feature}含む)")
    print("=" * 60)
    cols_without_target = [c for c in ALL_FEATURE_COLS if c != args.drop_feature]
    hit_b, std_b = run_cv(df, cols_without_target, lgb, GroupKFold, f"B: {args.drop_feature}抜き")
    print("=" * 60)

    diff = (hit_a - hit_b) * 100
    print(f"\n差: 的中率が {diff:+.1f}pt 変化(A→Bで{args.drop_feature}を抜いた場合)")
    if abs(diff) <= 1.0:
        print(f"→ ほぼ差がない。{args.drop_feature}の重要度は、他の特徴量と情報が重複している"
              "(重要度を"'横取り'"されている)か、そもそもあまり効いていない可能性が高い。")
    else:
        print(f"→ 差が大きい。{args.drop_feature}固有の情報(他の特徴量では代替できない情報)が"
              "実際に効いている可能性が高い。")


if __name__ == "__main__":
    main()
