"""
「boat_number(号艇)が効きすぎているのでは?」を確かめるための一回限りの実験スクリプト。

train_model.py と全く同じ手順(GroupKFold 5分割、lambdarank)で、
  (A) 通常通り全特徴量で学習
  (B) FEATURE_COLSからboat_numberだけを抜いて学習
の2パターンを回し、的中率(ndcg@1)がどれくら落ちるか/落ちないかを比較する。
model.txtは保存しない(本番モデルには一切影響しない使い捨ての実験)。

狙い:
    course_win_rate, course_nige_rate 等、boat_numberと相関の強い特徴量が
    重要度をboat_numberに"横取り"されているだけなら、(B)でも的中率はあまり落ちず、
    代わりにcourse_win_rate等の重要度が上がるはず。
    逆に(B)で的中率が大きく落ちるなら、boat_number固有の情報(他の特徴量では
    代替できない情報)がちゃんと効いていたということ。

使い方:
    python ablation_test.py --features features.csv
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
    args = ap.parse_args()

    try:
        import lightgbm as lgb
        from sklearn.model_selection import GroupKFold
    except ImportError as e:
        raise SystemExit(f"lightgbm / scikit-learn が見つかりません: {e}")

    df = pd.read_csv(args.features)

    print("=" * 60)
    hit_a, std_a = run_cv(df, ALL_FEATURE_COLS, lgb, GroupKFold, "A: 全特徴量(boat_number含む)")
    print("=" * 60)
    cols_without_boat = [c for c in ALL_FEATURE_COLS if c != "boat_number"]
    hit_b, std_b = run_cv(df, cols_without_boat, lgb, GroupKFold, "B: boat_number抜き")
    print("=" * 60)

    diff = (hit_a - hit_b) * 100
    print(f"\n差: 的中率が {diff:+.1f}pt 変化(A→Bで抜いた場合)")
    if abs(diff) <= 1.0:
        print("→ ほぼ差がない。boat_numberの重要度は、course_win_rate等と情報が重複している"
              "(重要度を"'横取り'"している)可能性が高い。")
    else:
        print("→ 差が大きい。boat_number固有の情報(他の特徴量では代替できない情報)が"
              "実際に効いている可能性が高い。")


if __name__ == "__main__":
    main()
