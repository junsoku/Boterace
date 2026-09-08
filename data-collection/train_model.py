"""
build_features.py が出力した特徴量CSVからLightGBMモデルを学習する。

使い方:
    python train_model.py --features features.csv --out model.txt

必要パッケージ: lightgbm, scikit-learn, pandas (requirements.txt に追加済み)
"""
import argparse

import pandas as pd

from build_features import FEATURE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="model.txt")
    ap.add_argument("--min-races", type=int, default=300,
                     help="このレース数未満なら学習を中止する(過学習・不安定なモデルを防ぐため)")
    args = ap.parse_args()

    try:
        import lightgbm as lgb
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import log_loss
    except ImportError as e:
        raise SystemExit(
            "lightgbm / scikit-learn が見つかりません。"
            "`pip install -r requirements.txt` を実行してから再度試してください。"
            f"(詳細: {e})"
        )

    df = pd.read_csv(args.features)
    df = df.dropna(subset=FEATURE_COLS + ["is_winner", "race_id"])

    n_races = df["race_id"].nunique()
    if n_races < args.min_races:
        raise SystemExit(
            f"結果確定レース数が {n_races} 件しかありません(目安 {args.min_races} 件以上)。"
            "データが溜まるまで学習は見送ってください。"
            "(ingest.py を毎日実行して結果データを蓄積し続けてください)"
        )

    # レース単位で train/test を分割(同じレースの艇が学習/検証両方に混ざらないように)
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["race_id"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    train_set = lgb.Dataset(train_df[FEATURE_COLS], label=train_df["is_winner"])
    valid_set = lgb.Dataset(test_df[FEATURE_COLS], label=test_df["is_winner"], reference=train_set)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 15,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    preds = model.predict(test_df[FEATURE_COLS])
    loss = log_loss(test_df["is_winner"], preds)
    print(f"検証データ logloss: {loss:.4f}")

    # レースごとに「予測1位が実際に勝ったか」を見る(単勝的中率に近い指標)
    test_df = test_df.copy()
    test_df["pred"] = preds
    hit_rate = test_df.loc[test_df.groupby("race_id")["pred"].idxmax(), "is_winner"].mean()
    print(f"予測1位の的中率(単勝的中率相当の目安): {hit_rate:.1%}")

    model.save_model(args.out)
    print(f"モデルを保存しました: {args.out}")


if __name__ == "__main__":
    main()
