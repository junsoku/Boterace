"""
build_features.py が出力した特徴量CSVから、3段階のLightGBMモデルを学習する。

    model.txt      : 1着になる確率(全6艇が対象)
    model_2nd.txt   : 2着になる確率(1着だった艇を除いた5艇が対象)
    model_3rd.txt    : 3着になる確率(1・2着だった艇を除いた4艇が対象)

3連単・2連単の予測時(export_today.py)は、この3つのモデルを順に使って
「1着がaの時、残りの中でbが2着になる確率」「a,bが決まった時、残りの中でcが3着になる確率」
を計算することで、以前の「1着確率をただ按分するだけ」の方式より実際の着順の癖
(1号艇は2着になりにくい、差し・まくりが得意な艇は2〜3着に絡みやすい、等)を
反映できるようにしている。

使い方:
    python train_model.py --features features.csv --out model.txt
    (--out で指定したファイル名から自動的に model_2nd.txt / model_3rd.txt も同じ場所に出力する)
"""
import argparse
from pathlib import Path

import pandas as pd

from build_features import FEATURE_COLS


def _stage_path(base_out: str, suffix: str) -> str:
    """model.txt -> model_2nd.txt のように、ベースのファイル名から段階別のパスを作る"""
    p = Path(base_out)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def _train_one_stage(df: pd.DataFrame, label_col: str, out_path: str, stage_name: str,
                      lgb, GroupShuffleSplit, log_loss) -> None:
    """1つの段階(1着/2着/3着)のモデルを学習して保存する共通処理"""
    df = df.dropna(subset=FEATURE_COLS + [label_col, "race_id"])
    if df.empty or df[label_col].nunique() < 2:
        print(f"[{stage_name}] 学習データが不足しているためスキップしました。")
        return

    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["race_id"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    train_set = lgb.Dataset(train_df[FEATURE_COLS], label=train_df[label_col])
    valid_set = lgb.Dataset(test_df[FEATURE_COLS], label=test_df[label_col], reference=train_set)

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
    loss = log_loss(test_df[label_col], preds)
    print(f"\n[{stage_name}] 検証データ logloss: {loss:.4f}")

    # レースごとに「予測1位が実際に該当着順だったか」を見る(この段階の的中率の目安)
    test_df = test_df.copy()
    test_df["pred"] = preds
    hit_rate = test_df.loc[test_df.groupby("race_id")["pred"].idxmax(), label_col].mean()
    print(f"[{stage_name}] 予測1位の的中率: {hit_rate:.1%}")

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "gain": model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance["gain_pct"] = (importance["gain"] / importance["gain"].sum() * 100).round(1)
    print(f"[{stage_name}] 特徴量重要度(上位5件):")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']:<22} {row['gain_pct']:>5.1f}%")

    model.save_model(out_path)
    print(f"[{stage_name}] モデルを保存しました: {out_path}")


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

    n_races = df["race_id"].nunique()
    if n_races < args.min_races:
        raise SystemExit(
            f"結果確定レース数が {n_races} 件しかありません(目安 {args.min_races} 件以上)。"
            "データが溜まるまで学習は見送ってください。"
            "(ingest.py を毎日実行して結果データを蓄積し続けてください)"
        )

    # ---- 1段階目: 1着モデル(全6艇が対象) ----
    _train_one_stage(df, "is_winner", args.out, "1着", lgb, GroupShuffleSplit, log_loss)

    # ---- 2段階目: 2着モデル(1着だった艇を除いた5艇が対象) ----
    df_2nd = df[df["is_winner"] == 0]
    _train_one_stage(df_2nd, "is_second", _stage_path(args.out, "2nd"), "2着",
                      lgb, GroupShuffleSplit, log_loss)

    # ---- 3段階目: 3着モデル(1・2着だった艇を除いた4艇が対象) ----
    df_3rd = df[(df["is_winner"] == 0) & (df["is_second"] == 0)]
    _train_one_stage(df_3rd, "is_third", _stage_path(args.out, "3rd"), "3着",
                      lgb, GroupShuffleSplit, log_loss)


if __name__ == "__main__":
    main()
