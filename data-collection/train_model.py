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

検証方法(交差検証):
    1回だけのtrain/testスプリットだと、たまたま検証用に選ばれたレースの難易度次第で
    logloss・的中率が大きくブレる(実際、日によって的中率が10pt以上動くことがあった)。
    そこで GroupKFold で5分割し、5回の検証結果を平均することで、より安定した
    (運に左右されにくい)評価値を出す。race_id単位でグループ化しているので、
    同じレースの艇が学習用・検証用の両方に混ざることはない。

    最終的にモデルファイルとして保存するのは、5分割の平均的な学習回数(best_iteration)を
    使って全データで学習し直したもの(データを1点も捨てずに使うため)。

使い方:
    python train_model.py --features features.csv --out model.txt
    (--out で指定したファイル名から自動的に model_2nd.txt / model_3rd.txt も同じ場所に出力する)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import FEATURE_COLS

N_FOLDS = 5


def _stage_path(base_out: str, suffix: str) -> str:
    """model.txt -> model_2nd.txt のように、ベースのファイル名から段階別のパスを作る"""
    p = Path(base_out)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def _train_one_stage(df: pd.DataFrame, label_col: str, out_path: str, stage_name: str,
                      lgb, GroupKFold, log_loss) -> None:
    """1つの段階(1着/2着/3着)のモデルを、交差検証で評価した上で全データで学習して保存する"""
    df = df.dropna(subset=FEATURE_COLS + [label_col, "race_id"]).reset_index(drop=True)
    if df.empty or df[label_col].nunique() < 2:
        print(f"[{stage_name}] 学習データが不足しているためスキップしました。")
        return

    n_races = df["race_id"].nunique()
    n_folds = min(N_FOLDS, n_races)  # レース数が極端に少ない場合の安全策
    if n_folds < 2:
        print(f"[{stage_name}] レース数が少なすぎて交差検証できないためスキップしました。")
        return

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 15,
    }

    gkf = GroupKFold(n_splits=n_folds)
    fold_losses = []
    fold_hit_rates = []
    fold_best_iters = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["race_id"]), start=1):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        train_set = lgb.Dataset(train_df[FEATURE_COLS], label=train_df[label_col])
        valid_set = lgb.Dataset(test_df[FEATURE_COLS], label=test_df[label_col], reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        preds = model.predict(test_df[FEATURE_COLS])
        loss = log_loss(test_df[label_col], preds)

        test_df = test_df.copy()
        test_df["pred"] = preds
        hit_rate = test_df.loc[test_df.groupby("race_id")["pred"].idxmax(), label_col].mean()

        fold_losses.append(loss)
        fold_hit_rates.append(hit_rate)
        fold_best_iters.append(model.best_iteration or 500)
        print(f"[{stage_name}] fold {fold_i}/{n_folds}: logloss={loss:.4f} 的中率={hit_rate:.1%} "
              f"(反復{model.best_iteration})")

    mean_loss = float(np.mean(fold_losses))
    std_loss = float(np.std(fold_losses))
    mean_hit = float(np.mean(fold_hit_rates))
    std_hit = float(np.std(fold_hit_rates))
    print(f"\n[{stage_name}] 交差検証({n_folds}分割)の平均: "
          f"logloss={mean_loss:.4f}(±{std_loss:.4f}) 的中率={mean_hit:.1%}(±{std_hit*100:.1f}pt)")

    # 最終モデルは、5分割の平均的な学習回数を使って全データで学習し直す
    # (検証専用に取り分けていた分のデータも無駄にしないため)
    final_num_round = max(10, round(float(np.mean(fold_best_iters))))
    full_set = lgb.Dataset(df[FEATURE_COLS], label=df[label_col])
    final_model = lgb.train(
        params,
        full_set,
        num_boost_round=final_num_round,
        callbacks=[lgb.log_evaluation(0)],
    )
    print(f"[{stage_name}] 全データ({len(df)}行)で最終モデルを学習しました(反復{final_num_round}回)")

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "gain": final_model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance["gain_pct"] = (importance["gain"] / importance["gain"].sum() * 100).round(1)
    print(f"[{stage_name}] 特徴量重要度(上位5件):")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']:<22} {row['gain_pct']:>5.1f}%")

    final_model.save_model(out_path)
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
        from sklearn.model_selection import GroupKFold
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
    _train_one_stage(df, "is_winner", args.out, "1着", lgb, GroupKFold, log_loss)

    # ---- 2段階目: 2着モデル(1着だった艇を除いた5艇が対象) ----
    df_2nd = df[df["is_winner"] == 0]
    _train_one_stage(df_2nd, "is_second", _stage_path(args.out, "2nd"), "2着",
                      lgb, GroupKFold, log_loss)

    # ---- 3段階目: 3着モデル(1・2着だった艇を除いた4艇が対象) ----
    df_3rd = df[(df["is_winner"] == 0) & (df["is_second"] == 0)]
    _train_one_stage(df_3rd, "is_third", _stage_path(args.out, "3rd"), "3着",
                      lgb, GroupKFold, log_loss)


if __name__ == "__main__":
    main()
