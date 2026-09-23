"""
build_features.py が出力した特徴量CSVから、LightGBMのランク学習(lambdarank)で
「1レース内の艇の着順」を直接学習するモデルを1つ作る。

    model.txt : 各艇の「強さ」スコアを出すモデル(スコアが高いほど上位に来やすい)

旧方式(1着/2着/3着をそれぞれ別の二値分類モデルで学習し、予測時に条件付き確率を
掛け合わせる方式)からの変更点:
    旧方式では3つのモデルがそれぞれ独立に「1着かどうか」「(1着を除いた中で)2着かどうか」
    を学習していたため、モデル間で一貫性がない(3着モデルが学習した艇の強さの序列と、
    1着モデルが学習した序列が微妙にズレる)可能性があった。
    ランク学習では「そのレースで実際にどの順で並んだか」を1つのモデルに直接学習させるため、
    1つのスコアが「1着になりやすさ」「2着になりやすさ」...のすべてに一貫して使える
    (Plackett-Luceモデルの考え方:スコアをsoftmaxで正規化すると1着確率になり、
    1着候補を除いて残りをsoftmaxし直すと2着確率になる、を繰り返す)。
    このため export_today.py 側も model_2nd / model_3rd を個別に読み込む必要がなくなる。

関連度ラベル(relevance):
    1着=6, 2着=5, 3着=4, 4着=3, 5着=2, 6着=1 (着順が良いほど高スコア)

検証方法(交差検証):
    train_model.py(旧版)と同様、GroupKFoldでrace_id単位に5分割して評価する。
    ランク学習の評価指標としては ndcg を使うが、直感的にわかりやすいよう
    「予測1位に選んだ艇が実際に1着だった割合」(的中率)も併せて出す。

学習結果の記録:
    学習のたびに、交差検証の的中率・ndcg@1・データ件数などを --history で指定した
    CSVに1行追記する(デフォルト: --out と同じディレクトリの training_history.csv)。
    「データを増やしていくと的中率がどう変わるか」を後から時系列で追えるようにするため。
    末尾を確認したい場合は `tail data-collection/training_history.csv` などで見られる。

使い方:
    python train_model.py --features features.csv --out model.txt
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from build_features import FEATURE_COLS

N_FOLDS = 5
MAX_RELEVANCE = 6  # 6艇立てを想定(relevance = MAX_RELEVANCE - arrival_order + 1)
HISTORY_COLUMNS = [
    "run_at", "n_rows", "n_races", "n_folds",
    "mean_ndcg1", "std_ndcg1", "mean_hit_rate", "std_hit_rate", "final_num_round",
]


def _append_history(history_path: str, row: dict) -> None:
    """学習結果を training_history.csv に1行追記する(ファイルが無ければヘッダーから作成)。"""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"学習履歴を記録しました: {history_path}")


def _add_relevance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["relevance"] = (MAX_RELEVANCE - df["arrival_order"] + 1).clip(lower=0).astype(int)
    return df


def _group_sizes(df: pd.DataFrame) -> np.ndarray:
    """lgb.Dataset の group 引数用に、race_idごとの行数をレース出現順に並べた配列を作る。
    呼び出し側で df は事前に race_id でソート済みであること(同じレースの行が連続している必要がある)。
    """
    return df.groupby("race_id", sort=False).size().to_numpy()


def _hit_rate(df: pd.DataFrame, preds: np.ndarray) -> float:
    """レースごとに予測スコア最大の艇が、実際に1着だったかどうかの割合。"""
    tmp = df.copy()
    tmp["pred"] = preds
    top_pick = tmp.loc[tmp.groupby("race_id")["pred"].idxmax()]
    return float((top_pick["arrival_order"] == 1).mean())


def train_model(df: pd.DataFrame, out_path: str, lgb, GroupKFold) -> dict | None:
    """モデルを学習・保存し、training_history.csv に追記するための結果サマリを返す
    (交差検証できず学習をスキップした場合は None を返す)。"""
    df = df.dropna(subset=FEATURE_COLS + ["arrival_order", "race_id"]).reset_index(drop=True)
    df = _add_relevance(df)

    n_races = df["race_id"].nunique()
    n_folds = min(N_FOLDS, n_races)
    if n_folds < 2:
        print("レース数が少なすぎて交差検証できないためスキップしました。")
        return None

    # データが9,966行→15,000行超まで増えてきたため、木の複雑さを少し上げても
    # 過学習しにくくなっている想定でnum_leavesを15→24に引き上げ、
    # 葉あたりの最低データ数(min_data_in_leaf)で過学習に軽く歯止めをかける。
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 24,
        "min_data_in_leaf": 20,
    }

    gkf = GroupKFold(n_splits=n_folds)
    fold_ndcg = []
    fold_hit_rates = []
    fold_best_iters = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["race_id"]), start=1):
        # 同じレースの行が連続している必要があるので、race_idでソートしてから使う
        train_df = df.iloc[train_idx].sort_values("race_id").reset_index(drop=True)
        test_df = df.iloc[test_idx].sort_values("race_id").reset_index(drop=True)

        train_set = lgb.Dataset(
            train_df[FEATURE_COLS], label=train_df["relevance"],
            group=_group_sizes(train_df),
        )
        valid_set = lgb.Dataset(
            test_df[FEATURE_COLS], label=test_df["relevance"],
            group=_group_sizes(test_df), reference=train_set,
        )

        model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        preds = model.predict(test_df[FEATURE_COLS])
        hit_rate = _hit_rate(test_df, preds)
        ndcg1 = model.best_score["valid_0"].get("ndcg@1")

        fold_ndcg.append(ndcg1)
        fold_hit_rates.append(hit_rate)
        fold_best_iters.append(model.best_iteration or 500)
        print(f"fold {fold_i}/{n_folds}: ndcg@1={ndcg1:.4f} 的中率={hit_rate:.1%} (反復{model.best_iteration})")

    mean_ndcg = float(np.mean(fold_ndcg))
    std_ndcg = float(np.std(fold_ndcg))
    mean_hit = float(np.mean(fold_hit_rates))
    std_hit = float(np.std(fold_hit_rates))
    print(f"\n交差検証({n_folds}分割)の平均: "
          f"ndcg@1={mean_ndcg:.4f}(±{std_ndcg:.4f}) 的中率={mean_hit:.1%}(±{std_hit*100:.1f}pt)")

    # 最終モデルは、5分割の平均的な学習回数を使って全データで学習し直す
    final_num_round = max(10, round(float(np.mean(fold_best_iters))))
    full_df = df.sort_values("race_id").reset_index(drop=True)
    full_set = lgb.Dataset(
        full_df[FEATURE_COLS], label=full_df["relevance"],
        group=_group_sizes(full_df),
    )
    final_model = lgb.train(
        params,
        full_set,
        num_boost_round=final_num_round,
        callbacks=[lgb.log_evaluation(0)],
    )
    print(f"全データ({len(df)}行)で最終モデルを学習しました(反復{final_num_round}回)")

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "gain": final_model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance["gain_pct"] = (importance["gain"] / importance["gain"].sum() * 100).round(1)
    print("特徴量重要度(上位5件):")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']:<22} {row['gain_pct']:>5.1f}%")

    final_model.save_model(out_path)
    print(f"モデルを保存しました: {out_path}")

    return {
        "run_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "n_rows": len(df),
        "n_races": n_races,
        "n_folds": n_folds,
        "mean_ndcg1": round(mean_ndcg, 4),
        "std_ndcg1": round(std_ndcg, 4),
        "mean_hit_rate": round(mean_hit, 4),
        "std_hit_rate": round(std_hit, 4),
        "final_num_round": final_num_round,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--out", default="model.txt")
    ap.add_argument("--history", default=None,
                     help="学習結果を記録するCSVのパス(省略時は--outと同じディレクトリのtraining_history.csv)")
    ap.add_argument("--min-races", type=int, default=300,
                     help="このレース数未満なら学習を中止する(過学習・不安定なモデルを防ぐため)")
    args = ap.parse_args()

    try:
        import lightgbm as lgb
        from sklearn.model_selection import GroupKFold
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

    history_path = args.history or str(Path(args.out).with_name("training_history.csv"))
    summary = train_model(df, args.out, lgb, GroupKFold)
    if summary is not None:
        _append_history(history_path, summary)


if __name__ == "__main__":
    main()
