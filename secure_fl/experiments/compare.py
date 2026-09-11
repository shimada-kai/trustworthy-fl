import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from secure_fl.evaluation.results import list_results, load_result


# =============================================================
# 比較対象とする評価指標
# =============================================================
#
# artifacts/results/*.json に保存された各実験の "final" から、
# 以下4つの指標を取り出して比較する。
#
METRICS = [
    "accuracy",
    "roc_auc",
    "pr_auc",
    "loss",
]


# =============================================================
# 表示用の指標名
# =============================================================
#
# JSON内部ではPythonで扱いやすいsnake_caseを使っているが、
# 表やグラフでは人間が読みやすい名前に変換する。
#
# 例:
#
# "roc_auc"
#     ↓
# "ROC-AUC"
#
DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "loss": "Loss",
}


def build_dataframe() -> pd.DataFrame:
    """
    保存済みの全実験結果を読み込み、比較用DataFrameを作成する。

    artifacts/results/ 以下に存在するJSONファイルを順番に読み込み、
    各実験の最終評価結果だけを取り出して1つの表にまとめる。

    想定するJSON構造の例
    --------------------
    {
        "experiment": "fedavg",
        "label": "FedAvg",
        "history": [...],
        "final": {
            "accuracy": 0.84,
            "roc_auc": 0.90,
            "pr_auc": 0.75,
            "loss": 0.33
        }
    }

    Returns
    -------
    pd.DataFrame
        各行が1実験に対応する比較用DataFrame。

        例:

        experiment      label        accuracy   roc_auc   pr_auc   loss
        ----------------------------------------------------------------
        centralized     Centralized  0.8532     0.9092    0.7750   0.3197
        fedavg          FedAvg       0.8464     0.9021    0.7582   0.3313

    Raises
    ------
    FileNotFoundError
        artifacts/results/ に実験結果JSONが1つも存在しない場合。
    """

    # 各実験の結果を1行ずつ格納するためのリスト。
    rows = []

    # ---------------------------------------------------------
    # 保存済みの全実験結果JSONを順番に処理
    # ---------------------------------------------------------
    #
    # list_results() は、
    #
    # artifacts/results/
    #
    # に存在する実験結果JSONファイルの一覧を返す。
    #
    # 例:
    #
    # centralized.json
    # fedavg.json
    # fedavg_poisoning.json
    #
    for path in list_results():

        # -----------------------------------------------------
        # JSONファイルをPythonのdictとして読み込む
        # -----------------------------------------------------
        #
        # load_result(path) の返り値例:
        #
        # {
        #     "experiment": "centralized",
        #     "label": "Centralized",
        #     "history": [...],
        #     "final": {...}
        # }
        #
        payload = load_result(path)

        # -----------------------------------------------------
        # 最終評価結果だけを取得
        # -----------------------------------------------------
        #
        # payload.get("final", {})
        #
        # は、
        #
        # "final" が存在する
        #     → そのdictを取得
        #
        # "final" が存在しない
        #     → 空dict {} を返す
        #
        # という意味。
        #
        # [] ではなく get() を使うことで、
        # キーが存在しなくてもKeyErrorにならない。
        #
        final = payload.get("final", {})

        # -----------------------------------------------------
        # まず実験名と表示名を1行分のdictへ格納
        # -----------------------------------------------------
        #
        # path.stem はファイル名から拡張子を除いた部分。
        #
        # 例:
        #
        # Path("fedavg.json").stem
        #     ↓
        # "fedavg"
        #
        # JSONにexperiment/labelが存在しない場合は
        # ファイル名を代わりに使う。
        #
        row = {
            "experiment": payload.get(
                "experiment",
                path.stem,
            ),
            "label": payload.get(
                "label",
                path.stem,
            ),
        }

        # -----------------------------------------------------
        # 比較対象の各指標をrowへ追加
        # -----------------------------------------------------
        #
        # METRICS =
        #
        # [
        #     "accuracy",
        #     "roc_auc",
        #     "pr_auc",
        #     "loss",
        # ]
        #
        # なので、最終的にrowは、
        #
        # {
        #     "experiment": "fedavg",
        #     "label": "FedAvg",
        #     "accuracy": 0.8464,
        #     "roc_auc": 0.9021,
        #     "pr_auc": 0.7582,
        #     "loss": 0.3313
        # }
        #
        # のようになる。
        #
        for metric in METRICS:
            row[metric] = final.get(metric)

        # 1実験分の結果をrowsへ追加。
        rows.append(row)

    # ---------------------------------------------------------
    # 実験結果が1件もなければエラー
    # ---------------------------------------------------------
    #
    # Pythonでは空リスト [] はFalseとして扱われる。
    #
    # つまり、
    #
    # if not rows:
    #
    # は、
    #
    # rowsが空なら
    #
    # という意味。
    #
    if not rows:
        raise FileNotFoundError(
            "No experiment JSON files found in artifacts/results/. "
            "Run centralized and/or Flower experiments first."
        )

    # ---------------------------------------------------------
    # list[dict] → pandas.DataFrame へ変換
    # ---------------------------------------------------------
    #
    # rows =
    #
    # [
    #     {
    #         "experiment": "centralized",
    #         "accuracy": ...
    #     },
    #     {
    #         "experiment": "fedavg",
    #         "accuracy": ...
    #     }
    # ]
    #
    # を表形式へ変換する。
    #
    return pd.DataFrame(rows)


def save_metric_plot(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """
    指定された評価指標について、実験間比較の棒グラフを保存する。

    Parameters
    ----------
    df : pd.DataFrame
        build_dataframe() で作成した実験比較DataFrame。

    metric : str
        グラフ化する評価指標。

        例:
        - "accuracy"
        - "roc_auc"
        - "pr_auc"

    output_dir : Path
        グラフ画像を保存するディレクトリ。

    Notes
    -----
    指定された指標が存在しない実験はグラフから除外する。

    出力例:

        artifacts/comparison/comparison_accuracy.png
        artifacts/comparison/comparison_roc_auc.png
        artifacts/comparison/comparison_pr_auc.png
    """

    # ---------------------------------------------------------
    # 指定されたmetricが欠損している行を除外
    # ---------------------------------------------------------
    #
    # 例えば一部実験でPR-AUCを保存していない場合でも、
    # 比較スクリプト全体が停止しないようにする。
    #
    plot_df = df.dropna(
        subset=[metric]
    )

    # 比較できる実験が1つもなければ何もしない。
    if plot_df.empty:
        return

    # ---------------------------------------------------------
    # MatplotlibのFigureとAxesを生成
    # ---------------------------------------------------------
    #
    # fig:
    #   グラフ全体
    #
    # ax:
    #   実際に棒グラフや軸を書く領域
    #
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    # ---------------------------------------------------------
    # 実験ごとの棒グラフを作成
    # ---------------------------------------------------------
    #
    # x軸:
    #   Centralized
    #   FedAvg
    #   ...
    #
    # y軸:
    #   Accuracyなどの評価指標
    #
    ax.bar(
        plot_df["label"],
        plot_df[metric],
    )

    # y軸名を設定。
    #
    # 例:
    #
    # metric = "roc_auc"
    #     ↓
    # DISPLAY_NAMES["roc_auc"]
    #     ↓
    # "ROC-AUC"
    #
    ax.set_ylabel(
        DISPLAY_NAMES[metric]
    )

    # グラフタイトルを設定。
    ax.set_title(
        f"Experiment comparison: {DISPLAY_NAMES[metric]}"
    )

    # 実験名が長くなっても重ならないよう、
    # x軸ラベルを25度回転させる。
    ax.tick_params(
        axis="x",
        rotation=25,
    )

    # タイトルや軸ラベルが画像の外にはみ出さないよう
    # 自動的に余白を調整する。
    fig.tight_layout()

    # ---------------------------------------------------------
    # PNGファイルとして保存
    # ---------------------------------------------------------
    #
    # 例:
    #
    # comparison_accuracy.png
    #
    fig.savefig(
        output_dir / f"comparison_{metric}.png",
        dpi=160,
    )

    # Figureを閉じてメモリを解放する。
    #
    # 実験数が増えて多数のグラフを生成する場合、
    # closeしないとFigureがメモリ上に残り続ける可能性がある。
    #
    plt.close(fig)


def main() -> None:
    """
    保存済みの全実験結果を比較し、表とグラフを生成する。

    処理の流れ
    ----------
    1. コマンドライン引数を読み込む
    2. artifacts/results/ のJSONを全て読み込む
    3. 実験結果をDataFrameへ変換する
    4. 実験を意味のある順番に並べる
    5. 比較表をターミナルへ表示する
    6. comparison.csvとして保存する
    7. Accuracy / ROC-AUC / PR-AUCのグラフを保存する

    現在の実験結果を以下の流れで比較できる。

    Centralized
        ↓
    FedAvg
        ↓
    FedAvg + Sign Flip
        ↓
    Coordinate-wise Median
        ↓
    Median + Sign Flip
        ↓
    ZK + Median + Sign Flip
        ↓
    ZK + TDX Median + Sign Flip
    """

    # =========================================================
    # 1. コマンドライン引数
    # =========================================================
    parser = argparse.ArgumentParser(
        description="Compare all experiment results"
    )

    # ---------------------------------------------------------
    # --no-plots オプション
    # ---------------------------------------------------------
    #
    # 通常:
    #
    # python -m secure_fl.experiments.compare
    #
    # → 表 + CSV + PNGを生成
    #
    # 以下の場合:
    #
    # python -m secure_fl.experiments.compare --no-plots
    #
    # → 表 + CSVのみ生成
    #
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Print/save comparison table only",
    )

    args = parser.parse_args()

    # =========================================================
    # 2. 全実験結果をDataFrameにまとめる
    # =========================================================
    df = build_dataframe()

    # =========================================================
    # 3. 実験を意味のある順番へ並び替える
    # =========================================================
    #
    # ファイル名順では、
    #
    # centralized
    # fedavg
    # mpc_zk
    # ...
    #
    # のように意図しない順番になる可能性がある。
    #
    # そのため、研究上比較したい順序を明示的に定義する。
    #
    preferred_order = [
        "centralized",
        "fedavg",
        "fedavg_poisoning",
        "plain_median_poisoning",
        "mpc_median_poisoning",
        "mpc_zk",
        "mpc_zk_dp",
    ]

    # ---------------------------------------------------------
    # 実験名 → 順位番号 のdictを作る
    # ---------------------------------------------------------
    #
    # enumerate() を使うことで、
    #
    # {
    #     "centralized": 0,
    #     "fedavg": 1,
    #     "fedavg_poisoning": 2,
    #     ...
    # }
    #
    # となる。
    #
    rank = {
        name: i
        for i, name in enumerate(preferred_order)
    }

    # ---------------------------------------------------------
    # 各実験に一時的な順位列 "_rank" を追加
    # ---------------------------------------------------------
    #
    # rankに存在しない将来の実験については、
    #
    # len(rank)
    #
    # を順位として与える。
    #
    # そのため新しい実験JSONを追加しても、
    # compare.pyを修正しなくても一応表示できる。
    #
    df["_rank"] = df["experiment"].map(
        lambda x: rank.get(
            x,
            len(rank),
        )
    )

    # ---------------------------------------------------------
    # 研究上の順序でDataFrameを並び替える
    # ---------------------------------------------------------
    #
    # まず "_rank"、
    # 同順位なら "experiment" の名前順でソートする。
    #
    # 最後に一時的に作った "_rank" 列を削除する。
    #
    df = (
        df.sort_values(
            ["_rank", "experiment"]
        )
        .drop(columns="_rank")
    )

    # =========================================================
    # 4. ターミナル・CSV表示用DataFrameを作成
    # =========================================================
    #
    # 元DataFrame:
    #
    # experiment
    # label
    # accuracy
    # roc_auc
    # pr_auc
    # loss
    #
    # から必要な列だけ取り出し、
    # 表示名へ変換する。
    #
    display = df[
        [
            "label",
            *METRICS,
        ]
    ].rename(
        columns={
            "label": "Experiment",
            **DISPLAY_NAMES,
        }
    )

    # =========================================================
    # 5. 比較表をターミナルへ表示
    # =========================================================
    print(
        "\n=== Experiment comparison ==="
    )

    # ---------------------------------------------------------
    # DataFrameを文字列へ変換して表示
    # ---------------------------------------------------------
    #
    # index=False:
    #
    # pandasが自動生成する
    #
    # 0
    # 1
    # 2
    #
    # という行番号を表示しない。
    #
    # float_format:
    #
    # 0.853243...
    #
    # を
    #
    # 0.8532
    #
    # のように小数点以下4桁で表示する。
    #
    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # =========================================================
    # 6. 出力ディレクトリを作成
    # =========================================================
    #
    # 保存先:
    #
    # artifacts/comparison/
    #
    output_dir = Path(
        "artifacts/comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # 7. 比較表をCSVとして保存
    # =========================================================
    csv_path = (
        output_dir / "comparison.csv"
    )

    display.to_csv(
        csv_path,
        index=False,
    )

    print(
        f"\nSaved table -> {csv_path}"
    )

    # =========================================================
    # 8. グラフを保存
    # =========================================================
    #
    # --no-plots が指定されていない場合のみ実行する。
    #
    if not args.no_plots:

        # 現在は、
        #
        # Accuracy
        # ROC-AUC
        # PR-AUC
        #
        # の3つをグラフ化する。
        #
        # Lossは比較表には含めるが、
        # 現時点ではグラフ化していない。
        #
        for metric in [
            "accuracy",
            "roc_auc",
            "pr_auc",
        ]:
            save_metric_plot(
                df,
                metric,
                output_dir,
            )

        print(
            f"Saved plots -> "
            f"{output_dir}/comparison_*.png"
        )


# =============================================================
# Pythonのエントリーポイント
# =============================================================
#
# 以下のように直接実行した場合のみmain()を呼ぶ。
#
# python -m secure_fl.experiments.compare
#
# 別のPythonファイルからimportしただけでは、
# 自動的に比較処理は開始されない。
#
if __name__ == "__main__":
    main()
