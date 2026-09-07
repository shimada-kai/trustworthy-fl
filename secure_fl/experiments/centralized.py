import argparse
import random
from pathlib import Path

import numpy as np
import torch

from secure_fl.dataset.adult import (
    get_input_dim,
    load_centralized_train_data,
    load_global_test_data,
)
from secure_fl.evaluation.results import save_result
from secure_fl.model.mlp import AdultMLP
from secure_fl.model.train import evaluate_model, train_model


def set_seed(seed: int) -> None:
    """
    実験結果を再現しやすくするため、各ライブラリの乱数シードを固定する。

    対象:
    - Python標準ライブラリ random
    - NumPy
    - PyTorch CPU
    - PyTorch CUDA（NVIDIA GPUが利用可能な場合）

    Parameters
    ----------
    seed : int
        実験全体で使用する乱数シード。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDAはCPUとは別の乱数生成器を持つため、
    # NVIDIA GPUが利用可能な場合はCUDA側のシードも固定する。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    PyTorchで使用する計算デバイスを選択する。

    優先順位:
    1. CUDA : NVIDIA GPU
    2. MPS  : Apple GPU
    3. CPU  : GPUを利用できない場合

    Returns
    -------
    torch.device
        学習および評価に使用するデバイス。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    """
    UCI Adultデータセットを用いた中央学習ベースラインを実行する。

    この実験は、後で連合学習（FedAvg）と性能比較するための
    基準となる中央学習モデルを作成することを目的とする。

    処理の流れ
    ----------
    1. コマンドライン引数を読み込む
    2. 乱数シードを固定する
    3. 計算デバイスを選択する
    4. 中央学習用の訓練データを読み込む
    5. 共通のGlobal Testデータを読み込む
    6. AdultMLPを生成する
    7. Epochごとに学習・評価する
    8. 最終モデルを保存する
    9. 実験設定と評価結果をJSON形式で保存する

    中央学習で使用するデータは、
    連合学習で7クライアントに分散される学習データを
    結合して作成している。

    そのため、

        Centralized Learning
        vs
        Federated Learning

    をほぼ同じ学習データ条件で比較できる。
    """

    # =========================================================
    # 1. コマンドライン引数の定義
    # =========================================================
    #
    # 例えば以下のように実行できる。
    #
    # python -m secure_fl.experiments.centralized \
    #     --epochs 10 \
    #     --batch-size 256 \
    #     --lr 0.001
    #
    parser = argparse.ArgumentParser(
        description="Centralized Adult baseline"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="中央学習を行うEpoch数。",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="学習時に使用するミニバッチサイズ。",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Optimizerで使用する学習率。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="実験再現性のための乱数シード。",
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=7,
        help=(
            "中央学習データを作成するときに基準とする"
            "連合学習のクライアント数。"
        ),
    )

    # 実際にコマンドラインから渡された値を取得する。
    args = parser.parse_args()

    # =========================================================
    # 2. 乱数シードを固定
    # =========================================================
    #
    # データ分割やモデル初期値などのランダム性を
    # できるだけ同じ条件にする。
    #
    set_seed(args.seed)

    # =========================================================
    # 3. 計算デバイスを決定
    # =========================================================
    #
    # Apple Silicon搭載Macの場合は通常、
    #
    # device = mps
    #
    # になる。
    #
    device = get_device()

    # =========================================================
    # 4. 中央学習用のTrainデータを取得
    # =========================================================
    #
    # load_centralized_train_data() は、
    # 連合学習で各クライアントが使用するTrainデータを
    # まとめて1つのDataLoaderにする。
    #
    # イメージ:
    #
    # Client 0のTrainデータ
    #          +
    # Client 1のTrainデータ
    #          +
    # Client 2のTrainデータ
    #          +
    #        ...
    #          +
    # Client 6のTrainデータ
    #          ↓
    # 中央学習用Trainデータ
    #
    trainloader = load_centralized_train_data(
        num_partitions=args.num_clients,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # =========================================================
    # 5. Global Testデータを取得
    # =========================================================
    #
    # このTestデータは、連合学習後のGlobal Modelの評価にも
    # 同じものを使用する。
    #
    # つまり、
    #
    # Centralized Model
    #       │
    #       ├──── 同じTest Data
    #       │
    # Federated Model
    #
    # とすることで性能を公平に比較する。
    #
    testloader = load_global_test_data(
        batch_size=512,
        seed=args.seed,
    )

    # =========================================================
    # 6. ニューラルネットワークを生成
    # =========================================================
    #
    # get_input_dim() は前処理後の特徴量数を取得する。
    #
    # Adultにはカテゴリ変数があるため、
    # One-Hot Encoding後は元データより特徴量数が増える。
    #
    # AdultMLPのネットワーク構造自体は、
    #
    # secure_fl/model/mlp.py
    #
    # に定義されている。
    #
    model = AdultMLP(
        get_input_dim(args.seed)
    )

    # =========================================================
    # 7. 実験条件を表示
    # =========================================================
    print("=== Centralized baseline ===")
    print(f"device        : {device}")
    print(f"train samples : {len(trainloader.dataset)}")
    print(f"test samples  : {len(testloader.dataset)}")
    print(f"epochs        : {args.epochs}")

    # =========================================================
    # 8. 各Epochの評価結果を保存するリスト
    # =========================================================
    #
    # 例えば最終的に以下のような構造になる。
    #
    # history = [
    #     {
    #         "epoch": 1,
    #         "train_loss": 0.41,
    #         "loss": 0.32,
    #         "accuracy": 0.85,
    #         "roc_auc": 0.90,
    #         "pr_auc": 0.77,
    #     },
    #     {
    #         "epoch": 2,
    #         ...
    #     }
    # ]
    #
    history = []

    # =========================================================
    # 9. 中央学習のTraining Loop
    # =========================================================
    for epoch in range(1, args.epochs + 1):

        # -----------------------------------------------------
        # モデルを1 Epochだけ学習する
        # -----------------------------------------------------
        #
        # train_model() の内部では通常のPyTorch学習、
        #
        # forward
        #    ↓
        # loss計算
        #    ↓
        # backward
        #    ↓
        # optimizer.step()
        #
        # が実行される。
        #
        train_loss = train_model(
            model=model,
            trainloader=trainloader,
            epochs=1,
            lr=args.lr,
            device=device,
        )

        # -----------------------------------------------------
        # 現在のモデルをTestデータで評価する
        # -----------------------------------------------------
        #
        # evaluate_model() は以下のようなdictを返す。
        #
        # {
        #     "loss": ...,
        #     "accuracy": ...,
        #     "roc_auc": ...,
        #     "pr_auc": ...
        # }
        #
        metrics = evaluate_model(
            model,
            testloader,
            device,
        )

        # -----------------------------------------------------
        # 現在Epochの結果を1つのdictにまとめる
        # -----------------------------------------------------
        #
        # **metrics はdictを展開するPython構文。
        #
        # 例えば、
        #
        # metrics = {
        #     "accuracy": 0.85,
        #     "roc_auc": 0.90
        # }
        #
        # のとき、
        #
        # {
        #     "epoch": epoch,
        #     **metrics
        # }
        #
        # は、
        #
        # {
        #     "epoch": 1,
        #     "accuracy": 0.85,
        #     "roc_auc": 0.90
        # }
        #
        # になる。
        #
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            **metrics,
        }

        # Epochごとの結果を履歴として追加する。
        history.append(row)

        # -----------------------------------------------------
        # 学習状況をターミナルへ表示
        # -----------------------------------------------------
        print(
            f"[CENTRAL][epoch={epoch}] "
            f"train_loss={train_loss:.4f} "
            f"test_loss={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} "
            f"pr_auc={metrics['pr_auc']:.4f}"
        )

    # =========================================================
    # 10. 学習済みモデルを保存
    # =========================================================
    #
    # 保存先:
    #
    # artifacts/models/centralized.pt
    #
    # parents=True:
    #   親ディレクトリも必要に応じて作成する。
    #
    # exist_ok=True:
    #   既にディレクトリが存在してもエラーにしない。
    #
    Path("artifacts/models").mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = Path(
        "artifacts/models/centralized.pt"
    )

    # state_dict() は、
    #
    #   Linear Layerのweight
    #   Linear Layerのbias
    #
    # など、学習されたモデルパラメータを保持するdict。
    #
    # モデルクラス全体ではなく、
    # 学習されたパラメータだけを保存する。
    #
    torch.save(
        model.state_dict(),
        model_path,
    )

    # =========================================================
    # 11. 実験条件・評価結果をJSONとして保存
    # =========================================================
    #
    # save_result("centralized", ...)
    #
    # とすることで、
    #
    # artifacts/results/centralized.json
    #
    # に保存される。
    #
    # 学習条件と結果を一緒に保存することで、
    # 後から
    #
    # 「このROC-AUCはどの設定で出たのか？」
    #
    # を確認できる。
    #
    result_path = save_result(
        "centralized",
        {
            # -----------------------------
            # 実験名
            # -----------------------------
            "experiment": "centralized",
            "label": "Centralized",
            "family": "centralized",

            # -----------------------------
            # 実験条件
            # -----------------------------
            "num_clients_equivalent": args.num_clients,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,

            # -----------------------------
            # データ数
            # -----------------------------
            "train_samples": len(trainloader.dataset),
            "test_samples": len(testloader.dataset),

            # -----------------------------
            # 全Epochの結果
            # -----------------------------
            "history": history,

            # -----------------------------
            # 最終Epochの結果
            # -----------------------------
            #
            # Pythonでは list[-1] で
            # リストの最後の要素を取得できる。
            #
            "final": history[-1],
        },
    )

    # 保存場所を確認できるように表示する。
    print(f"Saved model   -> {model_path}")
    print(f"Saved metrics -> {result_path}")


# =============================================================
# Pythonのエントリーポイント
# =============================================================
#
# 以下のように実行した場合だけ、
#
# python -m secure_fl.experiments.centralized
#
# main() が呼ばれる。
#
# 一方、別ファイルから
#
# import secure_fl.experiments.centralized
#
# とimportしただけではmain()は実行されない。
#
if __name__ == "__main__":
    main()
