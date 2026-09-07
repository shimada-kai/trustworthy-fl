# Trustworthy FL Starter

Flower + PyTorch + UCI Adult を使った **7クライアント水平連合学習 + 中央学習比較** のベースラインです。
今後、Model Poisoning、Robust Aggregation、MP-SPDZ、ZK/Commitment、DPを同じ実験基盤へ追加できます。

## 比較条件

中央学習とFedAvgで以下を共通にしています。

- UCI Adult
- 同じ前処理
- 同じ3層MLP
- 同じGlobal Test Set
- 同じseed
- 同じOptimizer / learning rate
- 中央学習データ = 7クライアントが実際にLocal Trainへ使うデータの和集合

## Setup

Python 3.12推奨。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

既に `pip install -e .` 済みでも、今回 `matplotlib` を追加したので更新版へ差し替えた後はもう一度実行してください。

## 1. 中央学習

```bash
python -m secure_fl.experiments.centralized --epochs 5
```

保存先:

```text
artifacts/
├── models/
│   └── centralized.pt
└── results/
    └── centralized.json
```

## 2. 7クライアントFedAvg

初回のみ:

```bash
flwr federation simulation-config \
  --num-supernodes 7 \
  --client-resources-num-cpus 1
```

実行:

```bash
flwr run . --stream
```

保存先:

```text
artifacts/
├── models/
│   └── fedavg.pt
└── results/
    └── fedavg.json
```

`fedavg.json` にはround 0〜最終roundのGlobal Test履歴を保存します。

## 3. 全実験を比較

```bash
python -m secure_fl.experiments.compare
```

`artifacts/results/*.json` を自動で読み込みます。そのため、今後PoisoningやMPC実験を追加しても比較コードを書き換える必要はありません。

出力:

```text
artifacts/comparison/
├── comparison.csv
├── comparison_accuracy.png
├── comparison_roc_auc.png
└── comparison_pr_auc.png
```

グラフ不要なら:

```bash
python -m secure_fl.experiments.compare --no-plots
```

## 今後の実験名

```text
centralized.json
fedavg.json
fedavg_poisoning.json
plain_median_poisoning.json
mpc_median_poisoning.json
mpc_zk.json
mpc_zk_dp.json
```

各JSONは共通して `experiment`, `label`, `history`, `final` を持つ形式にします。
