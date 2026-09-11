# Trustworthy FL

**Flower + PyTorch + RISC Zero + Intel TDX** を用いた、
連合学習における **Model Poisoning耐性・ZK検証・TEE集約** のPoCです。

UCI Adultデータセットを用いた7クライアントの水平連合学習をベースに、

- FedAvg
- Coordinate-wise Median
- Sign Flip Attack
- RISC ZeroによるSampled ZK Verification
- Intel TDXによるTrusted Aggregation
- Remote Attestation
- TLS Public-Key Binding
- Client → TDX 直接更新送信

を段階的に検証します。

---

## 1. Motivation

Federated Learningでは、生データを中央サーバへ集約せずに複数クライアントでモデルを学習できます。

一方で、モデル更新を送信するクライアント自体が悪意を持つ場合、

```text
Client
  ↓
malicious model update
  ↓
Aggregation Server
  ↓
corrupted Global Model
```

という **Model Poisoning** が問題になります。

Coordinate-wise MedianのようなRobust Aggregationは少数の攻撃者には強い一方、
攻撃者が多数派になると破綻する可能性があります。

本プロジェクトでは、

```text
Robust Aggregation
        +
Sampled ZK Verification
        +
Intel TDX
```

を組み合わせ、

1. 異常なモデル更新を集約前に拒否する
2. accepted updateのみをTDXへ送る
3. 個別更新をFlower Serverへ公開せずTDX内部でMedian集約する

構成を実装しています。

---

## 2. Architecture

```text
                       Flower Server
                            │
                    Global Model
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       Client 0          Client ...        Client 6
          │
          │ Local Training
          ▼
   PyTorch / Adam
          │
          │ sampled training trace
          ▼
     RISC Zero
   ZK Verification
          │
          ├── REJECT ──> update is not submitted
          │
          └── ACCEPT
                │
                │ Remote Attestation
                │ TLS Public-Key Binding
                ▼
        HTTPS Client → TDX
                │
                ▼
        Intel TDX VM
     ┌─────────────────┐
     │ accepted updates │
     │       ↓          │
     │ Coordinate-wise │
     │     Median       │
     └─────────────────┘
                │
                ▼
        Aggregated Model
                │
                ▼
          Flower Server
```

TDXを使用する経路では、acceptedされた個別モデル更新は
**ClientからTDXへ直接送信**されます。

Flower Serverは個々のクライアント更新を受け取らず、
TDXで集約済みのモデルのみを取得します。

---

## 3. Experiment Setup

| Item | Setting |
| --- | --- |
| Dataset         | UCI Adult                     |
| FL type         | Horizontal Federated Learning |
| Clients         | 7                             |
| Model           | MLP                           |
| Input dimension | 104                           |
| Hidden layers   | 128 → 64                      |
| Output          | 1                             |
| Parameters      | 21,761                        |
| Optimizer       | Adam                          |
| Learning rate   | 0.001                         |
| Batch size      | 256                           |
| Local epochs    | 1                             |
| Global rounds   | 5                             |
| Seed            | 42                            |

---

## 4. Model Poisoning

### Sign Flip

通常のlocal updateを

```text
Δw = w_local - w_global
```

とすると、Sign Flipでは

```text
Δw_attack = -scale × Δw
```

を送信します。

つまり、

```text
w_attack
= w_global - scale × (w_local - w_global)
```

となります。

本プロジェクトの標準実験では、

- 3 / 7 attackers
- 4 / 7 attackers

を比較します。

3 / 7はMedianが耐えられるケース、
4 / 7は攻撃者が多数派となりMedianが破綻するケースです。

---

## 5. Coordinate-wise Median

各モデルパラメータについてクライアント方向に中央値を計算します。

```text
Client 0: w0
Client 1: w1
Client 2: w2
...
Client 6: w6

        ↓

median(w0, w1, ..., w6)
```

少数の極端な更新の影響を抑制できます。

ただし、Medianそのものは

```text
悪意のある更新か
```

を判定しているわけではありません。

そのため攻撃者が多数派になると、Medianも攻撃側へ移動します。

---

## 6. RISC Zero Verification

現在のPoCでは、Adam training traceのうち

```text
net.0.weight[0, 0]
```

をサンプルし、

```text
weight_before
gradient
Adam state
weight_after
```

の更新関係をRISC Zeroで検証します。

### Current verification flow

```text
Local Training
      │
      ▼
sampled Adam trace
      │
      ▼
RISC Zero proof
      │
      ▼
proved final weight
      │
      │
      ├───────────────┐
      ▼               ▼
submitted model   compare
                      │
             error <= 1e-6
                │         │
             ACCEPT     REJECT
```

Sign Flipは正常学習の**後**にモデル更新を改ざんするため、
proofで確認したweightと実際にsubmitしようとするweightが一致しなくなり、
現在のPoCではREJECTされます。

### Important limitation

現在証明しているのは、

**sampled parameterに対するAdam更新関係**

です。

gradient自体はRISC Zeroへの入力として与えられているため、

```text
training data
   ↓
forward
   ↓
loss
   ↓
backward
   ↓
gradient
```

というgradient生成過程全体を証明しているわけではありません。

したがって本実装は、

> Sampled parameter-level Adam training verification

であり、

> Full Proof of Training

ではありません。

---

## 7. Intel TDX

ZK verificationを通過したupdateのみ、Intel TDX VMへ直接送信します。

TDX側では、

```text
Client updates
      ↓
TDX protected memory
      ↓
Coordinate-wise Median
      ↓
Aggregated Model
```

を実行します。

### Remote Attestation

モデル更新を送信する前にClient側でRemote Attestationを実行します。

```text
Client
  │
  │ fresh nonce
  ▼
TDX /attest
  │
  ▼
TDX Quote
  │
  ▼
gceprovenance verify
  │
  ▼
tdx-check
```

検証に失敗した場合、個別モデル更新は送信されません。

### TLS Public-Key Binding

Attestationの `REPORT_DATA` に

```text
fresh nonce
+
TLS public-key hash
```

をbindします。

これにより、

```text
Attested TDX instance
```

と

```text
HTTPS endpoint
```

を結び付けます。

---

## 8. Results

5-round experiments:

| Experiment | Accuracy | ROC-AUC | PR-AUC | Loss |
| --- | ---: | ---: | ---: | ---: |
| Centralized                         | 0.8532 | 0.9092     | 0.7750 | 0.3197  |
| FedAvg                              | 0.8436 | 0.9012     | 0.7539 | 0.3333  |
| FedAvg + Sign Flip 3/7              | 0.7550 | 0.8556     | 0.6635 | 0.5894  |
| Median                              | 0.8447 | 0.9016     | 0.7550 | 0.3325  |
| Median + Sign Flip 3/7              | 0.8373 | 0.8940     | 0.7350 | 0.3443  |
| Median + Sign Flip 4/7              | 0.2479 | **0.1846** | 0.1511 | 10.5580 |
| ZK + Median + Sign Flip 4/7         | 0.8448 | **0.9007** | 0.7530 | 0.3338  |
| ZK + TDX Median + Sign Flip 4/7     | 0.8448 | **0.9007** | 0.7530 | 0.3338  |

Medianは3 / 7 attackersでは比較的高い性能を維持します。

一方で4 / 7 attackersでは、

```text
ROC-AUC
0.9016
  ↓
0.1846
```

まで崩壊しました。

RISC Zero verificationにより4 malicious clientsを集約前にrejectすると、

```text
Median + Sign Flip 4/7
ROC-AUC = 0.1846

        ↓ ZK admission filtering

ZK + Median + Sign Flip 4/7
ROC-AUC = 0.9007
```

まで回復しました。

### Security metrics

4 / 7 attack experiment:

```text
Malicious clients         : 4
Rejected malicious clients: 4
Accepted honest clients   : 3

Attack Detection Rate     : 1.0
False Reject Rate         : 0.0
```

---

## 9. System Overhead

実測値は環境によって変動します。

代表的な5-round実験では、

```text
RISC Zero proving
≈ 40–45 sec / client

RISC Zero verification
≈ 10–20 ms

Remote Attestation
≈ 2 sec

HTTPS model submission
≈ 0.2 sec
```

程度でした。

TDX内のMedian computationは数ms〜数十ms程度でした。

Proving timeは実行環境による変動が大きいため、
固定値ではなく参考値として扱います。

---

## 10. Reproduction / Operation Guide

### Python setup

Python 3.11–3.13を使用します。開発環境ではPython 3.12で検証しています。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Flower App初期化

新しいFlower Appを作成する場合:

```bash
flwr new
```

このリポジトリをcloneして使うだけなら、通常は作り直す必要はありません。

### Federation確認

```bash
flwr federation list
```

### 7 SuperNodes設定

本プロジェクトは7クライアントを前提とします。

```bash
flwr federation simulation-config @none/default local \
  --num-supernodes 7 \
  --client-resources-num-cpus 1
```

CPU指定が不要な場合:

```bash
flwr federation simulation-config @none/default local \
  --num-supernodes 7
```

> [!IMPORTANT]
> Flower側の設定を初期化・再作成した場合は、SuperNode数が7と一致しなくなる可能性があります。
> **初期化・再作成後は上記の7 SuperNodes設定を再実行してください。**

通常実行:

```bash
flwr run . --stream
```

全Clientのログを確認したい場合:

```bash
RAY_DEDUP_LOGS=0 flwr run . --stream
```

実験条件は `--run-config` でも上書きできますが、条件の取り違えを避けるため通常は `scripts/*.sh` の利用を推奨します。

### RISC Zero build

Python側はデフォルトで次のrelease binaryを呼び出します。

```text
zkvm/target/release/host
```

Rust / RISC Zero toolchainを準備した上で:

```bash
cd zkvm
cargo build --release
cd ..
```

確認:

```bash
ls -l zkvm/target/release/host
```

zkVM workspace単体では:

```bash
cd zkvm
cargo run
```

ZK有効時のtraining traceは `zkvm/traces/` に生成されます。

---

## 11. Experiments

### 1. Centralized

```bash
./scripts/run_centralized.sh
```

### 2. FedAvg baseline

```bash
./scripts/run_fedavg_baseline.sh
```

### 3. FedAvg + Sign Flip 3/7

```bash
./scripts/run_fedavg_sign_flip.sh
```

### 4. Median baseline

```bash
./scripts/run_median_baseline.sh
```

### 5. Median + Sign Flip 3/7

```bash
./scripts/run_median_sign_flip.sh
```

### 6. Median + Sign Flip 4/7

```bash
./scripts/run_median_sign_flip_4of7.sh
```

### 7. ZK + Median + Sign Flip 4/7

```bash
./scripts/run_zk_median_sign_flip.sh
```

### Local experiment suite

TDXを除く主要な実験をまとめて実行できます。

```bash
./scripts/run_all_experiments.sh
```

TDX実験はCloud VMを使用するため、意図しない課金を避けるため
`run_all_experiments.sh` には含めていません。

---

## 12. TDX Experiment

> [!WARNING]
> 以下のスクリプトはGoogle Cloud上のIntel TDX VMを起動します。
> **VMがRUNNINGになるとCompute料金が発生します。**

```bash
./scripts/run_tdx_zk_median_sign_flip.sh
```

スクリプトは、

```text
VM status確認
   ↓
必要な場合のみVM起動
   ↓
TDX experiment
   ↓
VM停止
   ↓
TERMINATED確認
```

まで実行します。

スクリプト自身が起動したVMのみ自動停止します。

既にRUNNINGだったVMは、自動停止しません。

デフォルト:

```text
VM   : tdx-demo
Zone : asia-northeast1-b
```

必要であれば環境変数で変更できます。

```bash
TDX_VM_NAME=<vm-name> \
TDX_VM_ZONE=<zone> \
./scripts/run_tdx_zk_median_sign_flip.sh
```

---

## 13. Real-time Demo Dashboard

Streamlit dashboardでは実際の5-round experimentを実行し、

```text
Local Training
      ↓
RISC Zero
      ↓
ZK ACCEPT / REJECT
      ↓
Remote Attestation
      ↓
TLS Binding
      ↓
TDX Submit
      ↓
TDX Median
      ↓
Global Model Evaluation
```

をリアルタイム表示します。

> [!WARNING]
> DashboardのTDX demoもGoogle Cloud VMを起動するため、
> **VMがRUNNINGになるとCompute料金が発生します。**

```bash
./scripts/run_demo.sh
```

`Ctrl+C` で終了すると、スクリプト自身が起動したTDX VMを停止し、
`TERMINATED` まで確認します。

---

## 14. Result Files

実験結果は、

```text
artifacts/results/
```

へJSONで保存されます。

例:

```text
centralized.json
fedavg.json
fedavg_poisoning.json
median.json
median_poisoning_3of7.json
median_poisoning_4of7.json
zk_median_poisoning.json
zk_tdx_median_poisoning.json
```

Federated experimentsでは、

```text
history
train_metrics_history
final_train_metrics
```

を保存します。

security / overhead metricsとして、

```text
zk_accepted_clients
zk_rejected_clients
malicious_clients
rejected_malicious_clients
attack_detection_rate
false_reject_rate

zk_proving_time_ms_mean
zk_proving_time_ms_max
zk_verification_time_ms_mean

tdx_attestation_time_ms_mean
tdx_submit_time_ms_mean
tdx_aggregation_time_ms
tdx_accepted_clients
```

などを取得できます。

---

## 15. Tests

```bash
pytest -q
```

Current status:

```text
19 passed
```

テスト対象には、

- Sign Flip
- Adaptive Median attack
- Coordinate-wise Median
- TDX API
- TDX client
- Remote Attestation failure handling
- TDX aggregation metadata
- ZK全reject時のMedian停止
- security / overhead metrics aggregation

などが含まれます。

---

## 16. Project Structure

```text
trustworthy-fl/
├── demo/
│   └── app.py
├── scripts/
│   ├── lib/
│   │   └── tdx_vm.sh
│   ├── run_all_experiments.sh
│   ├── run_centralized.sh
│   ├── run_demo.sh
│   ├── run_fedavg_baseline.sh
│   ├── run_fedavg_sign_flip.sh
│   ├── run_median_baseline.sh
│   ├── run_median_sign_flip.sh
│   ├── run_median_sign_flip_4of7.sh
│   ├── run_tdx_zk_median_sign_flip.sh
│   └── run_zk_median_sign_flip.sh
├── secure_fl/
│   ├── aggregation/
│   ├── attack/
│   ├── dataset/
│   ├── evaluation/
│   ├── experiments/
│   ├── fl/
│   ├── model/
│   ├── tee/
│   └── zk/
├── tests/
├── pyproject.toml
└── README.md
```

---

## 17. Current Security Boundary

このPoCが現在検証している範囲は以下です。

### Implemented

```text
✓ Sign Flip model poisoning
✓ Coordinate-wise Median
✓ Sampled Adam update verification
✓ ZK admission filtering
✓ Client → TDX direct submission
✓ Intel TDX aggregation
✓ Remote Attestation
✓ TLS public-key binding
✓ HTTPS model submission
✓ TDX-side update storage
✓ TDX-side Median aggregation
```

### Not yet guaranteed

```text
✗ Full training computation proof
✗ Dataset correctness proof
✗ Full-model ZK verification
✗ Gradient provenance proof
✗ Byzantine robustness when malicious updates pass admission
✗ Protection against every TEE side channel
```

---

## 18. Future Work

Current verification uses a fixed sampled coordinate:

```text
net.0.weight[0,0]
```

A stronger design is:

```text
Local Training
      ↓
model/update fixed
      ↓
commitment
      ↓
Server random challenge
      ↓
challenged parameter proof
      ↓
proof ↔ submitted model binding
```

とする方式です。

これにより、Proverが事前に検査対象パラメータを知っている現在のPoCよりも、
検査回避を難しくできます。

今後は、

- random challenge
- proof / submitted-model binding
- sampled coordinate拡張
- TEE timing side-channel evaluation

などを検討します。

---

## 19. Important Commands Cheat Sheet

```bash
# Python
source .venv/bin/activate
pip install -e .

# Flower App初期化
flwr new

# Federation確認
flwr federation list

# 7 SuperNodes設定（Flower設定を初期化・再作成した後も再実行）
flwr federation simulation-config @none/default local \
  --num-supernodes 7 \
  --client-resources-num-cpus 1

# Flower通常実行
flwr run . --stream

# 全Clientログ
RAY_DEDUP_LOGS=0 flwr run . --stream

# RISC Zero release build
cd zkvm && cargo build --release && cd ..

# Local experiment suite
./scripts/run_all_experiments.sh

# Tests
pytest -q

# Shell syntax
bash -n scripts/*.sh scripts/lib/*.sh

# Whitespace
git diff --check

# TDX VM status
gcloud compute instances describe tdx-demo \
  --zone=asia-northeast1-b \
  --format='get(status)'

# TDX VM stop
gcloud compute instances stop tdx-demo \
  --zone=asia-northeast1-b
```

> [!WARNING]
> `./scripts/run_tdx_zk_median_sign_flip.sh` と `./scripts/run_demo.sh` は、必要に応じてGoogle Cloud TDX VMを起動します。
> **VMがRUNNINGになるとCompute料金が発生します。**
> 手動で `gcloud compute instances start ...` を実行する場合も同様です。

---

## 20. Disclaimer

本リポジトリは研究・教育目的のPoCです。

実運用のFederated Learning、Zero-Knowledge Proof、TEE、Remote Attestationの
完全なセキュリティを保証するものではありません。
