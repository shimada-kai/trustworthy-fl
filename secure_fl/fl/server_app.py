from pathlib import Path

import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from secure_fl.aggregation.median_strategy import CoordinateWiseMedian
from secure_fl.config import NUM_CLIENTS
from secure_fl.dataset.adult import get_input_dim, load_global_test_data
from secure_fl.evaluation.results import load_result, save_result
from secure_fl.model.mlp import AdultMLP
from secure_fl.model.train import evaluate_model

app = ServerApp()


def aggregate_train_metrics(
    records: list[RecordDict],
    weighting_metric_name: str,
) -> MetricRecord:
    """Aggregate training, security, and system-overhead metrics."""

    if not records:
        raise ValueError("records must contain at least one client result")

    client_metrics = [
        next(iter(record.metric_records.values()))
        for record in records
    ]

    weights = [
        float(metrics[weighting_metric_name])
        for metrics in client_metrics
    ]
    total_weight = sum(weights)

    if total_weight <= 0:
        raise ValueError("total client weight must be positive")

    train_loss = sum(
        float(metrics["train_loss"]) * weight
        for metrics, weight in zip(
            client_metrics,
            weights,
            strict=True,
        )
    ) / total_weight

    accepted_clients = sum(
        int(metrics.get("zk_accepted", 1))
        for metrics in client_metrics
    )
    rejected_clients = len(client_metrics) - accepted_clients

    malicious_clients = sum(
        int(metrics.get("is_malicious", 0))
        for metrics in client_metrics
    )
    honest_clients = len(client_metrics) - malicious_clients

    rejected_malicious_clients = sum(
        int(
            int(metrics.get("is_malicious", 0)) == 1
            and int(metrics.get("zk_accepted", 1)) == 0
        )
        for metrics in client_metrics
    )

    false_rejected_honest_clients = sum(
        int(
            int(metrics.get("is_malicious", 0)) == 0
            and int(metrics.get("zk_accepted", 1)) == 0
        )
        for metrics in client_metrics
    )

    def positive_values(key: str) -> list[float]:
        return [
            float(metrics.get(key, 0.0))
            for metrics in client_metrics
            if float(metrics.get(key, 0.0)) > 0.0
        ]

    def mean_or_zero(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    zk_proving_times = positive_values(
        "zk_proving_time_ms"
    )
    zk_verification_times = positive_values(
        "zk_verification_time_ms"
    )
    tdx_attestation_times = positive_values(
        "tdx_attestation_time_ms"
    )
    tdx_submit_times = positive_values(
        "tdx_submit_time_ms"
    )

    return MetricRecord(
        {
            "train_loss": train_loss,
            "zk_accepted_clients": accepted_clients,
            "zk_rejected_clients": rejected_clients,
            "malicious_clients": malicious_clients,
            "rejected_malicious_clients": (
                rejected_malicious_clients
            ),
            "attack_detection_rate": (
                rejected_malicious_clients / malicious_clients
                if malicious_clients
                else 0.0
            ),
            "false_reject_rate": (
                false_rejected_honest_clients / honest_clients
                if honest_clients
                else 0.0
            ),
            "zk_proving_time_ms_mean": mean_or_zero(
                zk_proving_times
            ),
            "zk_proving_time_ms_max": (
                max(zk_proving_times)
                if zk_proving_times
                else 0.0
            ),
            "zk_verification_time_ms_mean": mean_or_zero(
                zk_verification_times
            ),
            "tdx_attestation_time_ms_mean": mean_or_zero(
                tdx_attestation_times
            ),
            "tdx_submit_time_ms_mean": mean_or_zero(
                tdx_submit_times
            ),
        }
    )


def make_global_evaluate(
    seed: int,
    num_rounds: int,
    lr: float,
    experiment_name: str,
    experiment_label: str,
    attack_enabled: bool,
    attack_type: str,
    attack_scale: float,
    malicious_client_ids: str,
):

    history = []

    def global_evaluate(server_round: int, arrays: ArrayRecord):
        model = AdultMLP(get_input_dim(seed))
        model.load_state_dict(arrays.to_torch_state_dict())
        testloader = load_global_test_data(batch_size=512, seed=seed)

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        metrics = evaluate_model(model, testloader, device)

        row = {"round": int(server_round), **metrics}
        history.append(row)

        result_path = save_result(
            experiment_name,
            {
                "experiment": experiment_name,
                "label": experiment_label,
                "family": "federated",

                "num_clients": NUM_CLIENTS,
                "num_rounds": num_rounds,
                "learning_rate": lr,
                "seed": seed,

                # ==================================
                # 攻撃条件
                # ==================================
                "attack_enabled": attack_enabled,

                "attack_type": (
                    attack_type
                    if attack_enabled
                    else None
                ),

                "attack_scale": (
                    attack_scale
                    if attack_enabled
                    else None
                ),

                "malicious_client_ids": (
                    malicious_client_ids
                    if attack_enabled
                    else ""
                ),

                "test_samples": len(
                    testloader.dataset
                ),

                "history": history,
                "final": history[-1],
            },
        )

        print(
            f"[FL GLOBAL TEST][round={server_round}] "
            f"loss={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} "
            f"pr_auc={metrics['pr_auc']:.4f}"
        )
        if server_round == num_rounds:
            print(f"Saved metrics -> {result_path}")

        return {
            "global_loss": metrics["loss"],
            "global_accuracy": metrics["accuracy"],
            "global_roc_auc": metrics["roc_auc"],
            "global_pr_auc": metrics["pr_auc"],
        }

    return global_evaluate


@app.main()
def main(grid: Grid, context: Context) -> None:
    # ==========================================
    # 基本設定
    # ==========================================
    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    seed = int(context.run_config["seed"])
    aggregation_type = str(
        context.run_config["aggregation-type"]
    )
    tee_enabled = bool(
        context.run_config["tee-enabled"]
    )

    zk_verification_enabled = bool(
        context.run_config["zk-verification-enabled"]
    )
    # ==========================================
    # 攻撃設定
    # ==========================================
    attack_enabled = bool(
        context.run_config["attack-enabled"]
    )

    attack_type = str(
        context.run_config["attack-type"]
    )

    attack_scale = float(
        context.run_config["attack-scale"]
    )

    malicious_client_ids = str(
        context.run_config[
            "malicious-client-ids"
        ]
    )
    malicious_count = len(
        [
            client_id
            for client_id in malicious_client_ids.split(",")
            if client_id.strip()
        ]
    )

    attack_label = attack_type.replace("_", " ").title()

    # ==========================================
    # 攻撃あり/なしで実験名を分ける
    # ==========================================
    if aggregation_type == "median":
        if tee_enabled:
            if attack_enabled:
                if zk_verification_enabled:
                    experiment_name = "zk_tdx_median_poisoning"
                    experiment_label = (
                        f"ZK + TDX Median + {attack_label} "
                        f"({malicious_count}/{NUM_CLIENTS})"
                    )
                else:
                    experiment_name = "tdx_median_poisoning"
                    experiment_label = (
                        f"TDX Median + {attack_label} "
                        f"({malicious_count}/{NUM_CLIENTS})"
                    )
            else:
                if zk_verification_enabled:
                    experiment_name = "zk_tdx_median"
                    experiment_label = "ZK + TDX Median"
                else:
                    experiment_name = "tdx_median"
                    experiment_label = "TDX Median"
        else:
            if attack_enabled:
                if zk_verification_enabled:
                    experiment_name = "zk_median_poisoning"
                    experiment_label = (
                        f"ZK + Median + {attack_label} "
                        f"({malicious_count}/{NUM_CLIENTS})"
                    )
                else:
                    experiment_name = f"median_poisoning_{malicious_count}of{NUM_CLIENTS}"
                    experiment_label = (
                        f"Median + {attack_label} "
                        f"({malicious_count}/{NUM_CLIENTS})"
                    )
            else:
                if zk_verification_enabled:
                    experiment_name = "zk_median"
                    experiment_label = "ZK + Median"
                else:
                    experiment_name = "median"
                    experiment_label = "Median"

    elif aggregation_type == "fedavg":
        if attack_enabled:
            experiment_name = "fedavg_poisoning"
            experiment_label = (
                f"FedAvg + {attack_label} "
                f"({malicious_count}/{NUM_CLIENTS})"
            )
        else:
            experiment_name = "fedavg"
            experiment_label = "FedAvg"

    else:
        raise ValueError(
            f"Unknown aggregation type: {aggregation_type}"
        )
    # ==========================================
    # 実験条件をログ表示
    # ==========================================
    print(
        f"[CONFIG] "
        f"aggregation_type={aggregation_type} "
        f"tee_enabled={tee_enabled} "
        f"attack_enabled={attack_enabled} "
        f"attack_type={attack_type} "
        f"malicious_clients={malicious_client_ids} "
        f"attack_scale={attack_scale}"
    )

    torch.manual_seed(seed)

    model = AdultMLP(get_input_dim(seed))
    initial_arrays = ArrayRecord(model.state_dict())

    strategy_kwargs = {
        "fraction_train": 1.0,
        "fraction_evaluate": 1.0,
        "min_train_nodes": NUM_CLIENTS,
        "min_evaluate_nodes": NUM_CLIENTS,
        "min_available_nodes": NUM_CLIENTS,
        "train_metrics_aggr_fn": aggregate_train_metrics,
    }

    if aggregation_type == "median":
        strategy = CoordinateWiseMedian(
            **strategy_kwargs,
            tee_enabled=tee_enabled,
        )
    elif aggregation_type == "fedavg":
        strategy = FedAvg(
            **strategy_kwargs
        )
    else:
        raise ValueError(
            f"Unknown aggregation type: {aggregation_type}"
        )

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,

        train_config=ConfigRecord(
            {
                "lr": lr,
            }
        ),

        num_rounds=num_rounds,

        evaluate_fn=make_global_evaluate(
            seed=seed,
            num_rounds=num_rounds,
            lr=lr,

            experiment_name=experiment_name,
            experiment_label=experiment_label,

            attack_enabled=attack_enabled,
            attack_type=attack_type,
            attack_scale=attack_scale,

            malicious_client_ids=malicious_client_ids,
        ),
    )

    # ==========================================
    # Save aggregated training/security metrics
    # ==========================================
    result_path = Path(
        f"artifacts/results/{experiment_name}.json"
    )

    if result_path.exists():
        payload = load_result(result_path)

        train_metrics_history = [
            {
                "round": int(round_id),
                **{
                    key: (
                        int(value)
                        if isinstance(value, int)
                        else float(value)
                    )
                    for key, value in metrics.items()
                },
            }
            for round_id, metrics
            in sorted(
                result.train_metrics_clientapp.items()
            )
        ]

        payload["train_metrics_history"] = (
            train_metrics_history
        )

        payload["final_train_metrics"] = (
            train_metrics_history[-1]
            if train_metrics_history
            else {}
        )

        save_result(
            experiment_name,
            payload,
        )

    Path("artifacts/models").mkdir(
    parents=True,
    exist_ok=True,
    )

    if result.arrays is not None:

        # 攻撃あり/なしで保存ファイル名を変更する
        #
        # attack_enabled=False
        #   → artifacts/models/fedavg.pt
        #
        # attack_enabled=True
        #   → artifacts/models/fedavg_poisoning.pt
        #
        model_path = Path(
            f"artifacts/models/{experiment_name}.pt"
        )

        torch.save(
            result.arrays.to_torch_state_dict(),
            model_path,
        )

        print(
            f"Saved final model -> {model_path}"
        )
