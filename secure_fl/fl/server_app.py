from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from secure_fl.aggregation.median_strategy import CoordinateWiseMedian

from secure_fl.dataset.adult import get_input_dim, load_global_test_data
from secure_fl.evaluation.results import save_result
from secure_fl.model.mlp import AdultMLP
from secure_fl.model.train import evaluate_model

app = ServerApp()


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

                "num_clients": 7,
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
    # ==========================================
    # 攻撃あり/なしで実験名を分ける
    # ==========================================
    if aggregation_type == "median":
        if tee_enabled:
            if attack_enabled:
                experiment_name = "tdx_median_poisoning"
                experiment_label = "TDX Median + Sign Flip (3/7)"
            else:
                experiment_name = "tdx_median"
                experiment_label = "TDX Median"
        else:
            if attack_enabled:
                experiment_name = "median_poisoning"
                experiment_label = "Median + Sign Flip (3/7)"
            else:
                experiment_name = "median"
                experiment_label = "Median"

    elif aggregation_type == "fedavg":
        if attack_enabled:
            experiment_name = "fedavg_poisoning"
            experiment_label = "FedAvg + Sign Flip (3/7)"
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

    model = AdultMLP(get_input_dim(seed))
    initial_arrays = ArrayRecord(model.state_dict())

    strategy_kwargs = {
        "fraction_train": 1.0,
        "fraction_evaluate": 1.0,
        "min_train_nodes": 7,
        "min_evaluate_nodes": 7,
        "min_available_nodes": 7,
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
