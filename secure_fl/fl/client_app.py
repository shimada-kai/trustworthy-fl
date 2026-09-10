import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from secure_fl.dataset.adult import get_input_dim, load_client_data
from secure_fl.model.mlp import AdultMLP
from secure_fl.model.train import evaluate_model, train_model, train_model_with_trace
from secure_fl.attack.sign_flip import sign_flip_state_dict
from secure_fl.attack.adaptive_median import adaptive_median_state_dict
from secure_fl.tee.client import submit_update_via_tee_api
from secure_fl.zk.risc0_bridge import export_adam_trace, run_risc0_verification

app = ClientApp()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def parse_malicious_client_ids(raw_ids: str) -> set[int]:
    """
    "0,1,2" のような文字列を {0, 1, 2} に変換する。

    Parameters
    ----------
    raw_ids : str
        pyproject.tomlなどから取得した
        悪意クライアントIDの文字列。

    Returns
    -------
    set[int]
        悪意クライアントのpartition ID集合。
    """
    if not raw_ids.strip():
        return set()

    return {
        int(client_id.strip())
        for client_id in raw_ids.split(",")
        if client_id.strip()
    }

@app.train()
def train(msg: Message, context: Context):
    partition_id = int(
        context.node_config["partition-id"]
    )

    num_partitions = int(
        context.node_config["num-partitions"]
    )

    batch_size = int(
        context.run_config["batch-size"]
    )

    local_epochs = int(
        context.run_config["local-epochs"]
    )

    seed = int(
        context.run_config["seed"]
    )

    aggregation_type = str(
        context.run_config["aggregation-type"]
    )

    tee_enabled = bool(
        context.run_config["tee-enabled"]
    )

    lr = float(
        msg.content["config"]["lr"]
    )

    # ==========================================
    # Model Poisoningの設定を取得
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

    attack_start_round = int(
        context.run_config["attack-start-round"]
    )

    malicious_client_ids = parse_malicious_client_ids(
        str(
            context.run_config[
                "malicious-client-ids"
            ]
        )
    )

    # このクライアントが攻撃クライアントか判定する。
    #
    # 例:
    #
    # malicious_client_ids = {0, 1, 2}
    #
    # partition_id = 1
    #     ↓
    # is_malicious = True
    #
    is_malicious = (
        attack_enabled
        and partition_id in malicious_client_ids
    )

    model = AdultMLP(
    get_input_dim(seed)
    )

    # ==========================================
    # Serverから受け取ったGlobal Modelを保存
    # ==========================================
    #
    # Sign Flipでは、
    #
    # Δw = w_local - w_global
    #
    # を計算する必要があるため、
    # ローカル学習前のGlobal Modelを保持しておく。
    #
    global_state = {
        name: tensor.clone()
        for name, tensor in (
            msg.content["arrays"]
            .to_torch_state_dict()
            .items()
        )
    }

    server_round = int(
        msg.content["config"]["server-round"]
    )

    # ==========================================
    # Previous Global Model
    # ==========================================
    previous_global_state = None

    if "previous-global" in context.state.array_records:
        previous_global_state = (
            context.state["previous-global"]
            .to_torch_state_dict()
        )

    model.load_state_dict(
        global_state
    )

    trainloader, _ = load_client_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        seed=seed,
    )

    # ==========================================
    # Sampled ZK Verification用 Training Trace
    # ==========================================
    zk_verification_enabled = bool(
        context.run_config["zk-verification-enabled"]
    )

    training_trace = None

    if zk_verification_enabled:
        loss, training_trace = train_model_with_trace(
            model=model,
            trainloader=trainloader,
            epochs=local_epochs,
            lr=lr,
            device=get_device(),
        )

        print(
            f"[ZK-TRACE] "
            f"client={partition_id} "
            f"round={server_round} "
            f"steps={len(training_trace)}"
        )

        trace_path = export_adam_trace(
            training_trace=training_trace,
            client_id=partition_id,
            loss=loss,
            learning_rate=lr,
            output_path=(
                f"zkvm/traces/"
                f"client_{partition_id}_round_{server_round}.json"
            ),
        )

        print(
            f"[ZK-EXPORT] "
            f"client={partition_id} "
            f"round={server_round} "
            f"path={trace_path}"
        )

    else:
        loss = train_model(
            model=model,
            trainloader=trainloader,
            epochs=local_epochs,
            lr=lr,
            device=get_device(),
        )

    local_state = model.state_dict()

    zk_result = None

    if zk_verification_enabled:
        zk_result = run_risc0_verification(trace_path)


    # ==========================================
    # Model Poisoning
    # ==========================================
    #
    # 悪意クライアントに指定されたClientだけ、
    # Serverへ送る直前にモデル更新量を改ざんする。
    #
    if is_malicious:

        if attack_type == "sign_flip":

            local_state = sign_flip_state_dict(
                local_state=local_state,
                global_state=global_state,
                scale=attack_scale,
            )

            print(
                f"[ATTACK] "
                f"client={partition_id} "
                f"type=sign_flip "
                f"scale={attack_scale}"
            )

        elif attack_type == "adaptive_median":

            if (
                server_round >= attack_start_round
                and previous_global_state is not None
            ):

                local_state = adaptive_median_state_dict(
                    local_state=local_state,
                    global_state=global_state,
                    previous_global_state=previous_global_state,
                    scale=attack_scale,
                )

                print(
                    f"[ATTACK] "
                    f"client={partition_id} "
                    f"type=adaptive_median "
                    f"round={server_round} "
                    f"scale={attack_scale}"
                )

            else:
                print(
                    f"[ATTACK-WARMUP] "
                    f"client={partition_id} "
                    f"type=adaptive_median "
                    f"round={server_round}"
                )

        else:
            raise ValueError(
                f"Unsupported attack type: "
                f"{attack_type}"
            )

    
    zk_accepted = True

    if zk_verification_enabled:
        submitted_weight = float(
            local_state["net.0.weight"][0, 0].item()
        )
        proved_weight = float(
            zk_result["proved_final_weight"]
        )

        zk_error = abs(
            submitted_weight - proved_weight
        )
        zk_accepted = zk_error <= 1e-6

        print(
            f"[ZK-VERIFY] "
            f"client={partition_id} "
            f"round={server_round} "
            f"submitted={submitted_weight:.10f} "
            f"proved={proved_weight:.10f} "
            f"error={zk_error:.12f} "
            f"result={'ACCEPT' if zk_accepted else 'REJECT'}"
        )
    
    # ==========================================
    # Direct Client -> TDX submission
    # ==========================================
    context.state["previous-global"] = ArrayRecord(
        global_state
    )
    
    if aggregation_type == "median" and tee_enabled and (not zk_verification_enabled or zk_accepted):
        submit_result = submit_update_via_tee_api(
            round_id=server_round,
            client_id=str(partition_id),
            state=local_state,
        )

        print(
            f"[TDX-SUBMIT] "
            f"round={server_round} "
            f"client={partition_id} "
            f"accepted_updates="
            f"{submit_result['accepted_updates']}"
        )

    elif aggregation_type == "median" and tee_enabled and zk_verification_enabled and not zk_accepted:
        print(
            f"[TDX-SKIP] "
            f"round={server_round} "
            f"client={partition_id} "
            f"reason=zk_rejected"
        )

    metrics = MetricRecord(
        {
            "train_loss": float(loss),
            "is_malicious": int(is_malicious),
            "zk_accepted": int(zk_accepted),
            "num-examples": len(trainloader.dataset),
        }
    )

    # ==========================================
    # Reply to Flower Server
    # ==========================================
    if aggregation_type == "median" and tee_enabled:
        # Individual local_state has already been sent
        # directly to the attested TDX VM.
        #
        # Do NOT return it to the Flower Server.
        return Message(
            content=RecordDict(
                {
                    "metrics": metrics,
                }
            ),
            reply_to=msg,
        )

    # Non-TEE path:
    # FedAvg / Plain Median keep the original behavior.
    arrays = ArrayRecord(local_state)

    return Message(
        content=RecordDict(
            {
                "arrays": arrays,
                "metrics": metrics,
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    batch_size = int(context.run_config["batch-size"])
    seed = int(context.run_config["seed"])

    model = AdultMLP(get_input_dim(seed))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    _, valloader = load_client_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        seed=seed,
    )

    result = evaluate_model(
        model=model,
        dataloader=valloader,
        device=get_device(),
    )

    metrics = MetricRecord(
        {
            "eval_loss": result["loss"],
            "eval_accuracy": result["accuracy"],
            "eval_roc_auc": result["roc_auc"],
            "eval_pr_auc": result["pr_auc"],
            "num-examples": len(valloader.dataset),
        }
    )

    return Message(
        content=RecordDict({"metrics": metrics}),
        reply_to=msg,
    )
