import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from secure_fl.dataset.adult import get_input_dim, load_client_data
from secure_fl.model.mlp import AdultMLP
from secure_fl.model.train import evaluate_model, train_model
from secure_fl.attack.sign_flip import sign_flip_state_dict

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

    model.load_state_dict(
        global_state
    )

    trainloader, _ = load_client_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        seed=seed,
    )

    loss = train_model(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        lr=lr,
        device=get_device(),
    )

    local_state = model.state_dict()

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

        else:
            raise ValueError(
                f"Unsupported attack type: "
                f"{attack_type}"
            )
    
    arrays = ArrayRecord(local_state)
    metrics = MetricRecord(
        {
            "train_loss": float(loss),

            # 0 = 正常Client
            # 1 = 悪意Client
            "is_malicious": int(
                is_malicious
            ),

            "num-examples": len(
                trainloader.dataset
            ),
        }
    )

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
