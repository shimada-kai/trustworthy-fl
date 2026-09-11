import pytest
import torch
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

from secure_fl.aggregation.median_strategy import CoordinateWiseMedian


def make_reply(weight: float, zk_accepted: int) -> Message:
    content = RecordDict(
        {
            "arrays": ArrayRecord(
                {"weight": torch.tensor([weight])}
            ),
            "metrics": MetricRecord(
                {
                    "zk_accepted": zk_accepted,
                    "num-examples": 1,
                }
            ),
        }
    )

    return Message(
        content=content,
        dst_node_id=0,
        message_type="train",
    )


def test_plain_median_raises_when_all_zk_updates_rejected():
    strategy = CoordinateWiseMedian(
        tee_enabled=False,
        train_metrics_aggr_fn=lambda records, key: MetricRecord({}),
    )

    replies = [
        make_reply(1.0, 0),
        make_reply(2.0, 0),
        make_reply(3.0, 0),
    ]

    with pytest.raises(
        RuntimeError,
        match="No client updates accepted",
    ):
        strategy.aggregate_train(
            server_round=1,
            replies=replies,
        )
