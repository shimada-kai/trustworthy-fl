from collections.abc import Iterable

from flwr.app import ArrayRecord, MetricRecord, Message
from flwr.serverapp.strategy import FedAvg


class CoordinateWiseMedian(FedAvg):
    """Coordinate-wise median aggregation strategy."""

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:

        # Flower標準のvalidation処理を利用
        valid_replies, _ = self._check_and_log_replies(
            replies,
            is_train=True,
        )

        if not valid_replies:
            return None, None

        # ClientAppから返されたArrayRecordのkeyを取得
        record_key = list(
            valid_replies[0].content.array_records.keys()
        )[0]

        # 各clientのモデルをtorch state_dictへ変換
        client_states = [
            msg.content[record_key].to_torch_state_dict()
            for msg in valid_replies
        ]

        # 先ほどテストした純粋なMedian関数を使用
        from secure_fl.aggregation.median import (
            coordinate_wise_median_state_dict,
        )

        median_state = coordinate_wise_median_state_dict(
            client_states
        )

        aggregated_arrays = ArrayRecord(
            median_state
        )

        # training metricsはFedAvgと同じ方法で集約
        reply_contents = [
            msg.content
            for msg in valid_replies
        ]

        metrics = self.train_metrics_aggr_fn(
            reply_contents,
            self.weighted_by_key,
        )

        print(
            f"[MEDIAN][round={server_round}] "
            f"aggregated {len(valid_replies)} client models"
        )

        return aggregated_arrays, metrics