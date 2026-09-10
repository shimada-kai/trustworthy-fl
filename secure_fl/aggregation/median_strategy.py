from collections.abc import Iterable

from flwr.app import ArrayRecord, MetricRecord, Message
from flwr.serverapp.strategy import FedAvg

from secure_fl.aggregation.median import coordinate_wise_median_state_dict
from secure_fl.tee.client import get_aggregated_update_via_tee_api


class CoordinateWiseMedian(FedAvg):
    """Coordinate-wise median aggregation strategy."""

    def __init__(self, *args, tee_enabled: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.tee_enabled = tee_enabled

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:

        valid_replies, _ = self._check_and_log_replies(
            replies,
            is_train=True,
            validate=not self.tee_enabled,
        )

        if not valid_replies:
            return None, None

        # ==========================================
        # TDX Median
        # ==========================================
        if self.tee_enabled:
            median_state, accepted_clients = (
                get_aggregated_update_via_tee_api(
                    round_id=server_round,
                )
            )

            print(
                f"[TDX-MEDIAN][round={server_round}] "
                f"aggregated {accepted_clients} accepted updates"
            )

        # ==========================================
        # Plain Median
        # ==========================================
        else:
            record_key = list(
                valid_replies[0].content.array_records.keys()
            )[0]

            zk_filtered_replies = []

            for msg in valid_replies:
                metrics = msg.content.get("metrics")

                if metrics is None:
                    zk_filtered_replies.append(msg)
                    continue

                zk_accepted = int(
                    metrics.get("zk_accepted", 1)
                )

                if zk_accepted == 1:
                    zk_filtered_replies.append(msg)
                else:
                    print(
                        f"[ZK-FILTER][round={server_round}] "
                        f"rejected client update"
                    )

            client_states = [
                msg.content[record_key].to_torch_state_dict()
                for msg in zk_filtered_replies
            ]

            median_state = coordinate_wise_median_state_dict(
                client_states
            )

            print(
                f"[PLAIN-MEDIAN][round={server_round}] "
                f"aggregated {len(client_states)} client models"
            )

        aggregated_arrays = ArrayRecord(median_state)

        reply_contents = [
            msg.content
            for msg in valid_replies
        ]

        metrics = self.train_metrics_aggr_fn(
            reply_contents,
            self.weighted_by_key,
        )

        return aggregated_arrays, metrics