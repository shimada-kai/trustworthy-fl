from flwr.app import MetricRecord, RecordDict

from secure_fl.fl.server_app import aggregate_train_metrics


def make_record(**metrics):
    return RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "num-examples": 100,
                    **metrics,
                }
            )
        }
    )


def test_aggregate_train_metrics_security_and_overhead():
    records = []

    for _ in range(4):
        records.append(
            make_record(
                train_loss=0.8,
                is_malicious=1,
                zk_accepted=0,
                zk_proving_time_ms=40_000.0,
                zk_verification_time_ms=12.0,
                tdx_attestation_time_ms=0.0,
                tdx_submit_time_ms=0.0,
            )
        )

    for _ in range(3):
        records.append(
            make_record(
                train_loss=0.4,
                is_malicious=0,
                zk_accepted=1,
                zk_proving_time_ms=30_000.0,
                zk_verification_time_ms=10.0,
                tdx_attestation_time_ms=0.0,
                tdx_submit_time_ms=0.0,
            )
        )

    result = aggregate_train_metrics(
        records,
        weighting_metric_name="num-examples",
    )

    assert result["zk_accepted_clients"] == 3
    assert result["zk_rejected_clients"] == 4
    assert result["malicious_clients"] == 4
    assert result["rejected_malicious_clients"] == 4
    assert result["attack_detection_rate"] == 1.0
    assert result["false_reject_rate"] == 0.0
    assert result["zk_proving_time_ms_max"] == 40_000.0
    assert result["zk_verification_time_ms_mean"] > 0.0
