import pytest
import torch

from secure_fl.tee.client import (
    aggregate_via_tee_api,
    submit_update_via_tee_api,
)


def _client_states():
    return [
        {"weight": torch.tensor([1.0, 2.0])},
        {"weight": torch.tensor([2.0, 3.0])},
        {"weight": torch.tensor([100.0, -100.0])},
    ]


class AggregateResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "aggregated_state": {
                "weight": [2.0, 2.0],
            }
        }


def test_aggregate_is_called_after_successful_attestation(monkeypatch):
    calls = []

    def mock_verify(*args, **kwargs):
        return None

    def mock_post(url, *args, **kwargs):
        calls.append(url)

        assert url.endswith("/aggregate")
        return AggregateResponse()

    monkeypatch.setattr(
        "secure_fl.tee.client._verify_remote_attestation",
        mock_verify,
    )
    monkeypatch.setattr(
        "secure_fl.tee.client.requests.post",
        mock_post,
    )

    result = aggregate_via_tee_api(_client_states())

    assert torch.equal(
        result["weight"],
        torch.tensor([2.0, 2.0]),
    )

    assert len(calls) == 1
    assert calls[0].endswith("/aggregate")


def test_aggregate_is_not_called_when_attestation_fails(monkeypatch):
    aggregate_called = False

    def mock_verify(*args, **kwargs):
        raise RuntimeError("Remote attestation failed")

    def mock_post(*args, **kwargs):
        nonlocal aggregate_called
        aggregate_called = True
        raise AssertionError("/aggregate must not be called")

    monkeypatch.setattr(
        "secure_fl.tee.client._verify_remote_attestation",
        mock_verify,
    )
    monkeypatch.setattr(
        "secure_fl.tee.client.requests.post",
        mock_post,
    )

    with pytest.raises(RuntimeError, match="Remote attestation failed"):
        aggregate_via_tee_api(_client_states())

    assert aggregate_called is False

def test_attestation_rejects_tls_key_hash_mismatch(monkeypatch):
    import base64

    from secure_fl.tee.client import _verify_remote_attestation

    class AttestResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "quote": base64.b64encode(b"dummy-quote").decode(),
                "tls_key_hash": "00" * 32,
            }

    def mock_post(*args, **kwargs):
        return AttestResponse()

    def mock_tls_spki_hash(*args, **kwargs):
        return bytes.fromhex("11" * 32)

    monkeypatch.setattr(
        "secure_fl.tee.client.requests.post",
        mock_post,
    )
    monkeypatch.setattr(
        "secure_fl.tee.client._tls_spki_hash",
        mock_tls_spki_hash,
    )

    with pytest.raises(
        RuntimeError,
        match="TLS public-key hash mismatch",
    ):
        _verify_remote_attestation(
            api_url="https://localhost:8000",
            ca_cert="./server.crt",
            instance="dummy-instance",
            verifier_bin="dummy-verifier",
        )

def test_attestation_runs_tdx_check_after_provenance(monkeypatch, tmp_path):
    import base64
    import subprocess

    from secure_fl.tee.client import _verify_remote_attestation

    expected_hash = bytes.fromhex("11" * 32)
    calls = []

    class AttestResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "quote": base64.b64encode(b"dummy-quote").decode(),
                "tls_key_hash": expected_hash.hex(),
            }

    monkeypatch.setattr(
        "secure_fl.tee.client.requests.post",
        lambda *args, **kwargs: AttestResponse(),
    )
    monkeypatch.setattr(
        "secure_fl.tee.client._tls_spki_hash",
        lambda *args, **kwargs: expected_hash,
    )

    def mock_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        "secure_fl.tee.client.subprocess.run",
        mock_run,
    )

    _verify_remote_attestation(
        api_url="https://localhost:8000",
        ca_cert="./server.crt",
        instance="dummy-instance",
        verifier_bin="/fake/gceprovenance",
        tdx_check_bin="/fake/tdx-check",
    )

    assert len(calls) == 2
    assert calls[0][0] == "/fake/gceprovenance"
    assert calls[0][1] == "verify"

    assert calls[1][0] == "/fake/tdx-check"
    assert "-get_collateral" in calls[1]
    assert "-check_crl" in calls[1]
    assert "-report_data" in calls[1]


def test_attestation_rejects_tdx_check_failure(monkeypatch):
    import base64
    import subprocess

    from secure_fl.tee.client import _verify_remote_attestation

    expected_hash = bytes.fromhex("11" * 32)

    class AttestResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "quote": base64.b64encode(b"dummy-quote").decode(),
                "tls_key_hash": expected_hash.hex(),
            }

    monkeypatch.setattr(
        "secure_fl.tee.client.requests.post",
        lambda *args, **kwargs: AttestResponse(),
    )
    monkeypatch.setattr(
        "secure_fl.tee.client._tls_spki_hash",
        lambda *args, **kwargs: expected_hash,
    )

    call_count = 0

    def mock_run(command, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return subprocess.CompletedProcess(
                command,
                returncode=0,
                stdout="",
                stderr="",
            )

        return subprocess.CompletedProcess(
            command,
            returncode=1,
            stdout="",
            stderr="TCB verification failed",
        )

    monkeypatch.setattr(
        "secure_fl.tee.client.subprocess.run",
        mock_run,
    )

    with pytest.raises(
        RuntimeError,
        match="TDX TCB/CRL verification failed",
    ):
        _verify_remote_attestation(
            api_url="https://localhost:8000",
            ca_cert="./server.crt",
            instance="dummy-instance",
            verifier_bin="/fake/gceprovenance",
            tdx_check_bin="/fake/tdx-check",
        )


def test_submit_update_is_not_called_when_attestation_fails(monkeypatch):
    submit_called = False

    def mock_verify(*args, **kwargs):
        raise RuntimeError("Remote attestation failed")

    def mock_post(*args, **kwargs):
        nonlocal submit_called
        submit_called = True
        raise AssertionError(
            "/submit_update must not be called"
        )

    monkeypatch.setattr(
        "secure_fl.tee.client._verify_remote_attestation",
        mock_verify,
    )
    monkeypatch.setattr(
        "secure_fl.tee.client.requests.post",
        mock_post,
    )

    with pytest.raises(
        RuntimeError,
        match="Remote attestation failed",
    ):
        submit_update_via_tee_api(
            round_id=1,
            client_id="0",
            state={
                "weight": torch.tensor([1.0, 2.0]),
            },
        )

    assert submit_called is False

def test_get_aggregated_update_returns_metrics(monkeypatch):
    from secure_fl.tee.client import get_aggregated_update_via_tee_api

    class AggregateRoundResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "round_id": 1,
                "accepted_clients": 3,
                "aggregation_time_ms": 12.5,
                "aggregated_state": {
                    "weight": [2.0, 3.0],
                },
            }

    def mock_post(url, *args, **kwargs):
        assert url.endswith("/aggregate_round")
        return AggregateRoundResponse()

    monkeypatch.setattr(
        "secure_fl.tee.client.requests.post",
        mock_post,
    )

    state, accepted_clients, aggregation_time_ms = (
        get_aggregated_update_via_tee_api(round_id=1)
    )

    assert torch.equal(
        state["weight"],
        torch.tensor([2.0, 3.0]),
    )
    assert accepted_clients == 3
    assert aggregation_time_ms == 12.5
