import pytest
import torch

from secure_fl.tee.client import aggregate_via_tee_api


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