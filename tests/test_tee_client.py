import torch

from secure_fl.tee.client import aggregate_via_tee_api


class MockResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "aggregated_state": {
                "weight": [2.0, 2.0],
            }
        }


def test_aggregate_via_tee_api(monkeypatch):
    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(
        "secure_fl.tee.client.requests.post",
        mock_post,
    )

    client_states = [
        {"weight": torch.tensor([1.0, 2.0])},
        {"weight": torch.tensor([2.0, 3.0])},
        {"weight": torch.tensor([100.0, -100.0])},
    ]

    result = aggregate_via_tee_api(client_states)

    expected = torch.tensor([2.0, 2.0])

    assert torch.equal(result["weight"], expected)