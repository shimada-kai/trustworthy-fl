import torch

from secure_fl.aggregation.median import (
    coordinate_wise_median_state_dict,
)


def test_coordinate_wise_median():
    client_states = [
        {
            "weight": torch.tensor(
                [1.0, 10.0]
            )
        },
        {
            "weight": torch.tensor(
                [2.0, 20.0]
            )
        },
        {
            "weight": torch.tensor(
                [100.0, 30.0]
            )
        },
    ]

    aggregated = coordinate_wise_median_state_dict(
        client_states
    )

    expected = torch.tensor(
        [2.0, 20.0]
    )

    assert torch.allclose(
        aggregated["weight"],
        expected,
    )


def test_coordinate_wise_median_is_robust_to_one_extreme_client():
    client_states = [
        {
            "weight": torch.tensor(
                [1.0, 2.0]
            )
        },
        {
            "weight": torch.tensor(
                [1.1, 1.9]
            )
        },
        {
            "weight": torch.tensor(
                [10000.0, -10000.0]
            )
        },
    ]

    aggregated = coordinate_wise_median_state_dict(
        client_states
    )

    expected = torch.tensor(
        [1.1, 1.9]
    )

    assert torch.allclose(
        aggregated["weight"],
        expected,
    )


def test_coordinate_wise_median_rejects_empty_input():
    try:
        coordinate_wise_median_state_dict([])
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for empty client_states"
    )