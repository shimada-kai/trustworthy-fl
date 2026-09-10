import torch

from secure_fl.attack.adaptive_median import adaptive_median_state_dict


def test_adaptive_median_moves_to_opposite_side():
    previous_global = {
        "weight": torch.tensor([1.0, 2.0, 3.0]),
    }

    global_state = {
        "weight": torch.tensor([2.0, 1.0, 3.5]),
    }

    local_state = {
        "weight": torch.tensor([2.2, 0.8, 3.7]),
    }

    attacked = adaptive_median_state_dict(
        local_state=local_state,
        global_state=global_state,
        previous_global_state=previous_global,
        scale=1.0,
    )

    # global movement = [+1.0, -1.0, +0.5]
    # local update    = [+0.2, -0.2, +0.2]
    #
    # reference      = [1.0, 1.0, 0.5]
    # direction      = [-1, +1, -1]
    #
    # attacked
    # = global + direction * reference
    expected = torch.tensor([1.0, 2.0, 3.0])

    assert torch.allclose(
        attacked["weight"],
        expected,
    )


def test_adaptive_median_uses_larger_local_update_scale():
    previous_global = {
        "weight": torch.tensor([1.0]),
    }

    global_state = {
        "weight": torch.tensor([1.1]),
    }

    local_state = {
        "weight": torch.tensor([2.1]),
    }

    attacked = adaptive_median_state_dict(
        local_state=local_state,
        global_state=global_state,
        previous_global_state=previous_global,
        scale=2.0,
    )

    # global movement = +0.1
    # local update    = +1.0
    #
    # reference = 1.0
    # direction = -1
    #
    # attacked = 1.1 - 2.0 = -0.9
    expected = torch.tensor([-0.9])

    assert torch.allclose(
        attacked["weight"],
        expected,
        atol=1e-6,
    )


def test_adaptive_median_keeps_non_floating_values():
    previous_global = {
        "counter": torch.tensor([1], dtype=torch.int64),
    }

    global_state = {
        "counter": torch.tensor([2], dtype=torch.int64),
    }

    local_state = {
        "counter": torch.tensor([3], dtype=torch.int64),
    }

    attacked = adaptive_median_state_dict(
        local_state=local_state,
        global_state=global_state,
        previous_global_state=previous_global,
        scale=1.0,
    )

    assert torch.equal(
        attacked["counter"],
        local_state["counter"],
    )
