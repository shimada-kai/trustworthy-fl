import torch

from secure_fl.attack.sign_flip import (
    sign_flip_state_dict,
)


def test_sign_flip_reverses_model_update():
    """
    Sign Flipによってモデル更新量の符号が
    正しく反転されることを確認する。
    """

    # Global Model
    global_state = {
        "weight": torch.tensor(
            [1.0, 2.0]
        )
    }

    # Local Training後
    local_state = {
        "weight": torch.tensor(
            [1.5, 1.0]
        )
    }

    # 通常の更新量は
    #
    # Δw
    # = local - global
    #
    # = [1.5, 1.0]
    #   - [1.0, 2.0]
    #
    # = [+0.5, -1.0]
    #
    attacked = sign_flip_state_dict(
        local_state=local_state,
        global_state=global_state,
        scale=1.0,
    )

    # Sign Flip後は
    #
    # Δw_attack
    # = [-0.5, +1.0]
    #
    # なので、
    #
    # w_attack
    # = global + Δw_attack
    #
    # = [1.0, 2.0]
    #   + [-0.5, +1.0]
    #
    # = [0.5, 3.0]
    #
    expected = torch.tensor(
        [0.5, 3.0]
    )

    assert torch.allclose(
        attacked["weight"],
        expected,
    )