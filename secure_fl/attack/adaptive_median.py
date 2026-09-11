from collections.abc import Mapping

import torch


def adaptive_median_state_dict(
    local_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    previous_global_state: Mapping[str, torch.Tensor],
    scale: float = 10.0,
) -> dict[str, torch.Tensor]:
    """
    Coordinate-wise Medianの順位統計量を狙うAdaptive Attack。

    複数のmalicious clientsが同じ規則を独立に実行することを想定する。

    公開Global Modelの移動量

        d = w_global - w_previous

    を各座標で観測し、その学習方向とは反対側へ
    malicious valuesを配置する。

    d_j > 0:
        十分小さい値を送る

        [M, M, M, h1, h2, h3, h4]
                  ^
                median = h1

    d_j < 0:
        十分大きい値を送る

        [h1, h2, h3, h4, M, M, M]
                      ^
                    median = h4

    したがって、攻撃値そのものをMedianにするのではなく、
    Medianをhonest clientsの逆方向側の極値へ押すことを狙う。

    他clientの個別更新は使用しない。
    """

    attacked_state = {}

    for name, local_value in local_state.items():

        if not torch.is_floating_point(local_value):
            attacked_state[name] = local_value.clone()
            continue

        global_value = global_state[name].to(
            local_value.device
        )

        previous_value = previous_global_state[name].to(
            local_value.device
        )

        global_movement = (
            global_value
            - previous_value
        )

        local_update = (
            local_value
            - global_value
        )

        # 座標ごとの攻撃方向。
        #
        # globalが増加している座標 → 小さい側へ
        # globalが減少している座標 → 大きい側へ
        attack_direction = -torch.sign(
            global_movement
        )

        # malicious valueをhonest更新より外側へ配置するため、
        # public global movementと自身のlocal updateのうち
        # 大きい方を座標ごとの基準スケールとする。
        reference_scale = torch.maximum(
            torch.abs(global_movement),
            torch.abs(local_update),
        )

        # movementがほぼ0の座標で完全に攻撃が消えるのを防ぐ。
        reference_scale = torch.clamp(
            reference_scale,
            min=1e-6,
        )

        attacked_state[name] = (
            global_value
            + scale
            * attack_direction
            * reference_scale
        )

    return attacked_state
