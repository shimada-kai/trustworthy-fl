from collections.abc import Mapping

import torch


def sign_flip_state_dict(
    local_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    ローカル学習で得たモデル更新量を反転するSign Flip Attackを適用する。

    通常のローカル更新量を

        Δw = w_local - w_global

    とすると、攻撃後の更新量を

        Δw_attack = -scale * Δw

    とする。

    したがってサーバへ返す悪意あるモデル重みは

        w_attack = w_global + Δw_attack
                 = w_global - scale * (w_local - w_global)

    となる。

    Parameters
    ----------
    local_state : Mapping[str, torch.Tensor]
        ローカル学習後のモデルパラメータ。

    global_state : Mapping[str, torch.Tensor]
        ローカル学習前にFlower Serverから受け取った
        Global Modelのパラメータ。

    scale : float, default=1.0
        反転した更新量を何倍にするか。
        1.0なら純粋なSign Flip、
        5.0なら5倍に増幅したSign Flipとなる。

    Returns
    -------
    dict[str, torch.Tensor]
        Sign Flip Attack適用後のstate_dict。
    """
    attacked_state = {}

    for name, local_value in local_state.items():
        # global_valueをlocal_valueと同じdeviceへ移動する
        #
        # 例:
        # local_value  → mps:0
        # global_value → cpu
        #
        #        ↓ .to(local_value.device)
        #
        # local_value  → mps:0
        # global_value → mps:0
        #
        global_value = global_state[name].to(
            local_value.device
        )

        if torch.is_floating_point(local_value):

            # 通常のモデル更新量
            #
            # Δw = w_local - w_global
            #
            update = (
                local_value
                - global_value
            )

            # Sign Flip
            #
            # Δw_attack = -scale * Δw
            #
            # w_attack
            # = w_global - scale * Δw
            #
            attacked_state[name] = (
                global_value
                - scale * update
            )

        else:
            attacked_state[name] = (
                local_value.clone()
            )

    return attacked_state