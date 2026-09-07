from collections.abc import Mapping, Sequence

import torch


def coordinate_wise_median_state_dict(
    client_states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    複数クライアントのモデルパラメータに対して
    coordinate-wise medianを計算する。

    各パラメータの各要素について、

        w_agg[j] = Median(
            w_1[j],
            w_2[j],
            ...,
            w_n[j],
        )

    を計算する。

    Parameters
    ----------
    client_states:
        各クライアントから受信したstate_dict。

    Returns
    -------
    dict[str, torch.Tensor]
        Coordinate-wise medianで集約されたstate_dict。
    """
    if not client_states:
        raise ValueError(
            "client_states must contain at least one client"
        )

    reference_keys = list(client_states[0].keys())

    for state in client_states[1:]:
        if list(state.keys()) != reference_keys:
            raise ValueError(
                "All client state_dicts must have the same keys"
            )

    aggregated_state = {}

    for name in reference_keys:
        tensors = [
            state[name].detach().cpu()
            for state in client_states
        ]

        # Client軸を先頭に追加
        #
        # 例:
        #
        # Client 1: [1.0, 10.0]
        # Client 2: [2.0, 20.0]
        # Client 3: [100.0, 30.0]
        #
        # stacked:
        #
        # [
        #   [1.0,   10.0],
        #   [2.0,   20.0],
        #   [100.0, 30.0],
        # ]
        #
        stacked = torch.stack(
            tensors,
            dim=0,
        )

        if torch.is_floating_point(stacked):
            median_values = torch.median(
                stacked,
                dim=0,
            ).values

            aggregated_state[name] = median_values
        else:
            # 今回のMLPには基本的に整数parameterはないが、
            # 非float tensorが存在した場合には先頭clientの値を保持する。
            aggregated_state[name] = tensors[0].clone()

    return aggregated_state