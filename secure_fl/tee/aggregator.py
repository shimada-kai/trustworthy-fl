from collections.abc import Mapping, Sequence

import numpy as np


def aggregate_in_tee(
    client_states: Sequence[Mapping[str, list]],
) -> dict[str, list]:
    """Compute coordinate-wise median inside the TEE."""

    if not client_states:
        raise ValueError("client_states must contain at least one client")

    reference_keys = list(client_states[0].keys())

    for state in client_states[1:]:
        if list(state.keys()) != reference_keys:
            raise ValueError("All client states must have the same keys")

    aggregated_state = {}

    for name in reference_keys:
        arrays = [
            np.asarray(state[name])
            for state in client_states
        ]

        stacked = np.stack(arrays, axis=0)

        median_values = np.median(
            stacked,
            axis=0,
        )

        aggregated_state[name] = median_values.tolist()

    return aggregated_state