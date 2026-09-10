from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

ADULT_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
ADULT_TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week",
]

CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital_status", "occupation", "relationship",
    "race", "sex", "native_country",
]


def _read_adult() -> pd.DataFrame:
    train = pd.read_csv(
        ADULT_TRAIN_URL,
        names=COLUMNS,
        sep=r",\s*",
        engine="python",
        na_values="?",
    )
    test = pd.read_csv(
        ADULT_TEST_URL,
        names=COLUMNS,
        sep=r",\s*",
        engine="python",
        na_values="?",
        skiprows=1,
    )
    test["income"] = test["income"].str.replace(".", "", regex=False)
    df = pd.concat([train, test], ignore_index=True)
    return df.dropna().reset_index(drop=True)


@lru_cache(maxsize=8)
def prepare_adult(seed: int = 42):
    """Load Adult, preprocess it, and create one shared global train/test split."""
    df = _read_adult()
    X = df.drop(columns=["income"])
    y = (df["income"] == ">50K").astype(np.float32).to_numpy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    X_train = preprocessor.fit_transform(X_train_raw).astype(np.float32)
    X_test = preprocessor.transform(X_test_raw).astype(np.float32)

    return X_train, X_test, y_train.astype(np.float32), y_test.astype(np.float32)


def get_input_dim(seed: int = 42) -> int:
    X_train, _, _, _ = prepare_adult(seed)
    return int(X_train.shape[1])


def _to_loader(
    X,
    y,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _client_indices(
    partition_id: int,
    num_partitions: int,
    seed: int,
):
    """Return deterministic local train/validation indices for one client."""
    X_train, _, y_train, _ = prepare_adult(seed)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(X_train))
    partitions = np.array_split(shuffled, num_partitions)
    part_idx = partitions[partition_id]

    local_positions = np.arange(len(part_idx))
    local_train_pos, local_val_pos = train_test_split(
        local_positions,
        test_size=0.20,
        random_state=seed + partition_id,
        stratify=y_train[part_idx],
    )

    return part_idx[local_train_pos], part_idx[local_val_pos]


def load_client_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    seed: int = 42,
):
    """Load one IID client partition and split it into local train/validation."""
    X_train, _, y_train, _ = prepare_adult(seed)
    train_idx, val_idx = _client_indices(partition_id, num_partitions, seed)

    trainloader = _to_loader(
        X_train[train_idx],
        y_train[train_idx],
        batch_size=batch_size,
        shuffle=True,
        seed=seed + partition_id,
    )
    valloader = _to_loader(
        X_train[val_idx], y_train[val_idx], batch_size=batch_size, shuffle=False
    )
    return trainloader, valloader


def load_centralized_train_data(
    num_partitions: int = 7,
    batch_size: int = 256,
    seed: int = 42,
):
    """
    Build the centralized baseline from exactly the samples used for local training.

    This is intentionally NOT all of the global training split. It is the union of
    the seven clients' local-training subsets, so centralized learning and FL see
    exactly the same training examples.
    """
    X_train, _, y_train, _ = prepare_adult(seed)

    train_indices = []
    for partition_id in range(num_partitions):
        client_train_idx, _ = _client_indices(partition_id, num_partitions, seed)
        train_indices.append(client_train_idx)

    train_indices = np.concatenate(train_indices)

    return _to_loader(
        X_train[train_indices],
        y_train[train_indices],
        batch_size=batch_size,
        shuffle=True,
    )


@lru_cache(maxsize=8)
def load_global_test_data(batch_size: int = 512, seed: int = 42):
    """Shared held-out test set used by both centralized learning and FL."""
    _, X_test, _, y_test = prepare_adult(seed)
    return _to_loader(X_test, y_test, batch_size=batch_size, shuffle=False)
