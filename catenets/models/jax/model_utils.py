"""
Model utils shared across different nets
"""
# Author: Alicia Curth
from typing import Any, Optional

import jax.numpy as jnp
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from catenets.models.constants import DEFAULT_SEED, DEFAULT_VAL_SPLIT
from catenets.datasets.torch_dataset import BaseTorchDataset
from torch.utils.data import Subset


TRAIN_STRING = "training"
VALIDATION_STRING = "validation"


def check_shape_1d_data(y: jnp.ndarray) -> jnp.ndarray:
    # helper func to ensure that output shape won't clash
    # with jax internally
    shape_y = y.shape
    if len(shape_y) == 1:
        # should be shape (n_obs, 1), not (n_obs,)
        return y.reshape((shape_y[0], 1))
    return y


def check_X_is_np(X: pd.DataFrame) -> jnp.ndarray:
    # function to make sure we are using arrays only
    return jnp.asarray(X)


def make_val_split(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: Optional[jnp.ndarray] = None,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    stratify_w: bool = True,
) -> Any:
    if val_split_prop == 0:
        # return original data
        if w is None:
            return X, y, X, y, TRAIN_STRING

        return X, y, w, X, y, w, TRAIN_STRING

    # make actual split
    if w is None:
        X_t, X_val, y_t, y_val = train_test_split(
            X, y, test_size=val_split_prop, random_state=seed, shuffle=True
        )
        return X_t, y_t, X_val, y_val, VALIDATION_STRING

    if stratify_w:
        # split to stratify by group
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X,
            y,
            w,
            test_size=val_split_prop,
            random_state=seed,
            stratify=w,
            shuffle=True,
        )
    else:
        X_t, X_val, y_t, y_val, w_t, w_val = train_test_split(
            X, y, w, test_size=val_split_prop, random_state=seed, shuffle=True
        )

    return X_t, y_t, w_t, X_val, y_val, w_val, VALIDATION_STRING


def make_val_split_with_indices(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: Optional[jnp.ndarray] = None,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    stratify_w: bool = True,
) -> Any:
    if val_split_prop == 0:
        # return original data
        if w is None:
            return X, y, X, y, TRAIN_STRING

        return X, y, w, X, y, w, TRAIN_STRING

    indices = jnp.arange(X.shape[0])
    # make actual split
    if w is None:
        X_t, X_val, y_t, y_val, indices_t, indices_val = train_test_split(
            X, y, indices, test_size=val_split_prop, random_state=seed, shuffle=True
        )
        return X_t, y_t, indices_t, X_val, y_val, indices_val, VALIDATION_STRING

    if stratify_w:
        # split to stratify by group
        X_t, X_val, y_t, y_val, w_t, w_val, indices_t, indices_val = train_test_split(
            X,
            y,
            w,
            indices,
            test_size=val_split_prop,
            random_state=seed,
            stratify=w,
            shuffle=True,
        )
    else:
        X_t, X_val, y_t, y_val, w_t, w_val, indices_t, indices_val = train_test_split(
            X, y, w, indices, test_size=val_split_prop, random_state=seed, shuffle=True
        )

    return X_t, y_t, w_t, indices_t, X_val, y_val, w_val, indices_val, VALIDATION_STRING


def make_val_split_with_reps(
    X: jnp.ndarray,
    reps: jnp.ndarray,
    y: jnp.ndarray,
    w: Optional[jnp.ndarray] = None,
    indices: Optional[jnp.ndarray] = None,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    stratify_w: bool = True,
) -> Any:
    if val_split_prop == 0:
        # return original data
        if w is None:
            return X, y, reps, X, y, reps, TRAIN_STRING

        return X, y, w, reps, X, y, w, reps, TRAIN_STRING

    if indices is None:
        indices = jnp.arange(X.shape[0])

    # make actual split
    if w is None:
        (
            X_t,
            X_val,
            y_t,
            y_val,
            reps_t,
            reps_val,
            indices_t,
            indices_val,
        ) = train_test_split(
            X,
            y,
            reps,
            indices,
            test_size=val_split_prop,
            random_state=seed,
            shuffle=True,
        )
        return (
            X_t,
            y_t,
            reps_t,
            indices_t,
            X_val,
            y_val,
            reps_val,
            indices_val,
            VALIDATION_STRING,
        )

    if stratify_w:
        # split to stratify by group
        (
            X_t,
            X_val,
            y_t,
            y_val,
            w_t,
            w_val,
            reps_t,
            reps_val,
            indices_t,
            indices_val,
        ) = train_test_split(
            X,
            y,
            w,
            reps,
            indices,
            test_size=val_split_prop,
            random_state=seed,
            stratify=w,
            shuffle=True,
        )
    else:
        (
            X_t,
            X_val,
            y_t,
            y_val,
            w_t,
            w_val,
            reps_t,
            reps_val,
            indices_t,
            indices_val,
        ) = train_test_split(
            X,
            y,
            w,
            reps,
            indices,
            test_size=val_split_prop,
            random_state=seed,
            shuffle=True,
        )

    return (
        X_t,
        y_t,
        w_t,
        reps_t,
        indices_t,
        X_val,
        y_val,
        w_val,
        reps_val,
        indices_val,
        VALIDATION_STRING,
    )


from catenets.datasets.torch_dataset import BaseTorchDataset
from torch.utils.data import Subset

def variable_collate_fn(data):
    """Collates the batch of variable sized tensors.
    cats each item vertically

    Args:
        data (_type_): _description_
    """
    collated = []
    for i in range(len(data[0])):
        collated.append(torch.cat([d[i] for d in data]))
    return tuple(collated)


def dict_collate_fn(data):
    """Collates the batch of variable sized tensors.

    Args:
        data (_type_): _description_
    """
    keys = data[0].keys()
    collated = {}
    for k in keys:
        collated[k] = torch.cat([item[k] for item in data])
    return collated


def make_val_split_torch_DS(
    ads_train: BaseTorchDataset,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    stratify_w: bool = True, 
) -> Any:
    """This function splits the PairNet torch dataset into train and val datasets.
    The train and val splits are consistent as the same seed is used to split the indices.
    Args:
        ads_train (AgreementDataset): _description_
        val_split_prop (float, optional): _description_. Defaults to DEFAULT_VAL_SPLIT.
        seed (int, optional): _description_. Defaults to DEFAULT_SEED.
        stratify_w (bool, optional): _description_. Defaults to True.

    Returns:
        Any: _description_
    """
    indices = torch.arange(len(ads_train))
    beta = ads_train.beta
    
    # seed ensures consistency of train/val splits across pairnet and the baselines
    trn_indices, val_indices, _, _ = train_test_split(
        indices,
        beta,
        test_size=val_split_prop,
        stratify=beta,
        random_state=seed,
        shuffle=True,
    )

    # These are indices that are too far from others in the dataset. We better drop such detrimental pairs from the dataset 
    if ads_train.bad_indices is not None and len(ads_train.bad_indices) > 0:
        trn_indices = torch.tensor(
            [i for i in trn_indices if i not in ads_train.bad_indices], dtype=int
        )
        val_indices = torch.tensor(
            [i for i in val_indices if i not in ads_train.bad_indices], dtype=int
        )
        print(f"Removed {len(ads_train) - len(ads_train.bad_indices)} bad indices from training")
    
    if ads_train.bad_indices is not None and len(ads_train.bad_indices) > 0:
        assert len(trn_indices) + len(val_indices) == len(ads_train) - len(
            ads_train.bad_indices
        ), "Bad indices not removed correctly"
        bad_idx = ads_train.bad_indices[0]
        assert (
            bad_idx not in trn_indices and bad_idx not in val_indices
        ), "Bad indices not removed correctly"

    trn_indices, val_indices = torch.sort(trn_indices)[0], torch.sort(val_indices)[0]
    ads_train, ads_val = Subset(ads_train, trn_indices), Subset(ads_train, val_indices)
    return ads_train, ads_val, VALIDATION_STRING


def heads_l2_penalty(
    params_0: jnp.ndarray,
    params_1: jnp.ndarray,
    n_layers_out: jnp.ndarray,
    reg_diff: jnp.ndarray,
    penalty_0: jnp.ndarray,
    penalty_1: jnp.ndarray,
) -> jnp.ndarray:
    # Compute l2 penalty for output heads. Either seperately, or regularizing their difference

    # get l2-penalty for first head
    weightsq_0 = penalty_0 * sum(
        [jnp.sum(params_0[i][0] ** 2) for i in range(0, 2 * n_layers_out + 1, 2)]
    )

    # get l2-penalty for second head
    if reg_diff:
        weightsq_1 = penalty_1 * sum(
            [
                jnp.sum((params_1[i][0] - params_0[i][0]) ** 2)
                for i in range(0, 2 * n_layers_out + 1, 2)
            ]
        )
    else:
        weightsq_1 = penalty_1 * sum(
            [jnp.sum(params_1[i][0] ** 2) for i in range(0, 2 * n_layers_out + 1, 2)]
        )
    return weightsq_1 + weightsq_0
