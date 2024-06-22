"""
Module implements SNet1 and SNet2, which are based on  CFRNet/TARNet from Shalit et al (2017) and
DragonNet from Shi et al (2019), respectively.
"""
# Author: Alicia Curth
from typing import Any, Callable, List, Tuple

import jax.numpy as jnp
import numpy as onp
import jax
from jax import grad, jit, random
from jax.example_libraries import optimizers, stax
from catenets.datasets.torch_dataset import BaseTorchDataset
from torch.utils.data import DataLoader
from operator import getitem
from jax import nn


import catenets.logger as log
from catenets.models.constants import (
    DEFAULT_AVG_OBJECTIVE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS_OUT,
    DEFAULT_LAYERS_R,
    DEFAULT_N_ITER,
    DEFAULT_N_ITER_MIN,
    DEFAULT_N_ITER_PRINT,
    DEFAULT_NONLIN,
    DEFAULT_PATIENCE,
    DEFAULT_PENALTY_DISC,
    DEFAULT_PENALTY_L2,
    DEFAULT_SEED,
    DEFAULT_STEP_SIZE,
    DEFAULT_UNITS_OUT,
    DEFAULT_UNITS_R,
    DEFAULT_VAL_SPLIT,
    LARGE_VAL,
)
from catenets.models.jax.base import BaseCATENet, OutputHead, ReprBlock
from catenets.models.jax.model_utils import (
    check_shape_1d_data,
    heads_l2_penalty,
    make_val_split,
    make_val_split_torch_DS,
    variable_collate_fn,
    dict_collate_fn,
)


class SNet1(BaseCATENet):
    """
    Class implements Shalit et al (2017)'s TARNet & CFR (discrepancy regularization is NOT
    TESTED). Also referred to as SNet-1 in our paper.

    Parameters
    ----------
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    n_layers_r: int
        Number of shared representation layers before hypothesis layers
    n_units_r: int
        Number of hidden units in each representation layer
    penalty_l2: float
        l2 (ridge) penalty
    step_size: float
        learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    early_stopping: bool, default True
        Whether to use early stopping
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    reg_diff: bool, default False
        Whether to regularize the difference between the two potential outcome heads
    penalty_diff: float
        l2-penalty for regularizing the difference between output heads. used only if
        train_separate=False
    same_init: bool, False
        Whether to initialise the two output heads with same values
    nonlin: string, default 'elu'
        Nonlinearity to use in NN
    penalty_disc: float, default zero
        Discrepancy penalty. Defaults to zero as this feature is not tested.
    """

    def __init__(
        self,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        penalty_l2: float = DEFAULT_PENALTY_L2,
        step_size: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        early_stopping: bool = True,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        reg_diff: bool = False,
        penalty_diff: float = DEFAULT_PENALTY_L2,
        same_init: bool = False,
        nonlin: str = DEFAULT_NONLIN,
        penalty_disc: float = DEFAULT_PENALTY_DISC,
    ) -> None:
        # structure of net
        self.binary_y = binary_y
        self.n_layers_r = n_layers_r
        self.n_layers_out = n_layers_out
        self.n_units_r = n_units_r
        self.n_units_out = n_units_out
        self.nonlin = nonlin

        # penalties
        self.penalty_l2 = penalty_l2
        self.penalty_disc = penalty_disc
        self.reg_diff = reg_diff
        self.penalty_diff = penalty_diff
        self.same_init = same_init

        # training params
        self.step_size = step_size
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.val_split_prop = val_split_prop
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter_min = n_iter_min

    def _get_train_function(self) -> Callable:
        return train_snet1

    def _get_predict_function(self) -> Callable:
        return predict_snet1

    def _get_repr_function(self) -> Callable:
        return getrepr_snet1


class TARNet(SNet1):
    """Wrapper for TARNet"""

    def __init__(
        self,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        penalty_l2: float = DEFAULT_PENALTY_L2,
        step_size: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        early_stopping: bool = True,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        reg_diff: bool = False,
        penalty_diff: float = DEFAULT_PENALTY_L2,
        same_init: bool = False,
        nonlin: str = DEFAULT_NONLIN,
    ):
        super().__init__(
            binary_y=binary_y,
            n_layers_r=n_layers_r,
            n_units_r=n_units_r,
            n_layers_out=n_layers_out,
            n_units_out=n_units_out,
            penalty_l2=penalty_l2,
            step_size=step_size,
            n_iter=n_iter,
            batch_size=batch_size,
            val_split_prop=val_split_prop,
            early_stopping=early_stopping,
            patience=patience,
            n_iter_min=n_iter_min,
            n_iter_print=n_iter_print,
            seed=seed,
            reg_diff=reg_diff,
            penalty_diff=penalty_diff,
            same_init=same_init,
            nonlin=nonlin,
            penalty_disc=0,
        )


class CFRNet(SNet1):
    """Wrapper for TARNet"""

    def __init__(
        self,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        penalty_l2: float = DEFAULT_PENALTY_L2,
        step_size: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        early_stopping: bool = True,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        reg_diff: bool = False,
        penalty_diff: float = DEFAULT_PENALTY_L2,
        same_init: bool = False,
        nonlin: str = DEFAULT_NONLIN,
        penalty_disc: float = DEFAULT_PENALTY_DISC, # This is the only difference between TARNet and CFRNet
    ): 
        super().__init__(
            binary_y=binary_y,
            n_layers_r=n_layers_r,
            n_units_r=n_units_r,
            n_layers_out=n_layers_out,
            n_units_out=n_units_out,
            penalty_l2=penalty_l2,
            step_size=step_size,
            n_iter=n_iter,
            batch_size=batch_size,
            val_split_prop=val_split_prop,
            early_stopping=early_stopping,
            patience=patience,
            n_iter_min=n_iter_min,
            n_iter_print=n_iter_print,
            seed=seed,
            reg_diff=reg_diff,
            penalty_diff=penalty_diff,
            same_init=same_init,
            nonlin=nonlin,
            penalty_disc=penalty_disc,
        )


class SNet2(BaseCATENet):
    """
    Class implements SNet-2, which is based on Shi et al (2019)'s DragonNet (this version does
    NOT use targeted regularization and has a (possibly deeper) propensity head.

    Parameters
    ----------
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_layers_out_prop: int
        Number of hypothesis layers for propensity score(n_layers_out x n_units_out + 1 x Dense
        layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    n_units_out_prop: int
        Number of hidden units in each propensity score hypothesis layer
    n_layers_r: int
        Number of shared representation layers before hypothesis layers
    n_units_r: int
        Number of hidden units in each representation layer
    penalty_l2: float
        l2 (ridge) penalty
    step_size: float
        learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    early_stopping: bool, default True
        Whether to use early stopping
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    reg_diff: bool, default False
        Whether to regularize the difference between the two potential outcome heads
    penalty_diff: float
        l2-penalty for regularizing the difference between output heads. used only if
        train_separate=False
    same_init: bool, False
        Whether to initialise the two output heads with same values
    nonlin: string, default 'elu'
        Nonlinearity to use in NN
    """

    def __init__(
        self,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        penalty_l2: float = DEFAULT_PENALTY_L2,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
        step_size: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        early_stopping: bool = True,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        reg_diff: bool = False,
        same_init: bool = False,
        penalty_diff: float = DEFAULT_PENALTY_L2,
        nonlin: str = DEFAULT_NONLIN,
    ) -> None:
        self.binary_y = binary_y

        self.n_layers_r = n_layers_r
        self.n_layers_out = n_layers_out
        self.n_layers_out_prop = n_layers_out_prop
        self.n_units_r = n_units_r
        self.n_units_out = n_units_out
        self.n_units_out_prop = n_units_out_prop
        self.nonlin = nonlin

        self.penalty_l2 = penalty_l2
        self.step_size = step_size
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.reg_diff = reg_diff
        self.penalty_diff = penalty_diff
        self.same_init = same_init

        self.seed = seed
        self.n_iter_print = n_iter_print

    def _get_train_function(self) -> Callable:
        return train_snet2

    def _get_predict_function(self) -> Callable:
        return predict_snet2

    def _getrepr_function(self) -> Callable:
        return getrepr_snet2


class DragonNet(SNet2):
    """Wrapper for DragonNet"""

    def __init__(
        self,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        penalty_l2: float = DEFAULT_PENALTY_L2,
        n_units_out_prop: int = DEFAULT_UNITS_OUT,
        n_layers_out_prop: int = 0,
        step_size: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        early_stopping: bool = True,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        reg_diff: bool = False,
        same_init: bool = False,
        penalty_diff: float = DEFAULT_PENALTY_L2,
        nonlin: str = DEFAULT_NONLIN,
    ):
        super().__init__(
            binary_y=binary_y,
            n_layers_r=n_layers_r,
            n_units_r=n_units_r,
            n_layers_out=n_layers_out,
            n_units_out=n_units_out,
            penalty_l2=penalty_l2,
            n_units_out_prop=n_units_out_prop,
            n_layers_out_prop=n_layers_out_prop,
            step_size=step_size,
            n_iter=n_iter,
            batch_size=batch_size,
            val_split_prop=val_split_prop,
            early_stopping=early_stopping,
            patience=patience,
            n_iter_min=n_iter_min,
            n_iter_print=n_iter_print,
            seed=seed,
            reg_diff=reg_diff,
            penalty_diff=penalty_diff,
            same_init=same_init,
            nonlin=nonlin,
        )


# Training functions for SNet1 -------------------------------------------------
def mmd2_lin(X: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    # Squared Linear MMD as implemented in CFR
    # jax does not support indexing, so this is a workaround with reweighting in means
    n = w.shape[0]
    n_t = jnp.sum(w)

    # normalize X so scale matters
    X = X / jnp.sqrt(jnp.var(X, axis=0))

    mean_control = (n / (n - n_t)) * jnp.mean((1 - w) * X, axis=0)
    mean_treated = (n / n_t) * jnp.mean(w * X, axis=0)

    return jnp.sum((mean_treated - mean_control) ** 2)


def predict_snet1(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
    return_po: bool = False,
    return_prop: bool = False,
) -> jnp.ndarray:
    if return_prop:
        raise NotImplementedError("SNet1 does not implement a propensity model.")

    # unpack inputs
    predict_fun_repr, predict_fun_head = predict_funs
    param_repr, param_0, param_1 = (
        trained_params[0],
        trained_params[1],
        trained_params[2],
    )

    # get representation
    representation = predict_fun_repr(param_repr, X)

    # get potential outcomes
    mu_0 = predict_fun_head(param_0, representation)
    mu_1 = predict_fun_head(param_1, representation)

    if return_po:
        return mu_1 - mu_0, mu_0, mu_1
    else:
        return mu_1 - mu_0


def getrepr_snet1(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
) -> jnp.ndarray:
    # unpack inputs
    predict_fun_repr, predict_fun_head = predict_funs
    param_repr, param_0, param_1 = (
        trained_params[0],
        trained_params[1],
        trained_params[2],
    )

    # get representation
    representation = predict_fun_repr(param_repr, X)

    return representation


def train_snet1(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    binary_y: bool = False,
    n_layers_r: int = DEFAULT_LAYERS_R,
    n_units_r: int = DEFAULT_UNITS_R,
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_units_out: int = DEFAULT_UNITS_OUT,
    penalty_l2: float = DEFAULT_PENALTY_L2,
    penalty_disc: int = DEFAULT_PENALTY_DISC,
    step_size: float = DEFAULT_STEP_SIZE,
    n_iter: int = DEFAULT_N_ITER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    early_stopping: bool = True,
    patience: int = DEFAULT_PATIENCE,
    n_iter_min: int = DEFAULT_N_ITER_MIN,
    n_iter_print: int = DEFAULT_N_ITER_PRINT,
    seed: int = DEFAULT_SEED,
    return_val_loss: bool = False,
    reg_diff: bool = False,
    same_init: bool = False,
    penalty_diff: float = DEFAULT_PENALTY_L2,
    nonlin: str = DEFAULT_NONLIN,
    avg_objective: bool = DEFAULT_AVG_OBJECTIVE,
) -> Any:
    # function to train TARNET/CFRNet (Johansson et al) using jax
    # input check
    y, w = check_shape_1d_data(y), check_shape_1d_data(w)
    d = X.shape[1]
    input_shape = (-1, d)
    rng_key = random.PRNGKey(seed)
    onp.random.seed(seed)  # set seed for data generation via numpy as well

    if not reg_diff:
        penalty_diff = penalty_l2

    # get validation split (can be none)
    X, y, w, X_val, y_val, w_val, val_string = make_val_split(
        X, y, w, val_split_prop=val_split_prop, seed=seed
    )
    n = X.shape[0]  # could be different from before due to split

    # get representation layer
    init_fun_repr, predict_fun_repr = ReprBlock(
        n_layers=n_layers_r, n_units=n_units_r, nonlin=nonlin
    )

    # get output head functions (both heads share same structure)
    init_fun_head, predict_fun_head = OutputHead(
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        binary_y=binary_y,
        nonlin=nonlin,
    )

    def init_fun_snet1(rng: float, input_shape: Tuple) -> Tuple[Tuple, List]:
        # chain together the layers
        # param should look like [repr, po_0, po_1]
        rng, layer_rng = random.split(rng)
        input_shape_repr, param_repr = init_fun_repr(layer_rng, input_shape)
        rng, layer_rng = random.split(rng)
        if same_init:
            # initialise both on same values
            input_shape, param_0 = init_fun_head(layer_rng, input_shape_repr)
            input_shape, param_1 = init_fun_head(layer_rng, input_shape_repr)
        else:
            input_shape, param_0 = init_fun_head(layer_rng, input_shape_repr)
            rng, layer_rng = random.split(rng)
            input_shape, param_1 = init_fun_head(layer_rng, input_shape_repr)

        return input_shape, [param_repr, param_0, param_1]

    # Define loss functions
    # loss functions for the head
    if not binary_y:

        def loss_head(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch
            preds = predict_fun_head(params, inputs)
            return jnp.sum(weights * ((preds - targets) ** 2))

    else:

        def loss_head(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch
            preds = predict_fun_head(params, inputs)
            return -jnp.sum(
                weights
                * (targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))
            )

    # complete loss function for all parts
    @jit
    def loss_snet1(
        params: List,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        penalty_l2: float,
        penalty_disc: float,  # NOTE: This is 0 for tarnet
        penalty_diff: float,
    ) -> jnp.ndarray:
        # params: list[representation, head_0, head_1]
        # batch: (X, y, w)
        X, y, w = batch

        # get representation
        reps = predict_fun_repr(params[0], X)

        # get mmd
        disc = mmd2_lin(reps, w)

        # pass down to two heads
        loss_0 = loss_head(params[1], (reps, y, 1 - w))
        loss_1 = loss_head(params[2], (reps, y, w))

        # regularization on representation
        weightsq_body = sum(
            [jnp.sum(params[0][i][0] ** 2) for i in range(0, 2 * n_layers_r, 2)]
        )
        weightsq_head = heads_l2_penalty(
            params[1], params[2], n_layers_out, reg_diff, penalty_l2, penalty_diff
        )
        if not avg_objective:
            return (
                loss_0
                + loss_1
                + penalty_disc * disc
                + (penalty_l2 * weightsq_body + 0 * weightsq_head)
            )
        else:
            # We found that L2 penalty on Phi and no penalty on mu generally helps the model
            n_batch = y.shape[0]
            return (
                (loss_0 + loss_1) / n_batch
                + penalty_disc * disc
                + (penalty_l2 * weightsq_body + 0 * weightsq_head)
            )

    # Define optimisation routine
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    @jit
    def update(
        i: int, state: dict, batch: jnp.ndarray, penalty_l2: float, penalty_disc: float
    ) -> jnp.ndarray:
        # updating function
        params = get_params(state)
        return opt_update(
            i,
            grad(loss_snet1)(params, batch, penalty_l2, penalty_disc, penalty_diff),
            state,
        )

    # initialise states
    _, init_params = init_fun_snet1(rng_key, input_shape)
    opt_state = opt_init(init_params)

    # calculate number of batches per epoch
    batch_size = batch_size if batch_size < n else n
    n_batches = int(onp.round(n / batch_size)) if batch_size < n else 1
    train_indices = onp.arange(n)

    l_best = LARGE_VAL
    p_curr = 0

    # do training
    for i in range(n_iter):
        # shuffle data for minibatches
        onp.random.shuffle(train_indices)
        for b in range(n_batches):
            idx_next = train_indices[
                (b * batch_size) : min((b + 1) * batch_size, n - 1)
            ]
            next_batch = X[idx_next, :], y[idx_next, :], w[idx_next]
            opt_state = update(
                i * n_batches + b, opt_state, next_batch, penalty_l2, penalty_disc
            )

        if (i % n_iter_print == 0) or early_stopping:
            params_curr = get_params(opt_state)
            l_curr = loss_snet1(
                params_curr,
                (X_val, y_val, w_val),
                penalty_l2,
                penalty_disc,
                penalty_diff,
            )

        if i % n_iter_print == 0:
            log.info(f"Epoch: {i}, current {val_string} loss {l_curr}")

        if early_stopping:
            if l_curr < l_best:
                l_best = l_curr
                p_curr = 0
                params_best = params_curr
            else:
                if onp.isnan(l_curr):
                    # if diverged, return best
                    return params_best, (predict_fun_repr, predict_fun_head)
                p_curr = p_curr + 1

            if p_curr > patience and ((i + 1) * n_batches > n_iter_min):
                if return_val_loss:
                    # return loss without penalty
                    l_final = loss_snet1(params_curr, (X_val, y_val, w_val), 0, 0, 0)
                    return params_curr, (predict_fun_repr, predict_fun_head), l_final

                return params_curr, (predict_fun_repr, predict_fun_head)

    # return the parameters
    trained_params = get_params(opt_state)

    if return_val_loss:
        # return loss without penalty
        l_final = loss_snet1(get_params(opt_state), (X_val, y_val, w_val), 0, 0, 0)
        return trained_params, (predict_fun_repr, predict_fun_head), l_final

    return trained_params, (predict_fun_repr, predict_fun_head)


# SNET-2 -----------------------------------------------------------------------------------------
def train_snet2(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    binary_y: bool = False,
    n_layers_r: int = DEFAULT_LAYERS_R,
    n_units_r: int = DEFAULT_UNITS_R,
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_units_out: int = DEFAULT_UNITS_OUT,
    penalty_l2: float = DEFAULT_PENALTY_L2,
    n_units_out_prop: int = DEFAULT_UNITS_OUT,
    n_layers_out_prop: int = DEFAULT_LAYERS_OUT,
    step_size: float = DEFAULT_STEP_SIZE,
    n_iter: int = DEFAULT_N_ITER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    early_stopping: bool = True,
    patience: int = DEFAULT_PATIENCE,
    n_iter_min: int = DEFAULT_N_ITER_MIN,
    n_iter_print: int = DEFAULT_N_ITER_PRINT,
    seed: int = DEFAULT_SEED,
    return_val_loss: bool = False,
    reg_diff: bool = False,
    penalty_diff: float = DEFAULT_PENALTY_L2,
    nonlin: str = DEFAULT_NONLIN,
    avg_objective: bool = DEFAULT_AVG_OBJECTIVE,
    same_init: bool = False,
) -> Any:
    """
    SNet2 corresponds to DragonNet (Shi et al, 2019) [without TMLE regularisation term].
    """
    y, w = check_shape_1d_data(y), check_shape_1d_data(w)
    d = X.shape[1]
    input_shape = (-1, d)
    rng_key = random.PRNGKey(seed)
    onp.random.seed(seed)  # set seed for data generation via numpy as well

    if not reg_diff:
        penalty_diff = penalty_l2

    # get validation split (can be none)
    X, y, w, X_val, y_val, w_val, val_string = make_val_split(
        X, y, w, val_split_prop=val_split_prop, seed=seed
    )
    n = X.shape[0]  # could be different from before due to split

    # get representation layer
    init_fun_repr, predict_fun_repr = ReprBlock(
        n_layers=n_layers_r, n_units=n_units_r, nonlin=nonlin
    )

    # get output head functions (output heads share same structure)
    init_fun_head_po, predict_fun_head_po = OutputHead(
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        binary_y=binary_y,
        nonlin=nonlin,
    )
    # add propensity head
    init_fun_head_prop, predict_fun_head_prop = OutputHead(
        n_layers_out=n_layers_out_prop,
        n_units_out=n_units_out_prop,
        binary_y=True,
        nonlin=nonlin,
    )

    def init_fun_snet2(rng: float, input_shape: Tuple) -> Tuple[Tuple, List]:
        # chain together the layers
        # param should look like [repr, po_0, po_1, prop]
        rng, layer_rng = random.split(rng)
        input_shape_repr, param_repr = init_fun_repr(layer_rng, input_shape)

        rng, layer_rng = random.split(rng)
        if same_init:
            # initialise both on same values
            input_shape, param_0 = init_fun_head_po(layer_rng, input_shape_repr)
            input_shape, param_1 = init_fun_head_po(layer_rng, input_shape_repr)
        else:
            input_shape, param_0 = init_fun_head_po(layer_rng, input_shape_repr)
            rng, layer_rng = random.split(rng)
            input_shape, param_1 = init_fun_head_po(layer_rng, input_shape_repr)
        rng, layer_rng = random.split(rng)
        input_shape, param_prop = init_fun_head_prop(layer_rng, input_shape_repr)
        return input_shape, [param_repr, param_0, param_1, param_prop]

    # Define loss functions
    # loss functions for the head
    if not binary_y:

        def loss_head(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch
            preds = predict_fun_head_po(params, inputs)
            return jnp.sum(weights * ((preds - targets) ** 2))

    else:

        def loss_head(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> jnp.ndarray:
            # log loss function
            inputs, targets, weights = batch
            preds = predict_fun_head_po(params, inputs)
            return -jnp.sum(
                weights
                * (targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))
            )

    def loss_head_prop(
        params: List, batch: Tuple[jnp.ndarray, jnp.ndarray], penalty: float
    ) -> jnp.ndarray:
        # log loss function for propensities
        inputs, targets = batch
        preds = predict_fun_head_prop(params, inputs)

        return -jnp.sum(targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))

    # complete loss function for all parts
    @jit
    def loss_snet2(
        params: List,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        penalty_l2: float,
        penalty_diff: float,
    ) -> jnp.ndarray:
        # params: list[representation, head_0, head_1, head_prop]
        # batch: (X, y, w)
        X, y, w = batch

        # get representation
        reps = predict_fun_repr(params[0], X)

        # pass down to heads
        loss_0 = loss_head(params[1], (reps, y, 1 - w))
        loss_1 = loss_head(params[2], (reps, y, w))

        # pass down to propensity head
        loss_prop = loss_head_prop(params[3], (reps, w), penalty_l2)
        weightsq_prop = sum(
            [
                jnp.sum(params[3][i][0] ** 2)
                for i in range(0, 2 * n_layers_out_prop + 1, 2)
            ]
        )

        weightsq_body = sum(
            [jnp.sum(params[0][i][0] ** 2) for i in range(0, 2 * n_layers_r, 2)]
        )
        weightsq_head = heads_l2_penalty(
            params[1], params[2], n_layers_out, reg_diff, penalty_l2, penalty_diff
        )

        if not avg_objective:
            return (
                loss_0
                + loss_1
                + loss_prop
                + (penalty_l2 * (weightsq_body + weightsq_prop) + 0 * weightsq_head)
            )
        else:
            n_batch = y.shape[0]
            return (
                (loss_0 + loss_1) / n_batch
                + loss_prop / n_batch
                + (penalty_l2 * (weightsq_body + weightsq_prop) + 0 * weightsq_head)
            )

    # Define optimisation routine
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    @jit
    def update(
        i: int, state: dict, batch: jnp.ndarray, penalty_l2: float, penalty_diff: float
    ) -> jnp.ndarray:
        # updating function
        params = get_params(state)
        return opt_update(
            i, grad(loss_snet2)(params, batch, penalty_l2, penalty_diff), state
        )

    # initialise states
    _, init_params = init_fun_snet2(rng_key, input_shape)
    opt_state = opt_init(init_params)

    # calculate number of batches per epoch
    batch_size = batch_size if batch_size < n else n
    n_batches = int(onp.round(n / batch_size)) if batch_size < n else 1
    train_indices = onp.arange(n)

    l_best = LARGE_VAL
    p_curr = 0

    # do training
    for i in range(n_iter):
        # shuffle data for minibatches
        onp.random.shuffle(train_indices)
        for b in range(n_batches):
            idx_next = train_indices[
                (b * batch_size) : min((b + 1) * batch_size, n - 1)
            ]
            next_batch = X[idx_next, :], y[idx_next, :], w[idx_next]
            opt_state = update(
                i * n_batches + b, opt_state, next_batch, penalty_l2, penalty_diff
            )

        if (i % n_iter_print == 0) or early_stopping:
            params_curr = get_params(opt_state)
            l_curr = loss_snet2(
                params_curr, (X_val, y_val, w_val), penalty_l2, penalty_diff
            )

        if i % n_iter_print == 0:
            log.info(f"Epoch: {i}, current {val_string} loss {l_curr}")

        if early_stopping and ((i + 1) * n_batches > n_iter_min):
            # check if loss updated
            if l_curr < l_best:
                l_best = l_curr
                p_curr = 0
                params_best = params_curr
            else:
                if onp.isnan(l_curr):
                    # if diverged, return best
                    return params_best, (
                        predict_fun_repr,
                        predict_fun_head_po,
                        predict_fun_head_prop,
                    )
                p_curr = p_curr + 1

            if p_curr > patience:
                if return_val_loss:
                    # return loss without penalty
                    l_final = loss_snet2(params_curr, (X_val, y_val, w_val), 0, 0)
                    return (
                        params_curr,
                        (predict_fun_repr, predict_fun_head_po, predict_fun_head_prop),
                        l_final,
                    )

                return params_curr, (
                    predict_fun_repr,
                    predict_fun_head_po,
                    predict_fun_head_prop,
                )

    # return the parameters
    trained_params = get_params(opt_state)

    if return_val_loss:
        # return loss without penalty
        l_final = loss_snet2(get_params(opt_state), (X_val, y_val, w_val), 0, 0)
        return (
            trained_params,
            (predict_fun_repr, predict_fun_head_po, predict_fun_head_prop),
            l_final,
        )

    return trained_params, (
        predict_fun_repr,
        predict_fun_head_po,
        predict_fun_head_prop,
    )


def predict_snet2(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
    return_po: bool = False,
    return_prop: bool = False,
) -> jnp.ndarray:
    # unpack inputs
    predict_fun_repr, predict_fun_head, predict_fun_prop = predict_funs
    param_repr, param_0, param_1, param_prop = (
        trained_params[0],
        trained_params[1],
        trained_params[2],
        trained_params[3],
    )

    # get representation
    representation = predict_fun_repr(param_repr, X)

    # get potential outcomes
    mu_0 = predict_fun_head(param_0, representation)
    mu_1 = predict_fun_head(param_1, representation)

    te = mu_1 - mu_0
    if return_prop:
        # get propensity
        prop = predict_fun_prop(param_prop, representation)

    # stack other outputs
    if return_po:
        if return_prop:
            return te, mu_0, mu_1, prop
        else:
            return te, mu_0, mu_1
    else:
        if return_prop:
            return te, prop
        else:
            return te


def getrepr_snet2(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
) -> jnp.ndarray:
    # unpack inputs
    predict_fun_repr, predict_fun_head, predict_fun_prop = predict_funs
    param_repr, param_0, param_1, param_prop = (
        trained_params[0],
        trained_params[1],
        trained_params[2],
        trained_params[3],
    )

    # get representation
    representation = predict_fun_repr(param_repr, X)

    return representation


# Pairnet --------------------------------------------------------------------------------
def flatten_cfs(tensor):
    """Flatteens the middle dimension of a 3D tensor

    Args:
        tensor (_type_): _description_
    """
    if tensor.ndim == 2:
        return tensor
    assert tensor.ndim == 3 or tensor.ndim == 4, "Only 3D/4D tensors supported"
    return tensor.view(-1, tensor.shape[-1])


def torch_to_jnp(*args):
    """Converts torch tensors to jnp arrays.

    Returns:
        _type_: _description_
    """
    jnp_arr = [jnp.array(flatten_cfs(arg).numpy()) for arg in args]
    return (*jnp_arr,)  # unpack the tuple


class PairNet(BaseCATENet):
    """
    Class implements Shalit et al (2017)'s TARNet & CFR (discrepancy regularization is NOT
    TESTED). Also referred to as SNet-1 in our paper.

    Parameters
    ----------
    binary_y: bool, default False
        Whether the outcome is binary
    n_layers_out: int
        Number of hypothesis layers (n_layers_out x n_units_out + 1 x Dense layer)
    n_units_out: int
        Number of hidden units in each hypothesis layer
    n_layers_r: int
        Number of shared representation layers before hypothesis layers
    n_units_r: int
        Number of hidden units in each representation layer
    penalty_l2: float
        l2 (ridge) penalty
    step_size: float
        learning rate for optimizer
    n_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size
    val_split_prop: float
        Proportion of samples used for validation split (can be 0)
    early_stopping: bool, default True
        Whether to use early stopping
    patience: int
        Number of iterations to wait before early stopping after decrease in validation loss
    n_iter_min: int
        Minimum number of iterations to go through before starting early stopping
    n_iter_print: int
        Number of iterations after which to print updates
    seed: int
        Seed used
    reg_diff: bool, default False
        Whether to regularize the difference between the two potential outcome heads
    penalty_diff: float
        l2-penalty for regularizing the difference between output heads. used only if
        train_separate=False
    same_init: bool, False
        Whether to initialise the two output heads with same values
    nonlin: string, default 'elu'
        Nonlinearity to use in NN
    penalty_disc: float, default zero
        Discrepancy penalty. Defaults to zero as this feature is not tested.
    """

    def __init__(
        self,
        binary_y: bool = False,
        n_layers_r: int = DEFAULT_LAYERS_R,
        n_units_r: int = DEFAULT_UNITS_R,
        n_layers_out: int = DEFAULT_LAYERS_OUT,
        n_units_out: int = DEFAULT_UNITS_OUT,
        penalty_l2: float = DEFAULT_PENALTY_L2,
        step_size: float = DEFAULT_STEP_SIZE,
        n_iter: int = DEFAULT_N_ITER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        val_split_prop: float = DEFAULT_VAL_SPLIT,
        early_stopping: bool = True,
        patience: int = DEFAULT_PATIENCE,
        n_iter_min: int = DEFAULT_N_ITER_MIN,
        n_iter_print: int = DEFAULT_N_ITER_PRINT,
        seed: int = DEFAULT_SEED,
        reg_diff: bool = False,
        penalty_diff: float = DEFAULT_PENALTY_L2,
        same_init: bool = False,
        nonlin: str = DEFAULT_NONLIN,
        penalty_disc: float = DEFAULT_PENALTY_DISC,
        dynamic_phi: bool = False,
    ) -> None:
        # structure of net
        self.binary_y = binary_y
        self.n_layers_r = n_layers_r
        self.n_layers_out = n_layers_out
        self.n_units_r = n_units_r
        self.n_units_out = n_units_out
        self.nonlin = nonlin

        # penalties
        self.penalty_l2 = penalty_l2
        self.penalty_disc = penalty_disc
        self.reg_diff = reg_diff
        self.penalty_diff = penalty_diff
        self.same_init = same_init
        self.dynamic_phi = dynamic_phi

        # training params
        self.step_size = step_size
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.val_split_prop = val_split_prop
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter_min = n_iter_min

    def _get_train_function(self) -> Callable:
        return train_pairnet

    def _get_predict_function(self) -> Callable:
        return predict_pairnet

    def _get_repr_function(self) -> Callable:
        return getrepr_pairnet


def predict_pairnet(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
    return_po: bool = False,
    return_prop: bool = False,
) -> jnp.ndarray:
    if return_prop:
        raise NotImplementedError("SNet1 does not implement a propensity model.")
    params_phi, params_mu0, params_mu1 = trained_params
    pred_phi, pred_mu = predict_funs

    z = pred_phi(params_phi, X)
    mu0 = pred_mu(params_mu0, z)
    mu1 = pred_mu(params_mu1, z)
    cate = mu1 - mu0

    if return_po is False:
        return cate
    else:
        return cate, mu0, mu1


def getrepr_pairnet(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
) -> jnp.ndarray:
    params_phi, params_mu0, params_mu1 = trained_params
    pred_phi, pred_mu = predict_funs
    z = pred_phi(params_phi, X)
    return z
    

def train_pairnet(
    ads_trn: BaseTorchDataset,
    binary_y: bool = False,
    n_layers_r: int = DEFAULT_LAYERS_R,
    n_units_r: int = DEFAULT_UNITS_R,
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_units_out: int = DEFAULT_UNITS_OUT,
    penalty_l2: float = DEFAULT_PENALTY_L2,
    penalty_disc: int = DEFAULT_PENALTY_DISC,
    step_size: float = DEFAULT_STEP_SIZE,
    n_iter: int = DEFAULT_N_ITER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_split_prop: float = DEFAULT_VAL_SPLIT,
    early_stopping: bool = True,
    patience: int = DEFAULT_PATIENCE,
    n_iter_min: int = DEFAULT_N_ITER_MIN,
    n_iter_print: int = DEFAULT_N_ITER_PRINT,
    seed: int = DEFAULT_SEED,
    return_val_loss: bool = False,
    reg_diff: bool = False,
    same_init: bool = False,
    penalty_diff: float = DEFAULT_PENALTY_L2,
    nonlin: str = DEFAULT_NONLIN,
    avg_objective: bool = DEFAULT_AVG_OBJECTIVE,
    dynamic_phi: bool = False,
) -> Any:
    # This implements Proposal 1

    print(f"L2 Penalty: {penalty_l2}")
    
    xdim = ads_trn.X.shape[1]
    shape_x = (-1, xdim)
    shape_z = (-1, n_units_r)

    rng_key = random.PRNGKey(seed)
    onp.random.seed(seed)

    if not reg_diff:
        penalty_diff = penalty_l2

    ads_trn, ads_val, val_string = make_val_split_torch_DS(
        ads_train=ads_trn, val_split_prop=val_split_prop, seed=seed
    )

    num_trn = len(ads_trn)

    # get representation layer
    init_phi, pred_phi = ReprBlock(
        n_layers=n_layers_r, n_units=n_units_r, nonlin=nonlin
    )

    # Mu
    init_mu, pred_mu = OutputHead(
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        binary_y=binary_y,
        nonlin=nonlin,
    )

    def init_ftnbr_model(rng: float) -> Tuple[Tuple, List]:
        # Params must be [phi, mu0, mu1]
        rng, layer_rng = random.split(rng)
        _, param_phi = init_phi(layer_rng, shape_x)

        rng, layer_rng = random.split(rng)
        _, param_mu0 = init_mu(layer_rng, shape_z)

        rng, layer_rng = random.split(rng)
        out_shape, param_mu1 = init_mu(layer_rng, shape_z)

        return out_shape, [param_phi, param_mu0, param_mu1]

    if not binary_y:

        @jit
        def loss_agree(preds, gold, weights) -> jnp.ndarray:
            return jnp.sum(weights * ((preds - gold) ** 2))

    else:

        @jit
        def loss_binary_xent(preds, gold, weights) -> jnp.ndarray:
            return -jnp.sum(
                weights
                * (gold * jnp.log(preds) + (1 - gold) * jnp.log(1 - preds))
            )
        
        @jit
        # taken from https://github.com/google/flax/issues/2051#issue-1205532387
        def cross_entropy_loss(*, logits, labels, weights):
            labels = labels.astype(jnp.int32)
            logits = nn.log_softmax(logits)
            loss = -jax.vmap(getitem)(logits, labels)
            loss = jnp.sum(weights * loss)
            return loss

    # complete loss function for all parts
    @jit
    def loss_pairnet(
        params: List,
        batch: tuple,
        penalty_l2: float,
        penalty_disc: float,
        penalty_diff: float,
    ) -> jnp.ndarray:
        params_phi, params_mu0, params_mu1 = params
        x, b, y, xp, bp, yp = batch

        # get representation
        z = pred_phi(params_phi, x)
        zp = pred_phi(params_phi, xp)

        mu0z = pred_mu(params_mu0, z)
        mu0zp = pred_mu(params_mu0, zp)
        mu1z = pred_mu(params_mu1, z)
        mu1zp = pred_mu(params_mu1, zp)

        if not binary_y:
            
            gold_diff = (y - yp) * (2 * b - 1)
            pred_diff = ((mu1z - mu0zp) * b) + ((mu1zp - mu0z) * (1 - b))
            weights = jnp.ones_like(y)
            pairnet_loss = loss_agree(pred_diff, gold_diff, weights)
        else:
            gold_diff = y - yp 
            gold_diff = gold_diff + 1 # Change the labels to 0, 1, 2

            # Construct the three way logits

            # when y-yp = -1 => y=0, yp=1
            logits_y0: jnp.ndarray = ((mu0z * (1 - mu1zp)) * (1-b)) + ((mu1z * (1 - mu0zp)) * (1-b))
            # when y-yp = 0 => y=0, yp=0 or y=1, yp=1
            logits_y1: jnp.ndarray = ((mu0z * mu1zp) * (1-b)) + ((mu1z * mu0zp) * b) + (((1-mu0z) * (1-mu1zp)) * (1-b)) + (((1-mu1z) * (1-mu0zp)) * b)
            # when y-yp = 1 => y=1, yp=0
            logits_y2: jnp.ndarray = (((1-mu0z) * mu1zp) * (1-b)) + (((1-mu1z) * mu0zp) * b)
            
            logits = jnp.concatenate([logits_y0.reshape((-1, 1)), logits_y1.reshape((-1, 1)), logits_y2.reshape((-1, 1))], axis=1)
            diff_loss = cross_entropy_loss(logits=logits, labels=gold_diff, weights=jnp.ones_like(y))
            pairnet_loss = diff_loss

        # regularization on representation
        L2_phi = sum(
            [jnp.sum(params_phi[i][0] ** 2) for i in range(0, 2 * n_layers_r, 2)]
        )
        # L2_mu = heads_l2_penalty(
        #     params_mu0, params_mu1, n_layers_out, reg_diff, penalty_l2, penalty_diff
        # )
        L2_loss = penalty_l2 * L2_phi # our proposal does not involve a penalty on the \mu parameters

        if avg_objective:
            pairnet_loss = pairnet_loss / len(y)
        return pairnet_loss + L2_loss

    # Define optimisation routine
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    @jit
    def update(
        i: int, state: dict, batch: jnp.ndarray, penalty_l2: float, penalty_disc: float
    ) -> jnp.ndarray:
        params = get_params(state)
        return opt_update(
            i,
            grad(loss_pairnet)(params, batch, penalty_l2, penalty_disc, penalty_diff),
            state,
        )

    # initialise states
    _, init_params = init_ftnbr_model(rng_key)
    opt_state = opt_init(init_params)

    trn_loader = DataLoader(
        ads_trn, batch_size=batch_size, shuffle=True, collate_fn=dict_collate_fn
    )
    val_loader = DataLoader(
        ads_val, batch_size=len(ads_val), shuffle=False, collate_fn=dict_collate_fn
    )
    n_batches = int(onp.round(len(ads_trn) / batch_size)) if batch_size < num_trn else 1

    l_best = LARGE_VAL
    patience_curr = 0

    batch_dict_to_list = lambda batch: (
        batch["x"],
        batch["b"],
        batch["y"],
        batch["xp"],
        batch["bp"],
        batch["yp"],
    )
      
    # do training
    for i in range(n_iter):
        # shuffle data for minibatches
        
        # For dynamic_phi, we update the pairs derived out of distances obtained from the pairnet's phi function as it trains across epochs
        if dynamic_phi is True and i%5 == 0:
            all_X = ads_trn.dataset.X.cpu().numpy()
            emb = getrepr_pairnet(
                X=all_X, 
                trained_params=get_params(opt_state), 
                predict_funs=[pred_phi, pred_mu]
            )
            emb = onp.array(emb.copy())
            ads_trn.dataset._Xemb = emb

        for _, batch in enumerate(trn_loader):
            batch = batch_dict_to_list(batch)
            batch = torch_to_jnp(*batch)
            opt_state = update(
                i * n_batches + _, opt_state, batch, penalty_l2, penalty_disc
            )

        if (i % n_iter_print == 0) or early_stopping:
            params_curr = get_params(opt_state)
            for _, batch_val in enumerate(val_loader):
                batch_val = batch_dict_to_list(batch_val)
                batch_val = torch_to_jnp(*batch_val)
            l_curr = loss_pairnet(
                params_curr,
                batch_val,
                penalty_l2,
                penalty_disc,
                penalty_diff,
            )

        if i % n_iter_print == 0:
            log.info(f"Epoch: {i}, current {val_string} loss {l_curr}")

        if early_stopping:
            if l_curr < l_best:
                l_best = l_curr
                patience_curr = 0
                params_best = params_curr
            else:
                if onp.isnan(l_curr):
                    return params_best, (
                        pred_phi,
                        pred_mu,
                    )
                patience_curr = patience_curr + 1

            if patience_curr > patience and ((i + 1) * n_batches > n_iter_min):
                if return_val_loss:
                    l_final = loss_pairnet(params_curr, batch_val, 0, 0, 0)
                    return (
                        params_curr,
                        (pred_phi, pred_mu),
                        l_final,
                    )

                return params_curr, (pred_phi, pred_mu)

    # return the parameters
    trained_params = get_params(opt_state)

    if return_val_loss:
        # return loss without penalty
        l_final = loss_pairnet(get_params(opt_state), batch_val, 0, 0, 0)
        return (
            trained_params,
            (pred_phi, pred_mu),
            l_final,
        )

    return trained_params, (pred_phi, pred_mu)
