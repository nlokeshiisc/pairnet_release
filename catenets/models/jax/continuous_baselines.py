"""
Module implements TARNet and DRNet with binning of treatments
"""
# Author: Alicia Curth
from typing import Any, Callable, List, Tuple

import jax.numpy as jnp
import numpy as onp
from jax import grad, jit, random, vmap
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Elu, Relu, Sigmoid
from jax.nn.initializers import glorot_normal, normal
from jax.nn import elu
from scipy.spatial.distance import cdist

import torch
import wandb

import catenets.logger as log
from catenets.models.cont_constants import (
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
    DEFAULT_DROP_RATE,
    DEFAULT_X_DISTANCE_DELTA,
    DEFAULT_NUM_NEIGHBOURS,
    DEFAULT_SAMPLING_TEMPERATURE,
    DEFAULT_STATIC_PHI,
)
from catenets.models.jax.base import BaseCATENet, ReprBlock, OutputHead
from catenets.models.jax.model_utils import (
    check_shape_1d_data,
    make_val_split_with_indices,
    make_val_split_with_reps,
)

from catenets.models.jax.vc_net import (
    find_pairnet_indices,
    find_pairnet_indices,
    normalised_distances,
)

NUM_BINS = 5
GLOBAL_BINS = jnp.linspace(0, 1, NUM_BINS + 1)


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
        drop_rate: float = DEFAULT_DROP_RATE,
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
        self.drop_rate = drop_rate

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
        return train_tarnetc

    def _get_predict_function(self) -> Callable:
        return predict_tarnetc

    def _get_repr_function(self) -> Callable:
        return getrepr_tarnetc


class TARNetC(SNet1):
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


# Training functions for TARNet -------------------------------------------------


def predict_tarnetc(
    X,
    t,  # treatment
    trained_params: dict,
    predict_funs: list,
    return_po: bool = False,
    return_prop: bool = False,
) -> jnp.ndarray:
    if return_prop:
        raise NotImplementedError("VCNet does not implement a propensity model.")

    # unpack inputs
    predict_fun_repr, predict_fun_head = predict_funs
    param_repr = trained_params[0]
    params_heads = trained_params[1:]
    assert len(params_heads) == NUM_BINS

    # get representation
    representation = predict_fun_repr(param_repr, X)

    treatment_bins = jnp.digitize(t, GLOBAL_BINS) - 1  # digitize returns 1-index bins

    # get potential outcomes
    mu_t = 0
    for t_bin in range(NUM_BINS):
        mu_t += jnp.multiply(
            predict_fun_head(
                params_heads[t_bin], jnp.concatenate([representation, t], axis=1)
            ),
            (treatment_bins == t_bin),
        )

    if return_po:
        return mu_t, None, None
    else:
        return mu_t


def train_tarnetc(
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
    # function to train TARNET (Johansson et al) using jax
    # input check
    torch.manual_seed(seed)
    y, w = check_shape_1d_data(y), check_shape_1d_data(w)
    d = X.shape[1]
    input_shape = (-1, d)
    rng_key = random.PRNGKey(seed)
    onp.random.seed(seed)  # set seed for data generation via numpy as well

    if not reg_diff:
        penalty_diff = penalty_l2

    # get validation split (can be none)
    (
        X,
        y,
        w,
        indices,
        X_val,
        y_val,
        w_val,
        indices_val,
        val_string,
    ) = make_val_split_with_indices(
        X, y, w, val_split_prop=val_split_prop, seed=seed, stratify_w=False
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

    def init_fun_tarnetc(rng: float, input_shape: Tuple) -> Tuple[Tuple, List]:
        # chain together the layers
        rng, layer_rng = random.split(rng)
        input_shape_repr, param_repr = init_fun_repr(layer_rng, input_shape)
        param_t = [None for head_idx in range(NUM_BINS)]
        input_shape_repr = (
            input_shape_repr[0],
            input_shape_repr[1] + 1,
        )  # add 1 for concatenated treatment
        for head_idx in range(NUM_BINS):
            rng, layer_rng = random.split(rng)
            input_shape, param_t[head_idx] = init_fun_head(layer_rng, input_shape_repr)

        return input_shape, [param_repr, *param_t]

    # Define loss functions
    # loss functions for the head
    if not binary_y:

        def loss_head(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch
            treatment_bins = (
                jnp.digitize(weights, GLOBAL_BINS) - 1
            )  # digitize returns 1-index bins
            params_heads = params
            # get potential outcomes
            mu_t = 0
            for t_bin in range(NUM_BINS):
                mu_t += jnp.multiply(
                    predict_fun_head(
                        params_heads[t_bin], jnp.concatenate([inputs, weights], axis=1)
                    ),
                    (treatment_bins == t_bin),
                )
            return jnp.sum((mu_t - targets) ** 2)

    else:

        def loss_head(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch

            treatment_bins = (
                jnp.digitize(weights, GLOBAL_BINS) - 1
            )  # digitize returns 1-index bins
            params_heads = params
            # get potential outcomes
            mu_t = 0
            for t_bin in range(NUM_BINS):
                mu_t += jnp.multiply(
                    predict_fun_head(
                        params_heads[t_bin], jnp.concatenate([inputs, weights], axis=1)
                    ),
                    (treatment_bins == t_bin),
                )

            return -jnp.sum(
                (targets * jnp.log(mu_t) + (1 - targets) * jnp.log(1 - mu_t))
            )

    # complete loss function for all parts
    @jit
    def loss_tarnetc(
        params: List,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        penalty_l2: float,
        penalty_disc: float,
        penalty_diff: float,
    ) -> jnp.ndarray:
        # params: list[representation, head_0, head_1]
        # batch: (X, y, w)
        X, y, w = batch

        # get representation
        reps = predict_fun_repr(params[0], X)

        # pass down to VCNet output head
        loss_outputs = loss_head(params[1:], (reps, y, w))

        # regularization on representation
        weightsq_body = sum(
            [jnp.sum(params[0][i][0] ** 2) for i in range(0, 2 * n_layers_r, 2)]
        )
        weightsq_head = 0  # heads_l2_penalty(params[1], params[1], n_layers_out, reg_diff, penalty_l2, penalty_diff)
        if not avg_objective:
            return loss_outputs + (penalty_l2 * weightsq_body + weightsq_head)
        else:
            n_batch = y.shape[0]
            return (loss_outputs) / n_batch + (
                penalty_l2 * weightsq_body + weightsq_head
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
            grad(loss_tarnetc)(params, batch, penalty_l2, penalty_disc, penalty_diff),
            state,
        )

    # initialise states
    _, init_params = init_fun_tarnetc(rng_key, input_shape)
    opt_state = opt_init(init_params)

    # calculate number of batches per epoch
    batch_size = batch_size if batch_size < n else n
    n_batches = int(onp.round(n / batch_size)) if batch_size < n else 1
    train_indices = onp.arange(n)

    l_best = LARGE_VAL
    l_prev = LARGE_VAL
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
            l_curr = loss_tarnetc(
                params_curr,
                (X_val, y_val, w_val),
                penalty_l2,
                penalty_disc,
                penalty_diff,
            )
            wandb.log({"val_loss": l_curr})
        if i % n_iter_print == 0:
            log.info(f"Epoch: {i}, current {val_string} loss {l_curr}")
            print(f"Epoch: {i}, current {val_string} loss {l_curr}")

        if early_stopping:
            if l_curr < l_best:
                l_best = l_curr
                params_best = params_curr
            if l_curr < l_prev - 1e-2:
                p_curr = max(p_curr - 1, 0)

            else:
                if onp.isnan(l_curr):
                    # if diverged, return best
                    return (
                        params_best,
                        (predict_fun_repr, predict_fun_head),
                        (indices, indices_val),
                    )
                # p_curr = p_curr + 1

            if p_curr > patience and ((i + 1) * n_batches > n_iter_min):
                if return_val_loss:
                    # return loss without penalty
                    l_final = loss_tarnetc(params_curr, (X_val, y_val, w_val), 0, 0, 0)
                    return (
                        params_best,
                        (predict_fun_repr, predict_fun_head),
                        (indices, indices_val),
                        l_final,
                    )

                return (
                    params_best,
                    (predict_fun_repr, predict_fun_head),
                    (indices, indices_val),
                )
        l_prev = l_curr
    # return the parameters
    # trained_params = get_params(opt_state)

    if return_val_loss:
        # return loss without penalty
        l_final = loss_tarnetc(get_params(opt_state), (X_val, y_val, w_val), 0, 0, 0)
        return (
            params_best,
            (predict_fun_repr, predict_fun_head),
            (indices, indices_val),
            l_final,
        )

    return params_best, (predict_fun_repr, predict_fun_head), (indices, indices_val)


def getrepr_tarnetc(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
) -> jnp.ndarray:
    # unpack inputs
    predict_fun_repr, predict_fun_head = predict_funs
    param_repr, param_t = (
        trained_params[0],
        trained_params[1:],
    )

    # get representation
    representation = predict_fun_repr(param_repr, X)

    return representation


class SNet2(BaseCATENet):
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
        drop_rate: float = DEFAULT_DROP_RATE,
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
        self.drop_rate = drop_rate

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
        return train_drnetc

    def _get_predict_function(self) -> Callable:
        return predict_drnetc

    def _get_repr_function(self) -> Callable:
        return getrepr_drnetc


class DRNetC(SNet2):
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


# Training functions for DRNet -------------------------------------------------


def elementwise_w_treatment(fun, **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init_fun = lambda rng, input_shape: (input_shape, ())
    # inputs_treatments[0] = inputs
    # inputs_treatments[1] = treatments
    apply_fun = lambda params, inputs_treatments, **kwargs: (
        fun(inputs_treatments[0], **fun_kwargs),
        inputs_treatments[1],
    )
    return init_fun, apply_fun


Elu = elementwise_w_treatment(elu)


def Dense(out_dim, W_init=glorot_normal(), b_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)

        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1] + 1, out_dim)), b_init(
            k2,
            (out_dim,),
        )
        return output_shape, (W, b)

    def apply_fun(params, input_treatments, **kwargs):
        X, treatments = input_treatments  # X : N X L
        W, b = params
        t = treatments.reshape((-1, 1))  # N X 1
        X_t = jnp.concatenate([X, t], axis=1)  # N X (L + 1)

        return (X_t @ W + b, treatments)  # N X R

    return init_fun, apply_fun


def OutputHeadDRNet(
    n_layers_out: int = DEFAULT_LAYERS_OUT,
    n_units_out: int = DEFAULT_UNITS_OUT,
    binary_y: bool = False,
    n_layers_r: int = 0,
    n_units_r: int = DEFAULT_UNITS_R,
    nonlin: str = DEFAULT_NONLIN,
) -> Any:
    # Creates an output head using jax.stax
    if nonlin == "elu":
        NL = Elu
    elif nonlin == "relu":
        NL = Relu
    elif nonlin == "sigmoid":
        NL = Sigmoid
    else:
        raise ValueError("Unknown nonlinearity")

    layers: Tuple = ()

    # add required number of layers
    for i in range(n_layers_r):
        layers = (*layers, Dense(n_units_r), NL)

    # add required number of layers
    for i in range(n_layers_out):
        layers = (*layers, Dense(n_units_out), NL)

    # return final architecture
    if not binary_y:
        return stax.serial(*layers, Dense(1))
    else:
        return stax.serial(*layers, Dense(1), Sigmoid)


def predict_drnetc(
    X,
    t,  # treatment
    trained_params: dict,
    predict_funs: list,
    return_po: bool = False,
    return_prop: bool = False,
) -> jnp.ndarray:
    if return_prop:
        raise NotImplementedError("VCNet does not implement a propensity model.")

    # unpack inputs
    predict_fun_repr, predict_fun_head = predict_funs
    param_repr = trained_params[0]
    params_heads = trained_params[1:]
    assert len(params_heads) == NUM_BINS

    # get representation
    representation = predict_fun_repr(param_repr, X)

    treatment_bins = jnp.digitize(t, GLOBAL_BINS) - 1  # digitize returns 1-index bins

    # get potential outcomes
    mu_t = 0
    for t_bin in range(NUM_BINS):
        mu_t_bin, _ = predict_fun_head(
            params_heads[t_bin], (representation, t - GLOBAL_BINS[t_bin])
        )
        mu_t += jnp.multiply(mu_t_bin, (treatment_bins == t_bin))

    if return_po:
        return mu_t, None, None
    else:
        return mu_t


def train_drnetc(
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
    # function to train TARNET (Johansson et al) using jax
    # input check
    torch.manual_seed(seed)
    y, w = check_shape_1d_data(y), check_shape_1d_data(w)
    d = X.shape[1]
    input_shape = (-1, d)
    rng_key = random.PRNGKey(seed)
    onp.random.seed(seed)  # set seed for data generation via numpy as well

    if not reg_diff:
        penalty_diff = penalty_l2

    # get validation split (can be none)
    (
        X,
        y,
        w,
        indices,
        X_val,
        y_val,
        w_val,
        indices_val,
        val_string,
    ) = make_val_split_with_indices(
        X, y, w, val_split_prop=val_split_prop, seed=seed, stratify_w=False
    )
    n = X.shape[0]  # could be different from before due to split

    # get representation layer
    init_fun_repr, predict_fun_repr = ReprBlock(
        n_layers=n_layers_r, n_units=n_units_r, nonlin=nonlin
    )

    # get output head functions (both heads share same structure)
    init_fun_head, predict_fun_head = OutputHeadDRNet(
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        binary_y=binary_y,
        nonlin=nonlin,
    )

    def init_fun_drnet(rng: float, input_shape: Tuple) -> Tuple[Tuple, List]:
        # chain together the layers
        rng, layer_rng = random.split(rng)
        input_shape_repr, param_repr = init_fun_repr(layer_rng, input_shape)
        param_t = [None for _ in range(NUM_BINS)]
        for head_idx in range(NUM_BINS):
            rng, layer_rng = random.split(rng)
            input_shape, param_t[head_idx] = init_fun_head(layer_rng, input_shape_repr)

        return input_shape, [param_repr, *param_t]

    # Define loss functions
    # loss functions for the head
    if not binary_y:

        def loss_head(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch
            treatment_bins = (
                jnp.digitize(weights, GLOBAL_BINS) - 1
            )  # digitize returns 1-index bins
            params_heads = params

            # get potential outcomes
            mu_t = 0
            for t_bin in range(NUM_BINS):
                mu_t_bin, _ = predict_fun_head(
                    params_heads[t_bin], (inputs, weights - GLOBAL_BINS[t_bin])
                )
                mu_t += jnp.multiply(mu_t_bin, (treatment_bins == t_bin))
            return jnp.sum((mu_t - targets) ** 2)

    else:

        def loss_head(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch

            treatment_bins = (
                jnp.digitize(weights, GLOBAL_BINS) - 1
            )  # digitize returns 1-index bins
            params_heads = params
            # get potential outcomes
            mu_t = 0
            for t_bin in range(NUM_BINS):
                mu_t_bin = predict_fun_head(
                    params_heads[t_bin], (inputs, weights - GLOBAL_BINS[t_bin])
                )
                mu_t += jnp.multiply(mu_t_bin, (treatment_bins == t_bin))

            return -jnp.sum(
                (targets * jnp.log(mu_t) + (1 - targets) * jnp.log(1 - mu_t))
            )

    # complete loss function for all parts
    @jit
    def loss_drnet(
        params: List,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        penalty_l2: float,
        penalty_disc: float,
        penalty_diff: float,
    ) -> jnp.ndarray:
        # params: list[representation, head_0, head_1]
        # batch: (X, y, w)
        X, y, w = batch

        # get representation
        reps = predict_fun_repr(params[0], X)

        # pass down to VCNet output head
        # mse loss function

        loss_outputs = loss_head(params[1:], (reps, y, w))

        # regularization on representation
        weightsq_body = sum(
            [jnp.sum(params[0][i][0] ** 2) for i in range(0, 2 * n_layers_r, 2)]
        )
        weightsq_head = 0  # heads_l2_penalty(params[1], params[1], n_layers_out, reg_diff, penalty_l2, penalty_diff)
        if not avg_objective:
            return loss_outputs + (penalty_l2 * weightsq_body + weightsq_head)
        else:
            n_batch = y.shape[0]
            return (loss_outputs) / n_batch + (
                penalty_l2 * weightsq_body + weightsq_head
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
            grad(loss_drnet)(params, batch, penalty_l2, penalty_disc, penalty_diff),
            state,
        )

    # initialise states
    _, init_params = init_fun_drnet(rng_key, input_shape)
    opt_state = opt_init(init_params)

    # calculate number of batches per epoch
    batch_size = batch_size if batch_size < n else n
    n_batches = int(onp.round(n / batch_size)) if batch_size < n else 1
    train_indices = onp.arange(n)

    p_curr = 0
    l_best = LARGE_VAL
    l_prev = LARGE_VAL

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
            l_curr = loss_drnet(
                params_curr,
                (X_val, y_val, w_val),
                penalty_l2,
                penalty_disc,
                penalty_diff,
            )
            wandb.log({"val_loss": l_curr})
        if i % n_iter_print == 0:
            log.info(f"Epoch: {i}, current {val_string} loss {l_curr}")
            print(f"Epoch: {i}, current {val_string} loss {l_curr}")

        if early_stopping:
            if l_curr < l_best:
                l_best = l_curr
                params_best = params_curr
                p_curr = 0
            # if l_curr < l_prev - 1e-2:
            #     p_curr = max(p_curr - 1, 0)

            else:
                if onp.isnan(l_curr):
                    # if diverged, return best
                    return (
                        params_best,
                        (predict_fun_repr, predict_fun_head),
                        (indices, indices_val),
                    )
                # p_curr = p_curr + 1

            if p_curr > patience and ((i + 1) * n_batches > n_iter_min):
                if return_val_loss:
                    # return loss without penalty
                    l_final = loss_drnet(params_curr, (X_val, y_val, w_val), 0, 0, 0)
                    return (
                        params_best,
                        (predict_fun_repr, predict_fun_head),
                        (indices, indices_val),
                        l_final,
                    )

                return (
                    params_best,
                    (predict_fun_repr, predict_fun_head),
                    (indices, indices_val),
                )
        l_prev = l_curr
    # return the parameters
    trained_params = get_params(opt_state)

    if return_val_loss:
        # return loss without penalty
        l_final = loss_drnet(get_params(opt_state), (X_val, y_val, w_val), 0, 0, 0)
        return (
            params_best,
            (predict_fun_repr, predict_fun_head),
            (indices, indices_val),
            l_final,
        )

    return params_best, (predict_fun_repr, predict_fun_head), (indices, indices_val)


def getrepr_drnetc(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
) -> jnp.ndarray:
    # unpack inputs
    predict_fun_repr, predict_fun_head = predict_funs
    param_repr, param_t = (
        trained_params[0],
        trained_params[1:],
    )

    # get representation
    representation = predict_fun_repr(param_repr, X)

    return representation


"""
DRNetPairNet starts here
"""


class SNet3(BaseCATENet):
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
        drop_rate: float = DEFAULT_DROP_RATE,
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
        self.drop_rate = drop_rate

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
        return train_drnetpairnet

    def _get_predict_function(self) -> Callable:
        return predict_drnetpairnet

    def _get_repr_function(self) -> Callable:
        return getrepr_drnetpairnet


class DRNetPairNet(SNet3):
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


# Training functions for DRNet -------------------------------------------------


def predict_drnetpairnet(
    X,
    t,  # treatment
    trained_params: dict,
    predict_funs: list,
    return_po: bool = False,
    return_prop: bool = False,
) -> jnp.ndarray:
    if return_prop:
        raise NotImplementedError("VCNet does not implement a propensity model.")

    # unpack inputs
    predict_fun_repr, predict_fun_head = predict_funs
    param_repr = trained_params[0]
    params_heads = trained_params[1:]
    assert len(params_heads) == NUM_BINS

    # get representation
    representation = predict_fun_repr(param_repr, X)

    treatment_bins = jnp.digitize(t, GLOBAL_BINS) - 1  # digitize returns 1-index bins

    # get potential outcomes
    mu_t = 0
    for t_bin in range(NUM_BINS):
        mu_t_bin, _ = predict_fun_head(
            params_heads[t_bin], (representation, t - GLOBAL_BINS[t_bin])
        )
        mu_t += jnp.multiply(mu_t_bin, (treatment_bins == t_bin))

    if return_po:
        return mu_t, None, None
    else:
        return mu_t


def train_drnetpairnet(
    X: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    representation: jnp.ndarray,
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
    delta=DEFAULT_X_DISTANCE_DELTA,
    num_cfz=DEFAULT_NUM_NEIGHBOURS,
    sampling_temp=DEFAULT_SAMPLING_TEMPERATURE,
) -> Any:
    # function to train TARNET (Johansson et al) using jax
    # input check
    torch.manual_seed(seed)
    y, w = check_shape_1d_data(y), check_shape_1d_data(w)
    d = X.shape[1]
    input_shape = (-1, d)
    rng_key = random.PRNGKey(seed)
    onp.random.seed(seed)  # set seed for data generation via numpy as well

    if not reg_diff:
        penalty_diff = penalty_l2

    # get validation split (can be none)
    (
        X,
        y,
        w,
        reps,
        indices,
        X_val,
        y_val,
        w_val,
        reps_val,
        indices_val,
        val_string,
    ) = make_val_split_with_reps(
        X,
        representation,
        y,
        w,
        val_split_prop=val_split_prop,
        seed=seed,
        stratify_w=False,
    )
    n = X.shape[0]  # could be different from before due to split

    # get representation layer
    init_fun_repr, predict_fun_repr = ReprBlock(
        n_layers=n_layers_r, n_units=n_units_r, nonlin=nonlin
    )

    # get output head functions (both heads share same structure)
    init_fun_head, predict_fun_head = OutputHeadDRNet(
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        binary_y=binary_y,
        nonlin=nonlin,
    )

    def init_fun_drnet(rng: float, input_shape: Tuple) -> Tuple[Tuple, List]:
        # chain together the layers
        rng, layer_rng = random.split(rng)
        input_shape_repr, param_repr = init_fun_repr(layer_rng, input_shape)
        param_t = [None for _ in range(NUM_BINS)]
        for head_idx in range(NUM_BINS):
            rng, layer_rng = random.split(rng)
            input_shape, param_t[head_idx] = init_fun_head(layer_rng, input_shape_repr)

        return input_shape, [param_repr, *param_t]

    # Define loss functions
    # loss functions for the head

    def loss_head(
        params: List,
        batch: Tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ) -> jnp.ndarray:
        # mse loss function
        inputs, targets, weights, inputs_cf, targets_cf, weights_cf = batch
        treatment_bins = jnp.digitize(weights, GLOBAL_BINS) - 1
        treatment_bins_cf = (
            jnp.digitize(weights_cf, GLOBAL_BINS) - 1
        )  # digitize returns 1-index bins
        params_heads = params

        # get potential outcomes
        mu_t = 0
        for t_bin in range(NUM_BINS):
            mu_t_bin, _ = predict_fun_head(
                params_heads[t_bin], (inputs, weights - GLOBAL_BINS[t_bin])
            )
            mu_t += jnp.multiply(mu_t_bin, (treatment_bins == t_bin))
        # get potential outcomes
        mu_t_cf = 0
        for t_bin in range(NUM_BINS):
            mu_t_bin_cf, _ = predict_fun_head(
                params_heads[t_bin], (inputs_cf, weights_cf - GLOBAL_BINS[t_bin])
            )
            mu_t_cf += jnp.multiply(mu_t_bin_cf, (treatment_bins_cf == t_bin))
        return jnp.sum(((mu_t - mu_t_cf) - (targets - targets_cf)) ** 2)

    # complete loss function for all parts
    @jit
    def loss_drnet(
        params: List,
        batch: tuple,
        penalty_l2: float,
        penalty_disc: float,
        penalty_diff: float,
        factual=False,
    ) -> jnp.ndarray:
        X, y, w, X_cf, y_cf, w_cf = batch

        # get representation
        reps = predict_fun_repr(params[0], X)
        reps_cf = predict_fun_repr(params[0], X_cf)

        # pass down to VCNet output head
        loss_outputs = loss_head(params[1:], (reps, y, w, reps_cf, y_cf, w_cf))

        # regularization on representation
        weightsq_body = sum(
            [jnp.sum(params[0][i][0] ** 2) for i in range(0, 2 * n_layers_r, 2)]
        )
        weightsq_head = 0  # heads_l2_penalty(params[1], params[1], n_layers_out, reg_diff, penalty_l2, penalty_diff)
        if not avg_objective:
            return loss_outputs + (penalty_l2 * weightsq_body + weightsq_head)
        else:
            n_batch = y.shape[0]
            return (loss_outputs) / n_batch + (
                penalty_l2 * weightsq_body + weightsq_head
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
            grad(loss_drnet)(params, batch, penalty_l2, penalty_disc, penalty_diff),
            state,
        )

    # initialise states
    _, init_params = init_fun_drnet(rng_key, input_shape)
    opt_state = opt_init(init_params)

    # calculate number of batches per epoch
    batch_size = batch_size if batch_size < n else n
    n_batches = int(onp.round(n / batch_size)) if batch_size < n else 1
    train_indices = onp.arange(n)

    l_best = LARGE_VAL
    l_prev = LARGE_VAL
    p_curr = 0

    # Calculate distances
    reps_distances, reps_val_distances = normalised_distances(reps, reps_val, delta)

    # do training

    for i in range(n_iter):
        # shuffle data for minibatches
        onp.random.shuffle(train_indices)
        for b in range(n_batches):
            idx_next = train_indices[
                (b * batch_size) : min((b + 1) * batch_size, n - 1)
            ]
            params_curr = get_params(opt_state)
            X_batch = X[idx_next].repeat(num_cfz, axis=0)
            y_batch = y[idx_next].repeat(num_cfz, axis=0)
            w_batch = w[idx_next].repeat(num_cfz, axis=0)

            cf_idx = find_pairnet_indices(
                reps_distances, y, w, num_cfz, sampling_temp, idx_next
            ).reshape(-1)
            next_batch = (
                X_batch,
                y_batch,
                w_batch,
                X[cf_idx],
                y[cf_idx],
                w[cf_idx],
            )
            opt_state = update(
                i * n_batches + b, opt_state, next_batch, penalty_l2, penalty_disc
            )
        """"""
        # ABLATION 1: using current representations for neighbour distance computation instead of frozen representations
        if not DEFAULT_STATIC_PHI:
            reps = predict_fun_repr(params_curr[0], X)
            reps_val = predict_fun_repr(params_curr[0], X_val)
            reps_distances, reps_val_distances = normalised_distances(
                reps, reps_val, delta
            )
        """"""

        if (i % n_iter_print == 0) or early_stopping:
            params_curr = get_params(opt_state)
            X_batch = X_val.repeat(num_cfz, axis=0)
            y_batch = y_val.repeat(num_cfz, axis=0)
            w_batch = w_val.repeat(num_cfz, axis=0)
            cf_idx = find_pairnet_indices(
                reps_val_distances, y_val, w_val, num_cfz, sampling_temp
            ).reshape(-1)
            next_batch = (
                X_batch,
                y_batch,
                w_batch,
                X[cf_idx],
                y[cf_idx],
                w[cf_idx],
            )
            l_curr = loss_drnet(
                params_curr,
                next_batch,
                penalty_l2,
                penalty_disc,
                penalty_diff,
            )
            wandb.log({"val_loss": l_curr})
        if i % n_iter_print == 0:
            log.info(f"Epoch: {i}, current {val_string} loss {l_curr}")
            print(f"Epoch: {i}, current {val_string} loss {l_curr}")

        if early_stopping:
            if l_curr < l_best:
                l_best = l_curr
                params_best = params_curr
                p_curr = 0
            # if l_curr < l_prev - 1e-2:
            #     p_curr = max(p_curr - 1, 0)
            else:
                if onp.isnan(l_curr):
                    # if diverged, return best
                    return (
                        params_best,
                        (predict_fun_repr, predict_fun_head),
                        (indices, indices_val),
                    )
                # p_curr = p_curr + 1

            if p_curr > patience and ((i + 1) * n_batches > n_iter_min):
                if return_val_loss:
                    # return loss without penalty
                    params_curr = get_params(opt_state)
                    X_batch = X_val.repeat(num_cfz, axis=0)
                    y_batch = y_val.repeat(num_cfz, axis=0)
                    w_batch = w_val.repeat(num_cfz, axis=0)
                    cf_idx = find_pairnet_indices(
                        reps_val_distances, y_val, w_val, num_cfz, sampling_temp
                    ).reshape(-1)
                    next_batch = (
                        X_batch,
                        y_batch,
                        w_batch,
                        X[cf_idx],
                        y[cf_idx],
                        w[cf_idx],
                    )
                    l_final = loss_drnet(params_curr, next_batch, 0, 0, 0)
                    return (
                        params_best,
                        (predict_fun_repr, predict_fun_head),
                        (indices, indices_val),
                        l_final,
                    )

                return (
                    params_best,
                    (predict_fun_repr, predict_fun_head),
                    (indices, indices_val),
                )
        l_prev = l_curr
    # return the parameters
    trained_params = get_params(opt_state)

    if return_val_loss:
        # return loss without penalty
        params_curr = get_params(opt_state)
        X_batch = X_val.repeat(num_cfz, axis=0)
        y_batch = y_val.repeat(num_cfz, axis=0)
        w_batch = w_val.repeat(num_cfz, axis=0)
        cf_idx = find_pairnet_indices(
            reps_val_distances, y_val, w_val, num_cfz, sampling_temp
        ).reshape(-1)
        next_batch = (
            X_batch,
            y_batch,
            w_batch,
            X[cf_idx],
            y[cf_idx],
            w[cf_idx],
        )
        l_final = loss_drnet(get_params(opt_state), next_batch, 0, 0, 0)
        return (
            params_best,
            (predict_fun_repr, predict_fun_head),
            (indices, indices_val),
            l_final,
        )

    return params_best, (predict_fun_repr, predict_fun_head), (indices, indices_val)


def getrepr_drnetpairnet(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
) -> jnp.ndarray:
    # unpack inputs
    predict_fun_repr, predict_fun_head = predict_funs
    param_repr, param_t = (
        trained_params[0],
        trained_params[1:],
    )

    # get representation
    representation = predict_fun_repr(param_repr, X)

    return representation
