"""
Module implements VCNet
"""
# Author: Alicia Curth
from typing import Any, Callable, List, Tuple

import jax.numpy as jnp
import numpy as onp
from jax import grad, jit, random, vmap
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Relu, Sigmoid
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
from catenets.models.jax.base import BaseCATENet, ReprBlock
from catenets.models.jax.model_utils import (
    check_shape_1d_data,
    make_val_split_with_reps,
    make_val_split_with_indices,
)


def find_nnT_tgts(*, cf_minus_trn_dosages, delta_dosage, num_cfz, **kwargs):
    """
    If method if GI, we just need the target labels and so we return the y^cf directly
    If method is GP, then we have to return ths ids of data samples that fall withing a
    delta window of the counterfactual \beta.
    Note that number of samples in each window may be totally different.

    Returns:
        _type_: _description_
    """
    candids = [torch.where(row < delta_dosage)[0] for row in cf_minus_trn_dosages]

    bad_ids = [id for id, entry in enumerate(candids) if len(entry) <= num_cfz]
    for bid in bad_ids:
        for factor in range(2, 10):
            candids[bid] = torch.where(
                cf_minus_trn_dosages[bid] < delta_dosage * factor
            )[0]
            if len(candids[bid]) > num_cfz:
                break

    assert len(candids) == cf_minus_trn_dosages.shape[0]

    return candids


def sample_far_gp_t(*, batch_dosages, delta_dosage, num_samples=1, **kwargs):
    """This code samples dosages that are atleast \delta away from the factual dosage

    This code also implements other sampling distributions that we did in the ICTE paper for ablations

    Args:
        dosage (_type_): _description_
        linear_delta (_type_): _description_
        num_samples (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """

    def clamp(entry):
        return min(max(0, entry), 1)

    sampled_dosages = []
    for dose_idx, fct_dose in enumerate(batch_dosages):
        # Reject samples that are delta close to the factual dosage
        rej_wdow = [clamp(fct_dose - delta_dosage), clamp(fct_dose + delta_dosage)]

        gap = (rej_wdow[1] - rej_wdow[0]).item()
        delta_samples = torch.FloatTensor(1, num_samples).uniform_(
            0, 1 - gap
        )  # removed .cuda(
        delta_samples[delta_samples > rej_wdow[0]] += gap

        sampled_dosages.append(delta_samples)
    return torch.cat(sampled_dosages, dim=0).view(-1, num_samples)


def find_nbr_ids(
    *,
    trn_embs_distances,
    trn_dosages,
    cf_dosages=None,
    batch_ids=None,
    num_cfz=1,
    sample_far=False,
    sampling_temp=1,
    **kwargs,
):
    """Finds the nearest neighbor IDs for the factual samples at uniformly sampled \beta^CF
    First fiters the dataste w.r.t. the dosages
    Then finds the nearest neighbors w.r.t. the embeddings

    Args:
        trn_embs (_type_): embeddings of all points in the training dataset
        trn_dosages (_type_): all the train dosages
        dosage_cf: sampled new dosages in the current batch
        trn_ys (_type_): all train y
        batch_ids (_type_): current batch ids
        num_cfz : number of counterfactual t' neighbours sampled
        sample_far : (bool) samples far away counterfactual treatments t' for every t
    """
    if batch_ids is None:
        batch_ids = torch.arange(len(trn_dosages))
    if cf_dosages is None:
        if sample_far:
            epsilon = 0.3
            cf_dosages = sample_far_gp_t(
                batch_dosages=trn_dosages[batch_ids], delta_dosage=epsilon
            )
        else:
            cf_dosages = torch.rand(len(batch_ids))

    assert len(cf_dosages) == len(batch_ids), "Batch size mismatch"

    trn_embs_distances[torch.eye(trn_embs_distances.shape[0]).to(bool)] = float("inf")
    cf_dosages_col = onp.array(cf_dosages.view(-1, 1))
    trn_dosgaes_row = onp.array(trn_dosages.view(1, -1))
    cf_minus_trn_dosages = jnp.where(
        trn_embs_distances[batch_ids, :] == float("inf"),
        float("inf"),
        jnp.abs(cf_dosages_col - trn_dosgaes_row),
    )

    # First find the nearest neighbors w.r.t. the ctr treatments
    nbr_ids = []
    nnd_dosage_filtered = find_nnT_tgts(
        cf_minus_trn_dosages=torch.Tensor(onp.array(cf_minus_trn_dosages)),
        delta_dosage=0.1,
        num_cfz=num_cfz,
        **kwargs,
    )
    nnd_dosage_unfiltered = find_nnT_tgts(
        cf_minus_trn_dosages=torch.abs(torch.Tensor(cf_dosages_col - trn_dosgaes_row)),
        delta_dosage=0.1,
        num_cfz=num_cfz,
        **kwargs,
    )

    for curr_id, nnd_dosage_ids, nnd_dosage_ids_unfiltered in zip(
        batch_ids, nnd_dosage_filtered, nnd_dosage_unfiltered
    ):
        # Remove self from the nnd_dosage_ids to suppress self selection
        # nnd_dosage_ids = nnd_dosage_ids[nnd_dosage_ids != curr_id]
        if len(nnd_dosage_ids) == 0:
            nnd_dosage_ids = nnd_dosage_ids_unfiltered[
                nnd_dosage_ids_unfiltered != curr_id
            ]
            nbr_distances = trn_embs_distances[curr_id, nnd_dosage_ids]
            replace = False

            nbr_dose_emb = torch.LongTensor(
                onp.random.choice(
                    onp.arange(len(nnd_dosage_ids)),
                    num_cfz,
                    replace=replace,
                )
            )
        else:
            nbr_distances = trn_embs_distances[curr_id, nnd_dosage_ids]
            if len(nbr_distances) == 1:
                sampling_probs = torch.softmax(
                    -sampling_temp * torch.Tensor([nbr_distances]), dim=0
                )
            else:
                sampling_probs = torch.softmax(
                    -sampling_temp * torch.Tensor(nbr_distances), dim=0
                )
            if num_cfz > torch.count_nonzero(sampling_probs):
                replace = True
            else:
                replace = False

            nbr_dose_emb = torch.LongTensor(
                onp.random.choice(
                    onp.arange(len(nnd_dosage_ids)),
                    num_cfz,
                    p=sampling_probs.cpu().numpy(),  # for uniform sampling set sampling_temp = 0
                    replace=replace,
                )
            )  # removed .cuda(
        nbr_dose_emb = nnd_dosage_ids[nbr_dose_emb]
        nbr_ids.append(nbr_dose_emb)

    nbr_ids = torch.stack(nbr_ids).view(-1, num_cfz)
    assert len(nbr_ids) == len(batch_ids), "Batch size mismatch"
    return nbr_ids.cpu().numpy()


def find_pairnet_indices(
    reps_distances, y, w, num_cfz=3, sampling_temp=1, indices=None
):
    """
    return the indices 'cf_idx' of X[indices, :] from the set of all X
    in the representation space
    """
    return find_nbr_ids(
        trn_embs_distances=reps_distances,
        trn_dosages=torch.Tensor(onp.array(w)),
        trn_ys=torch.Tensor(onp.array(y)),
        batch_ids=indices,
        num_cfz=num_cfz,
        sampling_temp=sampling_temp,
    )


def set_k_largest_inf(matrix, delta):
    if delta == 0:
        return matrix
    N = matrix.shape[0]
    matrix_vec = matrix.reshape(-1)
    n = matrix_vec.shape[0]
    k = int(jnp.ceil(delta * n))
    indices = jnp.argpartition(matrix_vec, -k)[-k:]
    matrix_vec[indices] = float("inf")
    matrix = matrix_vec.reshape((N, -1))
    return matrix


def normalised_distances(reps, reps_val, delta=0.5):
    reps_norms = onp.linalg.norm(reps, axis=1, keepdims=True)
    reps_val_norms = onp.linalg.norm(reps_val, axis=1, keepdims=True)

    # Calculate distances
    reps_distances = cdist(reps, reps) / reps_norms
    reps_val_distances = cdist(reps_val, reps_val) / reps_val_norms

    reps_distances = set_k_largest_inf(reps_distances, delta)
    reps_val_distances = set_k_largest_inf(reps_val_distances, delta)

    return reps_distances, reps_val_distances


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


def spline(t, spline_knots, spline_degree, num_of_basis):
    # input: t, vector of size N
    # output: out, vector of size num_of_basis X N = 12 x N

    def compute_basis(_, t0, spline_knots, spline_degree):
        boolarr = _ <= spline_degree
        t = t0  # can transform t0 here to adjust range to fit (0,1)

        if spline_degree == 1:
            return boolarr * (t**_) + (1 - boolarr) * jnp.maximum(
                t - spline_knots[_ - spline_degree], 0
            )
        else:
            return boolarr * (t**_) + (1 - boolarr) * (
                jnp.maximum(t - spline_knots[_ - spline_degree - 1], 0) ** spline_degree
            )

    out = vmap(compute_basis, in_axes=(0, None, None, None))(
        jnp.arange(num_of_basis), t, spline_knots, spline_degree
    )
    return out


def Dense(out_dim, W_init=glorot_normal(), b_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer."""
    # spline_knots = jnp.arange(0.1, 1, 0.1)
    spline_knots = jnp.array([1 / 3, 2 / 3])
    spline_degree = 2
    num_of_basis = spline_degree + 1 + len(spline_knots)

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)

        k1, k2 = random.split(rng)
        WA, bA = W_init(k1, (input_shape[-1], out_dim, num_of_basis)), b_init(
            k2,
            (
                out_dim,
                num_of_basis,
            ),
        )
        return output_shape, (WA, bA)

    def apply_fun(params, input_treatments, **kwargs):
        inputs, treatments = input_treatments
        WA, bA = params
        t = treatments.squeeze()  # N
        # inputs : N x L
        # T = 12
        # WA: L x R X T
        # bA: R x T
        spl = spline(t, spline_knots, spline_degree, num_of_basis)  # T x N
        b = jnp.dot(bA, spl).T  # N X R
        W = jnp.dot(WA, spl)  # L x R X N

        def piece_mult(x_0, W_0):
            # x_0 : L
            # W_0 : L x R
            return jnp.dot(x_0, W_0)

        result = vmap(piece_mult, in_axes=(0, 2))(inputs, W)  # N x R
        return (result + b, treatments)  # N X R

    return init_fun, apply_fun


def OutputHead(
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
        return train_vcnet

    def _get_predict_function(self) -> Callable:
        return predict_vcnet

    def _get_repr_function(self) -> Callable:
        return getrepr_vcnet


class VCNet(SNet1):
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


# Training functions for VCNet -------------------------------------------------


def predict_vcnet(
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
    param_repr, param_t = (
        trained_params[0],
        trained_params[1],
    )

    # get representation
    representation = predict_fun_repr(param_repr, X)

    # get potential outcomes
    mu_t, _ = predict_fun_head(param_t, (representation, t))

    if return_po:
        return mu_t, _, _
    else:
        return mu_t


def train_vcnet(
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

    def init_fun_vcnet(rng: float, input_shape: Tuple) -> Tuple[Tuple, List]:
        # chain together the layers
        rng, layer_rng = random.split(rng)
        input_shape_repr, param_repr = init_fun_repr(layer_rng, input_shape)
        rng, layer_rng = random.split(rng)
        input_shape, param_t = init_fun_head(layer_rng, input_shape_repr)

        return input_shape, [param_repr, param_t]

    # Define loss functions
    # loss functions for the head
    if not binary_y:

        def loss_head(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch
            preds, _ = predict_fun_head(params, (inputs, weights))
            return jnp.sum((preds - targets) ** 2)

    else:

        def loss_head(
            params: List, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> jnp.ndarray:
            # mse loss function
            inputs, targets, weights = batch
            preds, _ = predict_fun_head(params, (inputs, weights))
            return -jnp.sum(
                (targets * jnp.log(preds) + (1 - targets) * jnp.log(1 - preds))
            )

    # complete loss function for all parts
    @jit
    def loss_vcnet(
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
        loss_1 = loss_head(params[1], (reps, y, w))

        # regularization on representation
        weightsq_body = sum(
            [jnp.sum(params[0][i][0] ** 2) for i in range(0, 2 * n_layers_r, 2)]
        )
        weightsq_head = 0  # heads_l2_penalty(params[1], params[1], n_layers_out, reg_diff, penalty_l2, penalty_diff)
        if not avg_objective:
            return loss_1 + (penalty_l2 * weightsq_body + weightsq_head)
        else:
            n_batch = y.shape[0]
            return (loss_1) / n_batch + (penalty_l2 * weightsq_body + weightsq_head)

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
            grad(loss_vcnet)(params, batch, penalty_l2, penalty_disc, penalty_diff),
            state,
        )

    # initialise states
    _, init_params = init_fun_vcnet(rng_key, input_shape)
    opt_state = opt_init(init_params)

    # calculate number of batches per epoch
    batch_size = batch_size if batch_size < n else n
    n_batches = int(onp.round(n / batch_size)) if batch_size < n else 1
    train_indices = onp.arange(n)

    l_prev = LARGE_VAL
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
            l_curr = loss_vcnet(
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
                    l_final = loss_vcnet(params_curr, (X_val, y_val, w_val), 0, 0, 0)
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
        l_final = loss_vcnet(get_params(opt_state), (X_val, y_val, w_val), 0, 0, 0)
        return (
            params_best,
            (predict_fun_repr, predict_fun_head),
            (indices, indices_val),
            l_final,
        )

    return params_best, (predict_fun_repr, predict_fun_head), (indices, indices_val)


def getrepr_vcnet(
    X: jnp.ndarray,
    trained_params: dict,
    predict_funs: list,
) -> jnp.ndarray:
    # unpack inputs
    predict_fun_repr, predict_fun_head = predict_funs
    param_repr, param_t = (
        trained_params[0],
        trained_params[1],
    )

    # get representation
    representation = predict_fun_repr(param_repr, X)

    return representation


"""
VCNetPairNet starts from here
"""


class SNet1Nbr(BaseCATENet):
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
        return train_vcnetpairnet

    def _get_predict_function(self) -> Callable:
        return predict_vcnetpairnet


class VCNetPairNet(SNet1Nbr):
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


# Training functions for VCNet -------------------------------------------------


def predict_vcnetpairnet(
    X,
    d,  # needed for continuous treatments
    trained_params: dict,
    predict_funs: list,
    return_po: bool = False,
    return_prop: bool = False,
) -> jnp.ndarray:
    if return_prop:
        raise NotImplementedError("VCNet does not implement a propensity model.")

    # unpack inputs
    predict_fun_repr, predict_fun_head = predict_funs
    param_repr, param_0 = (
        trained_params[0],
        trained_params[1],
    )

    # get representation
    representation = predict_fun_repr(param_repr, X)

    # get potential outcomes
    mu_d, _ = predict_fun_head(param_0, (representation, d))

    if return_po:
        return mu_d
    else:
        return mu_d


def train_vcnetpairnet(
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
    init_fun_head, predict_fun_head = OutputHead(
        n_layers_out=n_layers_out,
        n_units_out=n_units_out,
        binary_y=binary_y,
        nonlin=nonlin,
    )

    def init_fun_vcnet(rng: float, input_shape: Tuple) -> Tuple[Tuple, List]:
        # chain together the layers
        # param should look like [repr, po_0, po_1]
        rng, layer_rng = random.split(rng)
        input_shape_repr, param_repr = init_fun_repr(layer_rng, input_shape)
        rng, layer_rng = random.split(rng)
        if same_init:
            # initialise both on same values
            input_shape, param_0 = init_fun_head(layer_rng, input_shape_repr)
        else:
            input_shape, param_0 = init_fun_head(layer_rng, input_shape_repr)

        return input_shape, [param_repr, param_0]

    # Define loss functions
    # loss functions for the head
    def loss_head(
        params: List,
        batch: Tuple[
            jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
        ],
    ) -> jnp.ndarray:
        # mse loss function
        inputs, targets, weights, inputs_cf, targets_cf, weights_cf = batch
        preds, _ = predict_fun_head(params, (inputs, weights))
        preds_cf, _ = predict_fun_head(params, (inputs_cf, weights_cf))
        gamma = 0
        return gamma * jnp.sum((preds - targets) ** 2) + jnp.sum(
            ((preds - preds_cf) - (targets - targets_cf)) ** 2
        )

    # complete loss function for all parts
    @jit
    def loss_vcnetpairnet(
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

        preds, _ = predict_fun_head(params[1], (reps, w))
        pairnet_loss = factual * jnp.sum((preds - y) ** 2) + (1 - factual) * loss_head(
            params[1], (reps, y, w, reps_cf, y_cf, w_cf)
        )

        # regularization on representation
        weightsq_body = sum(
            [jnp.sum(params[0][i][0] ** 2) for i in range(0, 2 * n_layers_r, 2)]
        )
        weightsq_head = 0  # heads_l2_penalty(params[1], params[1], n_layers_out, reg_diff, penalty_l2, penalty_diff)
        if not avg_objective:
            return pairnet_loss + (penalty_l2 * weightsq_body + weightsq_head)
        else:
            n_batch = y.shape[0]
            return (pairnet_loss) / n_batch + (
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
            grad(loss_vcnetpairnet)(
                params, batch, penalty_l2, penalty_disc, penalty_diff
            ),
            state,
        )

    # initialise states
    _, init_params = init_fun_vcnet(rng_key, input_shape)
    opt_state = opt_init(init_params)

    # calculate number of batches per epoch
    batch_size = batch_size if batch_size < n else n
    n_batches = int(onp.round(n / batch_size)) if batch_size < n else 1
    train_indices = onp.arange(n)

    l_prev = LARGE_VAL
    l_best = LARGE_VAL
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
                w_batch.squeeze(),
                X[cf_idx],
                y[cf_idx],
                w[cf_idx].squeeze(),
            )
            opt_state = update(
                i * n_batches + b, opt_state, next_batch, penalty_l2, penalty_disc
            )

        """"""
        # ABLATION 1: using current representations for neighbour distance computation vs frozen representations
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
                w_batch.squeeze(),
                X[cf_idx],
                y[cf_idx],
                w[cf_idx].squeeze(),
            )
            l_curr = loss_vcnetpairnet(
                params_curr,
                next_batch,
                penalty_l2,
                penalty_disc,
                penalty_diff,
                factual=False,
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
                    print("loss is nan")
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
                        w_batch.squeeze(),
                        X[cf_idx],
                        y[cf_idx],
                        w[cf_idx].squeeze(),
                    )
                    l_curr = loss_vcnetpairnet(
                        params_curr, next_batch, 0, 0, 0, factual=False
                    )
                    return (
                        params_best,
                        (predict_fun_repr, predict_fun_head),
                        (indices, indices_val),
                        l_curr,
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
            w_batch.squeeze(),
            X[cf_idx],
            y[cf_idx],
            w[cf_idx].squeeze(),
        )
        l_curr = loss_vcnetpairnet(params_curr, next_batch, 0, 0, 0, factual=False)
        return (
            params_best,
            (predict_fun_repr, predict_fun_head),
            (indices, indices_val),
            l_curr,
        )

    return params_best, (predict_fun_repr, predict_fun_head), (indices, indices_val)
