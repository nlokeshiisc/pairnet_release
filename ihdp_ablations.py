"""
Utils to replicate IHDP ablations
"""
import csv
import os
from pathlib import Path
from typing import  Union
import copy
import torch

import numpy as np
from sklearn import clone

from continuous.utils.common_utils import set_seed

from catenets.datasets.dataset_ihdp import (
    get_one_data_set,
    load_raw,
    prepare_ihdp_pairnet_data,
)
from catenets.datasets.torch_dataset import (
    BaseTorchDataset as TorchDS,
)
from catenets.experiment_utils.base import eval_root_mse

from catenets.models.jax import (
    PAIRNET_NAME,
    PairNet,
)

DATA_DIR = Path("catenets/datasets/data/")
RESULT_DIR = Path("results/experiments_benchmarking/ihdp/ablations")

SEP = "_"

PARAMS_DEPTH = {"n_layers_r": 3, "n_layers_out": 2}

models = {
    PAIRNET_NAME: PairNet(**PARAMS_DEPTH),
}

model_hypers = {
    PAIRNET_NAME: {
        "penalty_disc": 0.0,
        "penalty_l2": 0.0,
    },
}

pair_data_args = {
    "det": False,
    "num_cfz": 3,
    "sm_temp": 1.0,
    "dist": "euc",  # cos/euc
    "pcs_dist": True,  # Process distances
    "drop_frac": 0.1,  # distance threshold
    "arbitrary_pairs": False,
    "check_perex_contrib": False,
    "OT": False,
}


def dict_to_str(dict):
    return SEP.join([f"--{k}{SEP}{v}" for k, v in dict.items()])


ALL_MODELS = {
    PAIRNET_NAME: PairNet(**PARAMS_DEPTH),
}


def do_ihdp_experiments(
    n_exp: Union[int, list] = 100,
    n_reps: int = 1,
    model_params: dict = None,
    file_name: str = "ihdp_all",
) -> None:

    setting = "C"

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # get data
    data_train, data_test = load_raw(DATA_DIR)

    out_file = open(
        RESULT_DIR
        / (
            file_name + ".csv"
        ),
        "w",
        buffering=1,
    )
    print(out_file)
    writer = csv.writer(out_file)
    header = (
        ["exp", "run", "cate_var_in", "cate_var_out", "y_var_in"]
        + [PAIRNET_NAME + "_in"]
        + [PAIRNET_NAME + "_out"]
    )
    writer.writerow(header)

    if isinstance(n_exp, int):
        experiment_loop = list(range(1, n_exp + 1))
    elif isinstance(n_exp, list):
        experiment_loop = n_exp
    else:
        raise ValueError(
            "n_exp should be either an integer or a list of integers."
        )

    for i_exp in experiment_loop:
        data_exp = get_one_data_set(data_train, i_exp=i_exp, get_po=True)
        data_exp_test = get_one_data_set(
            data_test, i_exp=i_exp, get_po=True
        )

        data_dict, ads_train = prepare_ihdp_pairnet_data(
            i_exp=i_exp,
            model_name=PAIRNET_NAME,
            data_train=data_exp,
            data_test=data_exp_test,
            setting=setting,
            **pair_data_args,
        )

        if pair_data_args["check_perex_contrib"] == True:
            continue

        X, y, w, cate_true_in, X_t, cate_true_out = (
            data_dict["X"],
            data_dict["y"],
            data_dict["w"],
            data_dict["cate_true_in"],
            data_dict["X_t"],
            data_dict["cate_true_out"],
        )

        ads_train: TorchDS = ads_train

        # compute some stats
        cate_var_in = np.var(cate_true_in)
        cate_var_out = np.var(cate_true_out)
        y_var_in = np.var(y)

        pehe_in = []
        pehe_out = []

        for model_name, estimator in models.items():
            k = 0
            print(f"Experiment {i_exp}, run {k}, with {model_name}")
            estimator_temp = clone(estimator)
            estimator_temp.set_params(seed=k)
            if model_name in model_hypers.keys():
                if model_params is None:
                    model_params = {}
                model_params.update(model_hypers[model_name])

            if model_params is not None:
                estimator_temp.set_params(**model_params)

            if model_name in model_hypers.keys():
                # Delete the keys from the model_params dictionary
                for key in model_hypers[model_name].keys():
                    del model_params[key]

            estimator_temp.agree_fit(ads_train)
            
            cate_pred_in = estimator_temp.predict(X)
            cate_pred_out = estimator_temp.predict(X_t)

            if isinstance(cate_pred_in, torch.Tensor):
                cate_pred_in = cate_pred_in.detach().numpy()
            if isinstance(cate_pred_out, torch.Tensor):
                cate_pred_out = cate_pred_out.detach().numpy()

            pehe_in.append(eval_root_mse(cate_pred_in, cate_true_in))
            pehe_out.append(eval_root_mse(cate_pred_out, cate_true_out))
            

        writer.writerow(
            [i_exp, k, cate_var_in, cate_var_out, y_var_in]
            + pehe_in
            + pehe_out
        )
    out_file.close()


# Ablation on delta_pair
set_seed(42)
seeds = list(np.random.randint(100, size=5))
for delta_pair in [0.5]:# [0, 0.1, 0.25, 0.5]:
    pair_data_args["drop_frac"] = delta_pair
    do_ihdp_experiments(
        n_exp=seeds,
        file_name=f"ihdp_ablation_delta_pair_{delta_pair}",
    )

# Ablations on num_cfz
set_seed(42)
seeds = list(np.random.randint(100, size=5))
for num_cfz in [1, 2, 3, 4, 5]:
    pair_data_args["num_cfz"] = num_cfz
    do_ihdp_experiments(
        n_exp=seeds,
        file_name=f"ihdp_ablation_num_cfz_{num_cfz}",
    )

# Ablations on the \phi_{\text{fct}}
set_seed(42)
seeds = list(np.random.randint(100, size=5))
pair_data_args["arbitrary_pairs"] = True
do_ihdp_experiments(
    n_exp=seeds,
    file_name=f"ihdp_ablation_arbitrary_pairs",   
)

# This is for using the PairNet's phi at run time to compute pairs. we need to make dynamic_phi True in the corresponding files
set_seed(42)
seeds = list(np.random.randint(100, size=5))
model_hypers[PAIRNET_NAME]["dynamic_phi"] = True
do_ihdp_experiments(
    n_exp=seeds,
    file_name=f"ihdp_ablation_dynamic_phi",   
)