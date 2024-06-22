"""
Utils to replicate ACIC2016 experiments with catenets
"""
# Author: Alicia Curth
import csv
import os
from pathlib import Path

import numpy as np
from sklearn import clone

from catenets.datasets import load
from catenets.datasets.torch_dataset import (
    BaseTorchDataset as TorchDS,
)
from catenets.datasets.dataset_acic2016 import load_agree_dataset
from catenets.experiment_utils.base import eval_root_mse

from catenets.models.jax import (
    RNET_NAME,
    T_NAME,
    TARNET_NAME,
    CFRNET_NAME,
    PAIRNET_NAME,
    XNET_NAME,
    DRAGON_NAME,
    FLEXTE_NAME,
    DRNET_NAME,
    RNet,
    TARNet,
    CFRNet,
    PairNet,
    FlexTENet,
    DragonNet,
    DRNet,
    TNet,
    XNet,
)

RESULT_DIR = Path("results/experiments_benchmarking/acic2016/")
SEP = "_"

PENALTY_DIFF = 0.01
PENALTY_ORTHOGONAL = 0.1

repr_dir = {
    TARNET_NAME: RESULT_DIR / TARNET_NAME,
    CFRNET_NAME: RESULT_DIR / CFRNET_NAME,
}
for v in repr_dir.values():
    if not os.path.isdir(v):
        os.makedirs(v)

SEP = "_"

PARAMS_DEPTH = {"n_layers_r": 3, "n_layers_out": 2}
PARAMS_DEPTH_2 = {
    "n_layers_r": 3,
    "n_layers_out": 2,
    "n_layers_r_t": 3,
    "n_layers_out_t": 2,
}

model_hypers = {
    CFRNET_NAME: {"penalty_disc": 0.5},
    PAIRNET_NAME: {
        "penalty_disc": 0.0,
        "penalty_l2": 1.0,
        "dynamic_phi": False,
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
    T_NAME: TNet(**PARAMS_DEPTH),
    TARNET_NAME: TARNet(**PARAMS_DEPTH),
    CFRNET_NAME: CFRNet(**PARAMS_DEPTH),
    PAIRNET_NAME: PairNet(**PARAMS_DEPTH),
    RNET_NAME: RNet(**PARAMS_DEPTH_2),
    XNET_NAME: XNet(**PARAMS_DEPTH_2),
    FLEXTE_NAME: FlexTENet(
        penalty_orthogonal=PENALTY_ORTHOGONAL, penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH
    ),
    DRNET_NAME: DRNet(first_stage_strategy="Tar", **PARAMS_DEPTH_2),
    DRAGON_NAME: DragonNet(**PARAMS_DEPTH),
}


def do_acic_experiments(
    n_exp: int = 10,
    n_reps=5,
    file_name: str = "results_catenets",
    simu_num: int = 1,
    models: dict = None,
    train_size: int = 4000,
    pre_trans: bool = True,
    save_reps: bool = False,
):
    model_params = None

    if models is None:
        models = ALL_MODELS

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    out_file = open(RESULT_DIR / f"v{simu_num}{SEP}{file_name}.csv", "w", buffering=1)
    
    writer = csv.writer(out_file)
    header = (
        ["file_name", "cate_var_in", "cate_var_out", "y_var_in"]
        + [name + "_in" for name in models.keys()]
        + [name + "_out" for name in models.keys()]
    )
    writer.writerow(header)
    
    print(f"Out file: {out_file.name}")

    for i_exp in range(n_exp):
        pehe_in = []
        pehe_out = []
        for model_name, estimator in models.items():
            try:
                print(f"Experiment {i_exp}, with {model_name}")
                estimator_temp = clone(estimator)
                estimator_temp.set_params(seed=0)

                # get data
                data_dict, ads_train = load_agree_dataset(
                    model_name=model_name,
                    data_path="acic2016",
                    preprocessed=pre_trans,
                    original_acic_outcomes=True,
                    i_exp=i_exp,
                    simu_num=simu_num,
                    train_size=train_size,
                    **pair_data_args,
                )

                (X, w, y, po_train, X_test, w_test, y_test, po_test) = (
                    data_dict["X_train"],
                    data_dict["w_train"],
                    data_dict["y_train"],
                    data_dict["po_train"],
                    data_dict["X_test"],
                    data_dict["w_test"],
                    data_dict["y_test"],
                    data_dict["po_test"],
                )

                ads_train: TorchDS = ads_train  # For IDE hints

                cate_in = po_train[:, 1] - po_train[:, 0]
                cate_out = po_test[:, 1] - po_test[:, 0]

                cate_var_in = np.var(cate_in)
                cate_var_out = np.var(cate_out)
                y_var_in = np.var(y)

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

                # fit estimator
                if model_name in [PAIRNET_NAME]:
                    estimator_temp.agree_fit(ads_train)
                else:
                    estimator_temp.fit(X=X, y=y, w=w)

                if model_name in [TARNET_NAME, CFRNET_NAME]:
                    cate_pred_in, mu0_tr, mu1_tr = estimator_temp.predict(
                        X, return_po=True
                    )
                    cate_pred_out, mu0_te, mu1_te = estimator_temp.predict(
                        X_test, return_po=True
                    )

                    if save_reps:
                        dump_reps(
                            simu_num,
                            train_size,
                            pre_trans,
                            model_name,
                            i_exp,
                            X,
                            X_test,
                            estimator_temp,
                            mu0_tr,
                            mu1_tr,
                            mu0_te,
                            mu1_te,
                        )
                else:
                    cate_pred_in = estimator_temp.predict(X)
                    cate_pred_out = estimator_temp.predict(X_test)

                pehe_in.append(eval_root_mse(cate_pred_in, cate_in))
                pehe_out.append(eval_root_mse(cate_pred_out, cate_out))
            except:
                print(
                    f"Experiment {i_exp}, with {model_name} failed"
                )
                pehe_in.append(-1)
                pehe_out.append(-1)

        writer.writerow(
            [i_exp, cate_var_in, cate_var_out, y_var_in]
            + pehe_in
            + pehe_out
        )

    out_file.close()


def dump_reps(
    simu_num,
    train_size,
    pre_trans,
    model_name,
    i_exp,
    X,
    X_test,
    estimator_temp,
    mu0_tr,
    mu1_tr,
    mu0_te,
    mu1_te,
):
    trn_reps = estimator_temp.getrepr(X)
    tst_reps = estimator_temp.getrepr(X_test)

    # concatenate mu0, mu1 to trn_reps
    trn_reps = np.concatenate([trn_reps, mu0_tr, mu1_tr], axis=1)
    tst_reps = np.concatenate([tst_reps, mu0_te, mu1_te], axis=1)

    # Save representations
    np.save(
        repr_dir[model_name]
        / f"acic-{SEP}{str(pre_trans)}{SEP}{str(simu_num)}{SEP}{str(train_size)}-{i_exp}-trn.npy",
        trn_reps,
    )
    np.save(
        repr_dir[model_name]
        / f"acic-{SEP}{str(pre_trans)}{SEP}{str(simu_num)}{SEP}{str(train_size)}-{i_exp}-tst.npy",
        tst_reps,
    )
