"""
Utils to replicate Twins experiments (Appendix E.2)
"""
# Author: Alicia Curth
import csv
import os
from pathlib import Path

import numpy as np
from sklearn import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from catenets.datasets.torch_dataset import PairDataset
import torch

from catenets.datasets.dataset_twins import load
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

from catenets.models.jax.base import check_shape_1d_data

RESULT_DIR = Path("results/experiments_inductive_bias/twins")
SEP = "_"

repr_dir = {
    TARNET_NAME: RESULT_DIR / TARNET_NAME,
}
for v in repr_dir.values():
    if not os.path.isdir(v):
        os.makedirs(v)

PARAMS_DEPTH = {"n_layers_r": 1, "n_layers_out": 1}
PARAMS_DEPTH_2 = {
    "n_layers_r": 1,
    "n_layers_out": 1,
    "n_layers_r_t": 1,
    "n_layers_out_t": 1,
}
PENALTY_DIFF = 0.01
PENALTY_ORTHOGONAL = 0.1

model_hypers = {
    CFRNET_NAME: {"penalty_disc": 0.5},
    PAIRNET_NAME: {
        "penalty_disc": 0.0,
        "penalty_l2": 1.0,
    },
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


meta_learners = [DRNET_NAME, XNET_NAME, RNET_NAME, T_NAME]


pair_data_args = {
    "det": False,
    "num_cfz": 3,
    "sm_temp": 1.0,
    "dist": "euc",  # cos/euc
    "pcs_dist": True,  # Process distances
    "drop_frac": 0.1,  # distance threshold
    "arbitrary_pairs": True,
}


def dict_to_str(dict):
    return SEP.join([f"--{k}{SEP}{v}" for k, v in dict.items()])


def do_twins_experiment_loop(
    n_train_loop= [500, 1000, 4000, None],
    prop_loop=[0.1, 0.25, 0.5, 0.75, 0.9],
    n_exp: int = 5,
    file_name: str = "twins",
    models: dict = None,
    test_size=0.5,
    save_reps: bool = False
):
    for n in n_train_loop:
        for prop in prop_loop:
            print(
                "Running twins experiment for {} training samples with {} treated.".format(
                    n, prop
                )
            )
            do_twins_experiments(
                n_exp=n_exp,
                file_name=file_name,
                models=models,
                subset_train=n,
                prop_treated=prop,
                test_size=test_size,
                save_reps=save_reps
            )


def do_twins_experiments(
    n_exp: int = 10,
    file_name: str = "twins",
    models: dict = None,
    subset_train: int = None,
    prop_treated=0.5,
    test_size=0.5,
    model_params=None,
    save_reps: bool = False,
):
    if models is None:
        models = ALL_MODELS

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    out_file = open(
        RESULT_DIR
        / (file_name + SEP + str(prop_treated) + SEP + str(subset_train) + ".csv"),
        "w",
        buffering=1,
    )

    writer = csv.writer(out_file)
    header = (
        [name + "_cate_in" for name in models.keys()]
        + [name + "_cate_out" for name in models.keys()]
    )

    writer.writerow(header)

    for i_exp in range(n_exp):
        pehe_out = []
        pehe_in = []
        auc_ite = []
        auc_mu0 = []
        auc_mu1 = []
        ap_mu0 = []
        ap_mu1 = []

        # get data
        data_path = Path("catenets/datasets/data")
        data_dict = load(
            data_path=data_path,
            train_ratio=1,
            treatment_type="rand",
            seed=i_exp,
            treat_prop=prop_treated,
        )
        # x, w, y, pos, _, _
        x = data_dict["train_x"]
        w = data_dict["train_w"]
        y = data_dict["train_y"]
        pos = data_dict["train_potential_y"]

        trn_indices, tst_indices = split_data(
            x.shape[0], random_state=i_exp, subset_train=subset_train, test_size=test_size
        )

        X, X_t = x[trn_indices, :], x[tst_indices, :]
        y, y_t = y[trn_indices], y[tst_indices]
        w, w_t = w[trn_indices], w[tst_indices]
        y0_in, y0_out = pos[trn_indices, 0], pos[tst_indices, 0]
        y1_in, y1_out = pos[trn_indices, 1], pos[tst_indices, 1]
        
        ite_in = y1_in - y0_in
        ite_out = y1_out - y0_out

        ite_out_encoded = label_binarize(y=ite_out, classes=[-1, 0, 1])

        n_test = X_t.shape[0]

        # split data
        for model_name, estimator in models.items():
            
            if model_name == PAIRNET_NAME:
                tar_path = Path(
                    "results/experiments_inductive_bias/twins/TARNet"
                )
                tar_train = np.load(
                    tar_path / f"twins-{prop_treated}-{subset_train}-{i_exp}-trn.npy"
                )
                tar_test = np.load(
                    tar_path / f"twins-{prop_treated}-{subset_train}-{i_exp}-tst.npy"
                )
                print(f"Loaded Embeddings from {str(tar_path)}")

                tar_train_emb = tar_train[:, :-2]
                tar_test_emb = tar_test[:, :-2]

                ads_train = PairDataset(
                    X=X,
                    beta=w,
                    y=y,
                    xemb=tar_train_emb,
                    **pair_data_args,
                )
            
            
            print(f"Experiment {i_exp} with {model_name}")
            estimator_temp = clone(estimator)
            estimator_temp.set_params(**{"binary_y": True})

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

            if save_reps is True and model_name in [TARNET_NAME]:
                cate_pred_in, mu0_tr, mu1_tr = estimator_temp.predict(X, return_po=True)
                cate_pred_out, mu0_te, mu1_te = estimator_temp.predict(
                    X_t, return_po=True
                )

                trn_reps = estimator_temp.getrepr(X)
                tst_reps = estimator_temp.getrepr(X_t)

                # concatenate mu0, mu1 to trn_reps
                trn_reps = np.concatenate([trn_reps, mu0_tr, mu1_tr], axis=1)
                tst_reps = np.concatenate([tst_reps, mu0_te, mu1_te], axis=1)

                np.save(
                    repr_dir[model_name]
                    / f"twins-{prop_treated}-{subset_train}-{i_exp}-trn.npy",
                    trn_reps,
                )
                np.save(
                    repr_dir[model_name]
                    / f"twins-{prop_treated}-{subset_train}-{i_exp}-tst.npy",
                    tst_reps,
                )

            if model_name not in meta_learners:
                cate_pred_in, mu0_pred, mu1_pred = estimator_temp.predict(
                    X, return_po=True
                )
                cate_pred_out, mu0_pred, mu1_pred = estimator_temp.predict(
                    X_t, return_po=True
                )

                # create probabilities for each possible level of ITE
                probs = np.zeros((n_test, 3))
                probs[:, 0] = (mu0_pred * (1 - mu1_pred)).reshape((-1,))  # P(Y1-Y0=-1)
                probs[:, 1] = (
                    (mu0_pred * mu1_pred) + ((1 - mu0_pred) * (1 - mu1_pred))
                ).reshape(
                    (-1,)
                )  # P(Y1-Y0=0)
                probs[:, 2] = (mu1_pred * (1 - mu0_pred)).reshape((-1,))  # P(Y1-Y0=1)
                auc_ite.append(roc_auc_score(ite_out_encoded, probs))

                # Let us only record the CATE performance
                auc_mu0.append(eval_roc_auc(y0_out, mu0_pred))
                auc_mu1.append(eval_roc_auc(y1_out, mu1_pred))
                ap_mu0.append(eval_ap(y0_out, mu0_pred))
                ap_mu1.append(eval_ap(y1_out, mu1_pred))
            else:
                cate_pred_out = estimator_temp.predict(X_t)
                cate_pred_in = estimator_temp.predict(X)

            pehe_in.append(eval_root_mse(cate_pred_in, ite_in))
            pehe_out.append(eval_root_mse(cate_pred_out, ite_out))

        writer.writerow(pehe_in + pehe_out)

    out_file.close()


# utils -------
def split_data(num_train, test_size=0.5, random_state=42, subset_train: int = None):

    all_indices = np.arange(num_train)
    trn_indices, tst_indices = train_test_split(
        all_indices, test_size=test_size, random_state=random_state
    )
    if subset_train is not None:
        trn_indices = trn_indices[:subset_train]
    
    return trn_indices, tst_indices
    

def eval_roc_auc(targets, preds):
    preds = check_shape_1d_data(preds)
    targets = check_shape_1d_data(targets)
    return roc_auc_score(targets, preds)


def eval_ap(targets, preds):
    preds = check_shape_1d_data(preds)
    targets = check_shape_1d_data(targets)
    return average_precision_score(targets, preds)
