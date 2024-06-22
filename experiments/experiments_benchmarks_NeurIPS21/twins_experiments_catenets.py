"""
Utils to replicate Twins experiments with catenets
"""
import csv

# Author: Alicia Curth
import os
from pathlib import Path

import numpy as onp
import pandas as pd
from sklearn import clone
from sklearn.model_selection import train_test_split
from catenets.datasets.torch_dataset import (
    BaseTorchDataset,
    PairDataset,
)
from catenets.models.jax import PAIRNET_NAME
import numpy as np

from catenets.datasets import load
from catenets.experiment_utils.base import eval_root_mse
from catenets.models.jax import (
    RNET_NAME,
    T_NAME,
    TARNET_NAME,
    CFRNET_NAME,
    XNET_NAME,
    PAIRNET_NAME,
    RNet,
    TARNet,
    TNet,
    CFRNet,
    XNet,
    PairNet,
)

RESULT_DIR = Path("results/experiments_benchmarking/twins/")
EXP_DIR = Path("experiments/experiments_benchmarks_NeurIPS21/twins_datasets/")
SEP = "_"

repr_dir = {
    TARNET_NAME: RESULT_DIR / TARNET_NAME,
    CFRNET_NAME: RESULT_DIR / CFRNET_NAME,
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

ALL_MODELS = {
    # T_NAME: TNet(**PARAMS_DEPTH),
    TARNET_NAME: TARNet(**PARAMS_DEPTH),
    CFRNET_NAME: CFRNet(**PARAMS_DEPTH),
    XNET_NAME: XNet(**PARAMS_DEPTH_2),
    # PAIRNET_NAME: PairNet(**PARAMS_DEPTH),
    # RNET_NAME: RNet(**PARAMS_DEPTH_2),
}

model_hypers = {
    CFRNET_NAME: {"penalty_disc": 0.5},
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
    "arbitrary_pairs": True,
}


def dict_to_str(dict):
    return SEP.join([f"--{k}{SEP}{v}" for k, v in dict.items()])


run_baselines = PAIRNET_NAME not in ALL_MODELS.keys()
if PAIRNET_NAME in ALL_MODELS.keys():
    assert len(ALL_MODELS.keys()) == 1, "Only PAIRNET_NAME should be in ALL_MODELS"


def do_twins_experiment_loop(
    n_train_loop=[500, 1000, 2000, 5000, None],
    n_exp: int = 10,
    file_name: str = "twins",
    models: dict = None,
    test_size=0.5,
):
    for n in n_train_loop:
        print(f"Running twins experiments for subset_train {n}")
        do_twins_experiments(
            n_exp=n_exp,
            file_name=file_name,
            models=models,
            subset_train=n,
            test_size=test_size,
        )


def do_twins_experiments(
    n_exp: int = 10,
    file_name: str = "twins",
    models: dict = None,
    subset_train: int = None,
    prop_treated=0.5,
    test_size=0.5,
    model_params=None,
):
    if models is None:
        models = ALL_MODELS

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    for numcfz in [3]:
        for dist in ["euc"]:
            for pcs_dist in [True]:
                # overwrite the args
                pair_data_args["num_cfz"] = numcfz
                pair_data_args["dist"] = dist
                pair_data_args["pcs_dist"] = pcs_dist

                if run_baselines is False:
                    out_file = open(
                        RESULT_DIR
                        / (
                            file_name
                            + SEP
                            + str(prop_treated)
                            + SEP
                            + str(subset_train)
                            + dict_to_str(pair_data_args)
                            + dict_to_str(model_hypers[PAIRNET_NAME])
                            + ".csv"
                        ),
                        "w",
                        buffering=1,
                    )
                else:
                    out_file = open(
                        RESULT_DIR
                        / (
                            file_name
                            + SEP
                            + str(prop_treated)
                            + SEP
                            + str(subset_train)
                            + ".csv"
                        ),
                        "w",
                        buffering=1,
                    )

                writer = csv.writer(out_file)
                print(out_file)
                header = [name + "_pehe" for name in models.keys()]

                writer.writerow(header)

                for i_exp in range(n_exp):
                    pehe_out = []

                    # get data
                    (
                        data_dict,
                        ads_train,
                    ) = prepare_twins_agreement_data(
                        model_name=TARNET_NAME
                        if run_baselines
                        else PAIRNET_NAME,  # HACK
                        i_exp=i_exp,
                        treat_prop=prop_treated,
                        subset_train=subset_train,
                        **pair_data_args,
                    )
                    X = data_dict["X"]
                    X_t = data_dict["X_t"]
                    y = data_dict["y"]
                    w = data_dict["w"]
                    y0_out = data_dict["y0_out"]
                    y1_out = data_dict["y1_out"]

                    ite_out = y1_out - y0_out

                    for model_name, estimator in models.items():
                        # split data
                        print(f"Experiment {i_exp} with {model_name}")
                        estimator_temp = clone(estimator)
                        estimator_temp.set_params(**{"binary_y": True, "seed": i_exp})

                        # NOTE: Adding code for mmd penalty in CFRNet
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

                        if model_name in [CFRNET_NAME, TARNET_NAME]:
                            cate_pred_in, mu0_tr, mu1_tr = estimator_temp.predict(
                                X, return_po=True
                            )
                            cate_pred_out, mu0_te, mu1_te = estimator_temp.predict(
                                X_t, return_po=True
                            )

                            trn_reps = estimator_temp.getrepr(X)
                            tst_reps = estimator_temp.getrepr(X_t)

                            # concatenate mu0, mu1 to trn_reps
                            trn_reps = np.concatenate(
                                [trn_reps, mu0_tr, mu1_tr], axis=1
                            )
                            tst_reps = np.concatenate(
                                [tst_reps, mu0_te, mu1_te], axis=1
                            )

                            # TODO Save representations
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
                        cate_pred_out = estimator_temp.predict(X_t)

                        pehe_out.append(eval_root_mse(cate_pred_out, ite_out))

                    writer.writerow(pehe_out)

                out_file.close()


# utils ---------------------------------------------------------------------
def prepare_twins(
    treat_prop=0.5, seed=42, test_size=0.5, subset_train: int = None, return_w_t=False
):
    if not os.path.isdir(EXP_DIR):
        os.makedirs(EXP_DIR)

    out_base = (
        "preprocessed"
        + SEP
        + str(treat_prop)
        + SEP
        + str(subset_train)
        + SEP
        + str(test_size)
        + SEP
        + str(seed)
    )
    outfile_train = EXP_DIR / (out_base + SEP + "train.csv")
    outfile_test = EXP_DIR / (out_base + SEP + "test.csv")

    feat_list = [
        "dmage",
        "mpcb",
        "cigar",
        "drink",
        "wtgain",
        "gestat",
        "dmeduc",
        "nprevist",
        "dmar",
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "dtotord",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
        "adequacy_1",
        "adequacy_2",
        "adequacy_3",
        "pldel_1",
        "pldel_2",
        "pldel_3",
        "pldel_4",
        "pldel_5",
        "resstatb_1",
        "resstatb_2",
        "resstatb_3",
        "resstatb_4",
    ]

    if os.path.exists(outfile_train):
        print(f"Reading existing preprocessed twins file {out_base}")
        # use existing file
        df_train = pd.read_csv(outfile_train)
        X = onp.asarray(df_train[feat_list])
        y = onp.asarray(df_train[["y"]]).reshape((-1,))
        w = onp.asarray(df_train[["w"]]).reshape((-1,))

        df_test = pd.read_csv(outfile_test)
        X_t = onp.asarray(df_test[feat_list])
        w_t = onp.asarray(df_test[["w"]]).reshape((-1,))
        y0_out = onp.asarray(df_test[["y0"]]).reshape((-1,))
        y1_out = onp.asarray(df_test[["y1"]]).reshape((-1,))
    else:
        # create file
        print(f"Creating preprocessed twins file {out_base}")
        onp.random.seed(seed)

        x, w, y, pos, _, _ = load(
            "twins", seed=seed, treat_prop=treat_prop, train_ratio=1
        )

        X, X_t, y, y_t, w, w_t, y0_in, y0_out, y1_in, y1_out = train_test_split(
            x, y, w, pos[:, 0], pos[:, 1], test_size=test_size, random_state=seed
        )
        if subset_train is not None:
            X, y, w, y0_in, y1_in = (
                X[:subset_train, :],
                y[:subset_train],
                w[:subset_train],
                y0_in[:subset_train],
                y1_in[:subset_train],
            )

        # save data
        save_df_train = pd.DataFrame(X, columns=feat_list)
        save_df_train["y0"] = y0_in
        save_df_train["y1"] = y1_in
        save_df_train["w"] = w
        save_df_train["y"] = y
        save_df_train.to_csv(outfile_train)

        save_df_train = pd.DataFrame(X_t, columns=feat_list)
        save_df_train["y0"] = y0_out
        save_df_train["y1"] = y1_out
        save_df_train["w"] = w_t
        save_df_train["y"] = y_t
        save_df_train.to_csv(outfile_test)

    if return_w_t is False:
        return X, X_t, y, w, y0_out, y1_out
    else:
        return X, X_t, y, w, y0_out, y1_out, w_t


def prepare_twins_agreement_data(
    model_name,
    i_exp,
    treat_prop=0.5,
    seed=42,
    test_size=0.5,
    subset_train: int = None,
    **kwargs,
):
    X_trn, X_test, y_trn, w_trn, y0_out, y1_out, w_tst = prepare_twins(
        treat_prop=treat_prop,
        seed=seed,
        test_size=test_size,
        subset_train=subset_train,
        return_w_t=True,
    )
    # Load the CFR embeddings
    tar_path = Path(
        "results/experiments_benchmarking/ihdp/TARNet"
    )
    tar_test_emb, tar_test_emb = None, None

    if model_name in [PAIRNET_NAME]:
        tar_train = np.load(
            tar_path / f"twins-{treat_prop}-{subset_train}-{i_exp}-trn.npy"
        )
        tar_test = np.load(
            tar_path / f"ihdp-{treat_prop}-{subset_train}-{i_exp}-tst.npy"
        )
        print(f"Loaded Embeddings from {str(tar_path)}")

        tar_train_emb = tar_train[:, :-2]
        tar_test_emb = tar_test[:, :-2]

    ads_train = None
    if model_name == PAIRNET_NAME:
        ads_train = PairDataset(
            X=X_trn,
            beta=w_trn,
            y=y_trn,
            xemb=tar_test_emb,
            **kwargs,
        )

    return (
        {
            "X": X_trn,
            "y": y_trn,
            "w": w_trn,
            "cate_true_in": None,
            "X_t": X_test,
            "w_t": w_tst,
            "cate_true_out": y1_out - y0_out,
            "y1_out": y1_out,
            "y0_out": y0_out,
            "cfr_train_emb": None,
            "cfr_test_emb": None,
        },
        ads_train,
    )
