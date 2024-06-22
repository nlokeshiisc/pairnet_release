# Run IHDP continuous experiments

import continuous.src.data_helper as dh
from catenets.models.jax import (
    VCNet,
    VCNetPairNet,
    DRNetC,
    DRNetPairNet,
    VCNET_NAME,
    VCNETPAIRNET_NAME,
    DRNETC_NAME,
    DRNETPAIRNET_NAME,
)
import continuous.utils.common_utils as cont_cu
from pathlib import Path
import pickle as pkl
import numpy as np
import torch
import wandb
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", default="sample", type=str)
parser.add_argument("--dump_reps", default=False, type=bool)
args = parser.parse_args()
wandb.login()
import csv

IHDP_CONT = "ihdp"
TCGA_SINGLE_0 = "tcga_single"
TCGA_SINGLE_1 = "tcga_single_1"
TCGA_SINGLE_2 = "tcga_single_2"
NEWS_CONT = "news"

all_datasets = [
    IHDP_CONT,
    NEWS_CONT,
    TCGA_SINGLE_0,
    TCGA_SINGLE_1,
    TCGA_SINGLE_2,
]
all_seeds = {
    IHDP_CONT: np.arange(10),
    NEWS_CONT: np.arange(10),
    TCGA_SINGLE_0: np.arange(10),
    TCGA_SINGLE_1: np.arange(10),
    TCGA_SINGLE_2: np.arange(10),
}

dataset_names = {
    IHDP_CONT: "ihdp",
    TCGA_SINGLE_0: "tcga0",
    TCGA_SINGLE_1: "tcga1",
    TCGA_SINGLE_2: "tcga2",
    NEWS_CONT: "news",
}

stopping_params = {
    IHDP_CONT: 1000,
    NEWS_CONT: 400,
    TCGA_SINGLE_0: 200,
    TCGA_SINGLE_1: 200,
    TCGA_SINGLE_2: 200,
}

# to download dataset files
file_id = "1EMjfcYPFsdk_ErNDAt4pN9qW6dyiuCwQ"
output_file = "cont_datasets.zip"

if os.path.exists(output_file):
    print("continuous datasets already downloaded")
else:
    import gdown

    gdown.download(id=file_id, output=output_file, quiet=False)

if not os.path.exists("continuous/dataset"):
    import zipfile

    with zipfile.ZipFile("cont_datasets.zip", "r") as zip_ref:
        zip_ref.extractall("continuous")


for dataset_name, dataset_text in dataset_names.items():
    n_iter = stopping_params[dataset_name]
    model_dict = {
        VCNET_NAME: VCNet,
        VCNETPAIRNET_NAME: VCNetPairNet,
        DRNETC_NAME: DRNetC,
        DRNETPAIRNET_NAME: DRNetPairNet,
    }
    outpath = f"continuous/results/final/{dataset_text}_cont_{args.file_name}.csv"
    outfile = open(outpath, "w", buffering=1)
    writer = csv.writer(outfile)
    header = (
        [
            "exp",
        ]
        + [name + "_MISE_in" for name in model_dict.keys()]
        + [name + "_MISE_out" for name in model_dict.keys()]
    )
    writer.writerow(header)
    for dataset_num in all_seeds[dataset_name]:
        mises = [dataset_num]
        mises_in = []
        mises_out = []
        for model_name, model_type in model_dict.items():
            print(f"Training {model_name} on {dataset_text} : exp {dataset_num}")

            cont_cu.set_seed(dataset_num)
            (
                train_matrix,
                test_matrix,
                t_grid,
                indim,
                *train_test_indices,
            ) = dh.load_dataset(dataset_name, dataset_num=dataset_num)

            if "tcga" in dataset_name:
                train_idxs, tst_idxs = (
                    train_test_indices[0].dataset["metadata"]["train_index"],
                    train_test_indices[0].dataset["metadata"]["test_index"],
                )
            else:
                train_idxs, tst_idxs = (train_test_indices[0], train_test_indices[1])

            d, X, y = (
                train_matrix[:, 0].view(-1, 1),
                train_matrix[:, 1:-1],
                train_matrix[:, -1].view(-1, 1),
            )

            model = model_type(n_iter=n_iter)

            wandb.init(
                project="vcnet-cts",
                config={
                    "method": model_name,
                    "dataset": dataset_text,
                    "experiment": dataset_num,
                    "patience": 10,
                    # "representations": "fixed",
                    # "stopping": "factual",
                    # "loss": "floss+diffloss",
                },
            )
            if model_name in [VCNETPAIRNET_NAME, DRNETPAIRNET_NAME]:
                if model_name == VCNETPAIRNET_NAME:
                    reps_path = f"continuous/vcnet-representations/{dataset_text}/"
                elif model_name == DRNETPAIRNET_NAME:
                    reps_path = f"continuous/drnet-representations/{dataset_text}/"
                file_name = f"{dataset_num}.npy"
                full_file_path = os.path.join(reps_path, file_name)
                if model_name == VCNETPAIRNET_NAME:
                    if not os.path.exists(full_file_path):
                        raise FileNotFoundError(
                            "VCNet needs to be trained for this dataset and experiment"
                        )
                elif model_name == DRNETPAIRNET_NAME:
                    if not os.path.exists(full_file_path):
                        raise FileNotFoundError(
                            "DRNet needs to be trained for this dataset and experiment"
                        )
                with open(full_file_path, "rb") as f:
                    train_reps = np.load(f)

                model.cont_agree_fit(
                    X.cpu().numpy(), y.cpu().numpy(), d.cpu().numpy(), train_reps
                )
            else:
                model.cont_fit(X.cpu().numpy(), y.cpu().numpy(), d.cpu().numpy())
            wandb.finish()
            if model_name in [VCNET_NAME, DRNETC_NAME] and args.dump_reps:
                if model_name == VCNET_NAME:
                    reps_path = f"continuous/vcnet-representations/{dataset_text}/"
                elif model_name == DRNETC_NAME:
                    reps_path = f"continuous/drnet-representations/{dataset_text}/"
                file_name = f"{dataset_num}.npy"
                if not os.path.exists(reps_path):
                    os.makedirs(reps_path)
                full_file_path = os.path.join(reps_path, file_name)
                train_reps = model.getrepr(X.cpu().numpy())
                with open(full_file_path, "wb") as f:
                    np.save(f, train_reps)

            dataset_dir = Path("continuous/dataset")
            response_files = {
                IHDP_CONT: dataset_dir
                / IHDP_CONT
                / "tr_h_1.0_te_l_0.0_h1.0/ihdp_response.pkl",
                NEWS_CONT: dataset_dir
                / NEWS_CONT
                / "tr_h_1.0_te_h_1.0/news_response.pkl",
                TCGA_SINGLE_0: dataset_dir
                / "tcga"
                / "cf_responses_tcga_single.pkl",
                TCGA_SINGLE_1: dataset_dir
                / "tcga"
                / "cf_responses_tcga_single_1.pkl",
                TCGA_SINGLE_2: dataset_dir
                / "tcga"
                / "cf_responses_tcga_single_2.pkl",
            }

            cf_responses_all = pkl.load(open(response_files[dataset_name], "rb"))
            cf_responses = cf_responses_all[tst_idxs]

            train_indices = train_idxs[np.array(model.train_indices)]
            val_indices = train_idxs[np.array(model.val_indices)]
            trn_d, trn_X, trn_y = (
                d[np.array(model.train_indices)],
                X[np.array(model.train_indices)],
                y[np.array(model.train_indices)],
            )
            cf_responses_train = cf_responses_all[train_indices]

            tst_d, tst_X, tst_y = (
                test_matrix[:, 0].view(-1, 1),
                test_matrix[:, 1:-1],
                test_matrix[:, -1].view(-1, 1),
            )

            num_integration_samples = 64
            t_samples = torch.arange(0.01, 1, 1 / num_integration_samples).reshape(
                1, -1
            )

            t_samples_test = torch.repeat_interleave(t_samples, tst_X.shape[0], dim=0)
            t_samples_test = torch.cat([tst_d, t_samples_test], dim=1)
            t_samples_test = t_samples_test.view(-1, 1)
            tst_X = torch.repeat_interleave(tst_X, num_integration_samples + 1, dim=0)

            t_samples_train = torch.repeat_interleave(t_samples, trn_X.shape[0], dim=0)
            t_samples_train = torch.cat([trn_d, t_samples_train], dim=1)
            t_samples_train = t_samples_train.view(-1, 1)
            trn_X = torch.repeat_interleave(trn_X, num_integration_samples + 1, dim=0)

            """
            computing risk on the entire data is infeasible for TCGA
            """
            # cf_preds_test = model.predict_cont(
            #     tst_X.cpu().numpy(), t_samples_test.cpu().numpy()
            # ).reshape(-1, num_integration_samples + 1)

            # cf_preds_train = model.predict_cont(
            #     trn_X.cpu().numpy(), t_samples_train.cpu().numpy()
            # ).reshape(-1, num_integration_samples + 1)

            # Define the size of each chunk or slice
            chunk_size = 1000  # You can adjust this based on your available GPU memory

            # Get the total number of samples in the input matrix
            num_samples = trn_X.shape[0]

            # Initialize an empty array to store the concatenated results
            results = []

            # Iterate over the input matrix in chunks
            for i in range(0, num_samples, chunk_size):
                # Slice the input matrices into chunks
                trn_X_chunk = trn_X[i : min(i + chunk_size, num_samples)]
                t_samples_chunk = t_samples_train[i : min(i + chunk_size, num_samples)]

                # Make predictions for the chunk
                cf_preds_chunk = model.predict_cont(
                    trn_X_chunk.cpu().numpy(), t_samples_chunk.cpu().numpy()
                )

                # Append the chunk's results to the results list
                results.append(cf_preds_chunk)

            # Concatenate the results from all chunks
            cf_preds_train = np.concatenate(results, axis=0)
            cf_preds_train = cf_preds_train.reshape(-1, num_integration_samples + 1)

            # Define the size of each chunk or slice
            chunk_size = 1000  # You can adjust this based on your available GPU memory

            # Get the total number of samples in the input matrix
            num_samples = tst_X.shape[0]

            # Initialize an empty array to store the concatenated results
            results = []

            # Iterate over the input matrix in chunks
            for i in range(0, num_samples, chunk_size):
                # Slice the input matrices into chunks
                tst_X_chunk = tst_X[i : min(i + chunk_size, num_samples)]
                t_samples_chunk = t_samples_test[i : min(i + chunk_size, num_samples)]

                # Make predictions for the chunk
                cf_preds_chunk = model.predict_cont(
                    tst_X_chunk.cpu().numpy(), t_samples_chunk.cpu().numpy()
                )

                # Append the chunk's results to the results list
                results.append(cf_preds_chunk)

            # Concatenate the results from all chunks
            cf_preds_test = np.concatenate(results, axis=0)
            cf_preds_test = cf_preds_test.reshape(-1, num_integration_samples + 1)

            # Now final_result contains the concatenated results for the entire matrix

            gold_diff_test = tst_y.view(-1, 1) - cf_responses
            pred_diff_test = cf_preds_test[:, 0].reshape(-1, 1) - cf_preds_test[:, 1:]
            pred_diff_test = torch.tensor(np.array(pred_diff_test))
            mise_test = torch.mean(
                torch.sqrt(
                    torch.mean(torch.square(gold_diff_test - pred_diff_test), dim=1)
                )
            ).item()

            gold_diff_train = trn_y.view(-1, 1) - cf_responses_train
            pred_diff_train = (
                cf_preds_train[:, 0].reshape(-1, 1) - cf_preds_train[:, 1:]
            )
            pred_diff_train = torch.tensor(np.array(pred_diff_train))
            mise_train = torch.mean(
                torch.sqrt(
                    torch.mean(torch.square(gold_diff_train - pred_diff_train), dim=1)
                )
            ).item()
            print(f"{model_name}, MISE Train = {mise_train}, MISE Test = {mise_test}")
            mises_in.append(mise_train)
            mises_out.append(mise_test)
        mises.extend(mises_in)
        mises.extend(mises_out)
        writer.writerow(mises)
