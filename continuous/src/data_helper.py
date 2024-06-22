import torch
import continuous.src.constants as constants
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

dataset_dir = constants.catenets_dir / "continuous" / "dataset"
import pandas as pd
import continuous.utils.common_utils as cu
from continuous.src.tcga.tcga_data import TCGA_Data, get_dataset_splits
import pickle as pkl


class Dataset_from_matrix(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        sample = self.data_matrix[idx, :]
        return (sample[0:-1], sample[-1], idx)


def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator


def load_dataset(dataset_name, dataset_num, entire_dataset=None):
    if dataset_name == constants.IHDP_CONT:
        assert dataset_num < 100, "We have only 100 splits for ihdp dataset"
        data_dir = dataset_dir / "ihdp" / "tr_h_1.0_te_l_0.0_h1.0"

        data_matrix = torch.load(data_dir / "data_matrix.pt")
        mini, maxi = torch.min(data_matrix[:, -1]), torch.max(data_matrix[:, -1])
        # data_matrix[:, -1] = (data_matrix[:,-1] - mini) / (maxi - mini)
        data_grid = torch.load(data_dir / "t_grid.pt")

        train_idx = torch.load(data_dir / "eval" / str(dataset_num) / "idx_train.pt")
        test_idx = torch.load(data_dir / "eval" / str(dataset_num) / "idx_test.pt")

        train_matrix = data_matrix[train_idx]
        test_matrix = data_matrix[test_idx]

        t_grid = data_grid[:, test_idx]
        # t_grid[1, :] = (t_grid[1, :] - mini) / (maxi - mini)
        indim = 25

        return train_matrix, test_matrix, t_grid, indim, train_idx, test_idx

    if dataset_name == constants.IHDP_DISC:
        # seeds = [1, 9, 13, 26, 68]
        seeds = [9, 13]
        assert dataset_num < 5, "We have only 100 splits for ihdp dataset"
        data_dir = Path("continuous/dataset/ihdp")

        data_matrix = pkl.load(
            open(
                data_dir / f"ihdp_binary_{seeds[dataset_num]}.pkl",
                "rb",
            )
        )
        data_matrix = torch.Tensor(data_matrix)[:, :-1]

        train_idx = torch.arange(len(data_matrix))[:-75]
        test_idx = torch.arange(len(data_matrix))[-75:]

        train_matrix = data_matrix[train_idx]
        test_matrix = data_matrix[test_idx]

        t_grid = None
        indim = 25

        return train_matrix, test_matrix, t_grid, indim, test_idx

    if dataset_name == constants.NEWS_CONT:
        assert dataset_num < 20, "News dataset has only 20 splits"
        # data_dir = dataset_dir / 'news' / 'tr_h_1.0_te_h_1.0'
        data_dir = Path(
            constants.catenets_dir / "continuous" / "dataset/news/tr_h_1.0_te_h_1.0"
        )
        data_matrix = torch.load(data_dir / "data_matrix.pt")
        mini, maxi = torch.min(data_matrix[:, -1]), torch.max(data_matrix[:, -1])
        # data_matrix[:, -1] = (data_matrix[:,-1] - mini) / (maxi - mini)

        data_grid = torch.load(data_dir / "t_grid.pt")
        train_idx = torch.load(data_dir / "eval" / str(dataset_num) / "idx_train.pt")
        test_idx = torch.load(data_dir / "eval" / str(dataset_num) / "idx_test.pt")
        train_matrix = data_matrix[train_idx]
        test_matrix = data_matrix[test_idx]

        t_grid = data_grid[:, test_idx]
        # t_grid[1, :] = (t_grid[1, :] - mini) / (maxi - mini)
        indim = 498

        return train_matrix, test_matrix, t_grid, indim, train_idx, test_idx

    elif dataset_name == constants.TCGA_SINGLE_0:
        cu.set_seed(dataset_num * 100)

        dataset_params = dict()
        dataset_params["num_treatments"] = 3
        dataset_params["treatment_selection_bias"] = 2
        dataset_params["dosage_selection_bias"] = 2
        dataset_params["save_dataset"] = True
        dataset_params["test_fraction"] = 0.2
        data_class = TCGA_Data(
            args=dataset_params, dataset_num=dataset_num, treatment_type=0
        )

        dataset = data_class.dataset
        dataset_train, dataset_test = get_dataset_splits(dataset)

        indim = dataset_train["x"].shape[1]

        T = torch.FloatTensor
        vccat = lambda t, x, y: torch.cat(
            (T(t).view(-1, 1), T(x), T(y).view(-1, 1)), dim=1
        ).to(dtype=torch.float64)

        train_matrix = vccat(dataset_train["d"], dataset_train["x"], dataset_train["y"])
        test_matrix = vccat(dataset_test["d"], dataset_test["x"], dataset_test["y"])

        return train_matrix, test_matrix, None, indim, data_class

    elif dataset_name == constants.TCGA_SINGLE_1:
        cu.set_seed(dataset_num * 100)

        dataset_params = dict()
        dataset_params["num_treatments"] = 3
        dataset_params["treatment_selection_bias"] = 2
        dataset_params["dosage_selection_bias"] = 2
        dataset_params["save_dataset"] = True
        dataset_params["test_fraction"] = 0.2
        data_class = TCGA_Data(
            args=dataset_params, dataset_num=dataset_num, treatment_type=1
        )

        dataset = data_class.dataset
        dataset_train, dataset_test = get_dataset_splits(dataset)

        indim = dataset_train["x"].shape[1]

        T = torch.FloatTensor
        vccat = lambda t, x, y: torch.cat(
            (T(t).view(-1, 1), T(x), T(y).view(-1, 1)), dim=1
        ).to(dtype=torch.float64)

        train_matrix = vccat(dataset_train["d"], dataset_train["x"], dataset_train["y"])
        test_matrix = vccat(dataset_test["d"], dataset_test["x"], dataset_test["y"])

        return train_matrix, test_matrix, None, indim, data_class

    elif dataset_name == constants.TCGA_SINGLE_2:
        cu.set_seed(dataset_num * 100)

        dataset_params = dict()
        dataset_params["num_treatments"] = 3
        dataset_params["treatment_selection_bias"] = 2
        dataset_params["dosage_selection_bias"] = 2
        dataset_params["save_dataset"] = True
        dataset_params["test_fraction"] = 0.2
        data_class = TCGA_Data(
            args=dataset_params, dataset_num=dataset_num, treatment_type=2
        )

        dataset = data_class.dataset
        dataset_train, dataset_test = get_dataset_splits(dataset)

        indim = dataset_train["x"].shape[1]

        T = torch.FloatTensor
        vccat = lambda t, x, y: torch.cat(
            (T(t).view(-1, 1), T(x), T(y).view(-1, 1)), dim=1
        ).to(dtype=torch.float64)

        train_matrix = vccat(dataset_train["d"], dataset_train["x"], dataset_train["y"])
        test_matrix = vccat(dataset_test["d"], dataset_test["x"], dataset_test["y"])

        return train_matrix, test_matrix, None, indim, data_class


def t_x_y(dataset_name, x_batch, t_batch):
    y_star = []
    if dataset_name in [constants.IHDP_CONT]:
        for x, t in zip(x_batch, t_batch):
            h_max = 1.0
            cate_idx1 = torch.tensor([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            cate_mean1 = torch.tensor(0.2923)
            alpha = 5.0

            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[4]
            x5 = x[5]

            # v1
            factor1 = 0.5
            factor2 = 1.5

            # v2
            factor1 = 1.5
            factor2 = 0.5

            # original
            factor1 = 1.0
            factor2 = 1.0

            # y = 1./(1.2 - t) * torch.sin(t * 2. * 3.14159) * (factor1 * torch.tanh((torch.sum(x[cate_idx1])/10. - cate_mean1) * alpha) +
            #                                                  factor2 * 0.2 * torch.exp(0.2 * (x1 - x5))/(0.1 + min(x2, x3, x4)))
            y_item = (
                1.0
                / (1.2 - t / h_max)
                * torch.sin((t / h_max) * 3.0 * 3.14159)
                * (
                    factor1
                    * torch.tanh((torch.sum(x[cate_idx1]) / 10.0 - cate_mean1) * alpha)
                    + factor2 * torch.exp(0.2 * (x1 - x5)) / (0.1 + min(x2, x3, x4))
                )
            )
            y_star.append(y_item)

        return torch.Tensor(y_star)

    elif dataset_name in [constants.NEWS_CONT]:
        import numpy as np

        num_feature = 498
        np.random.seed(5)
        v1 = np.random.randn(num_feature)
        v1 = torch.Tensor(v1 / np.sqrt(np.sum(v1**2))).to(cu.get_device())
        v2 = np.random.randn(num_feature)
        v2 = torch.Tensor(v2 / np.sqrt(np.sum(v2**2))).to(cu.get_device())
        v3 = np.random.randn(num_feature)
        v3 = torch.Tensor(v3 / np.sqrt(np.sum(v3**2))).to(cu.get_device())
        h_max = 1.0
        for x, t in zip(x_batch, t_batch):
            # return 10. * (np.sum(v1 * x) + 5. * np.sin(3.14159 * np.sum(v2 * x) / np.sum(v3 * x) * t))
            res1 = max(
                -2,
                min(
                    2,
                    torch.exp(
                        0.3
                        * (
                            torch.sum(3.14159 * torch.sum(v2 * x) / torch.sum(v3 * x))
                            - 1
                        )
                    ),
                ),
            )
            res2 = 20.0 * (torch.sum(v1 * x))
            res = (
                2
                * (4 * (t / h_max - 0.5) ** 2 * torch.sin(0.5 * 3.14159 * t / h_max))
                * (res1 + res2)
            )
            y_star.append(res)

        return torch.Tensor(y_star)

    else:
        assert False


def t_x_y_vector(dataset_name, x_batch, t_batch):
    if dataset_name in [constants.IHDP_CONT]:
        h_max = 1.0
        cate_idx1 = torch.tensor([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        cate_mean1 = torch.tensor(0.2923)
        alpha = 5.0

        x1 = x_batch[:, 0]
        x2 = x_batch[:, 1]
        x3 = x_batch[:, 2]
        x4 = x_batch[:, 4]
        x5 = x_batch[:, 5]

        # v1
        factor1 = 0.5
        factor2 = 1.5

        # v2
        factor1 = 1.5
        factor2 = 0.5

        # original
        factor1 = 1.0
        factor2 = 1.0

        # y = 1./(1.2 - t) * torch.sin(t * 2. * 3.14159) * (factor1 * torch.tanh((torch.sum(x[cate_idx1])/10. - cate_mean1) * alpha) +
        #                                                  factor2 * 0.2 * torch.exp(0.2 * (x1 - x5))/(0.1 + min(x2, x3, x4)))

        term1 = 1.0 / (1.2 - t_batch / h_max)
        term1 = torch.mul(term1, torch.sin((t_batch / h_max) * 3.0 * 3.14159))
        term2 = factor1 * torch.tanh(
            (torch.sum(x_batch[:, cate_idx1], dim=1) / 10.0 - cate_mean1) * alpha
        )
        term3 = torch.minimum(x2, x3)
        term3 = torch.minimum(term3, x4)
        term4 = term2 + torch.div((factor2 * torch.exp(0.2 * (x1 - x5))), 0.1 + term3)
        y_star = torch.mul(term1, term4)

        # y_star = 1. / (1.2 - t_batch/h_max) * torch.sin((t_batch/h_max) * 3. * 3.14159) * (
        #            factor1 * torch.tanh((torch.sum(x_batch[:,cate_idx1], dim=1) / 10. - cate_mean1) * alpha) +
        #            factor2 * torch.exp(0.2 * (x1 - x5)) / (0.1 + min(x2, x3, x4)))

        return y_star

    elif dataset_name in [constants.NEWS_CONT]:
        import numpy as np

        num_feature = 498
        np.random.seed(5)
        v1 = np.random.randn(num_feature)
        v1 = torch.Tensor(v1 / np.sqrt(np.sum(v1**2))).to(cu.get_device())
        v2 = np.random.randn(num_feature)
        v2 = torch.Tensor(v2 / np.sqrt(np.sum(v2**2))).to(cu.get_device())
        v3 = np.random.randn(num_feature)
        v3 = torch.Tensor(v3 / np.sqrt(np.sum(v3**2))).to(cu.get_device())

        h_max = 1.0

        # return 10. * (np.sum(v1 * x) + 5. * np.sin(3.14159 * np.sum(v2 * x) / np.sum(v3 * x) * t))
        temp1 = torch.exp(
            0.3
            * (
                3.14159
                * torch.div(
                    torch.sum(v2 * x_batch, dim=1), torch.sum(v3 * x_batch, dim=1)
                )
                - 1
            )
        )
        temp1 = torch.nan_to_num(temp1)
        # res1 = torch.clamp( torch.exp(0.3 * (3.14159 * torch.div(torch.sum(v2 * x_batch, dim=1) , torch.sum(v3 * x_batch, dim=1)) - 1)) , min=-2  , max=2   )
        # bmax= torch.Tensor([2]).to(cu.get_device())
        # bmin= torch.Tensor([-2]).to(cu.get_device())
        res1 = torch.where(temp1 > 2, 2, temp1)
        res1 = torch.where(res1 < -2, -2, res1)
        # res1 = torch.min(temp1, bmax.expand_as(res1))
        res2 = 20.0 * (torch.sum(v1 * x_batch, dim=1))
        y_star = (
            2
            * (
                4
                * torch.pow((t_batch / h_max - 0.5), 2)
                * torch.sin(0.5 * 3.14159 * t_batch / h_max)
            )
            * (res1 + res2)
        )

        return y_star

    else:
        assert False
