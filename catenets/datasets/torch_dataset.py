import torch
import numpy as np
from torch.utils.data import Subset
import typing
import matplotlib.pyplot as plt

from ott.geometry import pointcloud
from ott.geometry import geometry
from ott.solvers.linear import sinkhorn
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod


def flatten_cfs(tensor):
    """Flatteens the middle dimension of a 3D tensor

    Args:
        tensor (_type_): _description_
    """
    if tensor.ndim == 2:
        return tensor
    assert tensor.ndim == 3 or tensor.ndim == 4, "Only 3D/4D tensors supported"
    return tensor.view(-1, tensor.shape[-1])


class BaseTorchDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, X, beta, y, xemb, **kwargs):
        self.X = torch.Tensor(X)
        self.beta = torch.Tensor(beta).view(-1, 1)
        self.y = torch.Tensor(y).view(-1, 1)
        self.Xemb = torch.Tensor(xemb)
        self.bad_indices = []

        self.emb_dim = self.Xemb.shape[-1]

        self.one_idxs = torch.where(self.beta == 1)[0]
        self.zero_idxs = torch.where(self.beta == 0)[0]

        assert len(xemb) == len(X), "xemb and x should be of same length"

        self.det = None
        self.num_cfz = None
        self.sm_temp = None
        self.dist_metric = None
        self.pcs_dist = None
        self.drop_frac = None
        self.OT = None  # This is for optimal transport
        self.__init_kwargs(**kwargs)

        # Preprocess the distances
        self.Xemb = (
            torch.nn.functional.normalize(self.Xemb, dim=1)
            if self.dist_metric == "cos"
            else self.Xemb
        )

        if self.OT == False:
            self.nearest_nbrs()
        else:
            self.nearest_nbrs_OT()

        if self.check_perex_contrib == True:
            self.check_perex_contribs()
            
        # self.check_pair_var_cate()

    @property
    def _Xemb(self):
        return self.Xemb
    
    @_Xemb.setter
    def _Xemb(self, Xemb):
        if not isinstance(Xemb, torch.Tensor):
            Xemb = torch.Tensor(Xemb)
        assert len(Xemb) == len(self.Xemb), "Xemb and self.Xemb should be of same length"
        
        self.Xemb = Xemb
        self.Xemb = (
            torch.nn.functional.normalize(self.Xemb, dim=1)
            if self.dist_metric == "cos"
            else self.Xemb
        )

        if self.OT == False:
            self.nearest_nbrs()
        else:
            self.nearest_nbrs_OT()
    
    def nearest_nbrs(self):
        D = torch.cdist(self.Xemb, self.Xemb, p=2)
        # Suppress self selection
        D[torch.eye(len(D)).bool()] = 1e10

        # If this is true, we drop the \delta_{pair} distances
        if self.pcs_dist == True:
            """Assume that we have 5 items and we need to drop 10% pairs.
            - We have 5*5 = 25 distances above
            - we need to exclude self pairs, i.e. 5 pairs
            - To account for symmetry, we drop 20*10%=20% pairs which is 0.2*20+5=4+9 pairs
            """
            num_pairs = D.shape[0] ** 2
            drop_num = int( 2 * self.drop_frac * (num_pairs - D.shape[0]) ) + D.shape[0]
            drop_thr = torch.topk(D.view(-1), drop_num)[0][-1]
            D[D > drop_thr] = 1e10

        self.one_probs = torch.softmax(-D[:, self.one_idxs] / self.sm_temp, dim=1)
        self.zero_probs = torch.softmax(-D[:, self.zero_idxs] / self.sm_temp, dim=1)

        num_pairs = min(10, min(len(self.one_idxs), len(self.zero_idxs)) - 1)
        self.cf_pairs = [
            self.create_cf_idx(idx, num_pairs=num_pairs) for idx in range(len(self))
        ]

    def nearest_nbrs_OT(self):
        """This creates pairs based on optimal transport.
        """
        
        # create anm OT geometry
        geom = pointcloud.PointCloud(
                self.Xemb[self.zero_idxs].cpu().numpy(),
                self.Xemb[self.one_idxs].cpu().numpy(),
            )

        if self.pcs_dist == True:
            M = geom.cost_matrix
            drop_thr = np.percentile(M, (1 - self.drop_frac) * 100)

        solve_fn = jax.jit(sinkhorn.solve)
        sh = solve_fn(geom, a=None, b=None)
        
        # Gs is a transport plan of shape n_0 X n_1. Each row sums to 1/n_0 and each column sums to 1/n_1
        Gs = np.asarray(sh.matrix).copy() # copy reqd to make mutable
        Gs = torch.Tensor(Gs)

        # init the probability for each node to be paired with the control group
        self.zero_probs = torch.ones(len(self), len(self.zero_idxs)) * torch.nan
        # Do the same for the treated group
        self.one_probs = torch.ones(len(self), len(self.one_idxs)) * torch.nan
        
        # To get the probability of treated group pairing with control grp, we just need to scale each column with n_1
        zero_probs = Gs.T * (len(self.one_idxs))
        one_probs = Gs * (len(self.zero_idxs))

        self.zero_probs[self.one_idxs] = zero_probs
        self.one_probs[self.zero_idxs] = one_probs
        
        # create a cache of 10 pairs for each item, and then in each minibatch randomly sample num_z' entries from them.
        num_pairs = min(10, min(len(self.one_idxs), len(self.zero_idxs)) - 1)
        self.cf_pairs = [
            self.create_cf_idx(idx, num_pairs=num_pairs) for idx in range(len(self))
        ]
        pass

    def __init_kwargs(self, **kwargs):
        self.det = kwargs.get(
            "det", False
        )  # should we sample the counerfactuals deterministically?
        self.num_cfz = kwargs.get("num_cfz", 1)
        self.sm_temp = kwargs.get(
            "sm_temp", 1.0
        )  # the softmax temparature while creating the pairs
        self.dist_metric = kwargs.get("dist", "euc")
        self.pcs_dist = kwargs.get("pcs_dist", False)
        self.drop_frac = kwargs.get("drop_frac", None)
        assert self.drop_frac < 1, "drop_frac should be a fraction"

        self.check_perex_contrib = kwargs.get("check_perex_contrib", False)
        self.i_exp = kwargs.get("i_exp", 0)
        self.OT = kwargs.get("OT", False)

    def __str__(self):
        return f"AgrDS: {len(self)} samples, {self.emb_dim} dim emb, {self.num_cfz} cfz, \
                {self.dist_metric} dist, {self.sm_temp} sm_temp, {self.drop_frac} drop_dist, \
                {self.det} det, {self.pcs_dist} pcs_dist)"

    def check_perex_contribs(self):
        """Tracks the count of number of times each example occurs in the cf_pairs

        Returns:
            _type_: _description_
        """
        self.perex_contribs = torch.zeros(len(self))
        for item_dict in self.cf_pairs:
            self.perex_contribs[item_dict["nids"]] += 1
        plt.clf()
        plt.cla()
        plt.bar(range(len(self)), self.perex_contribs.cpu().numpy())
        plt.text(0, 0, f"Mean: {self.perex_contribs.mean().item()}", color="red")
        plt.text(0, 5, f"Std: {self.perex_contribs.std().item()}", color="red")
        plt.text(0, 10, f"Min: {self.perex_contribs.min().item()}", color="red")
        plt.text(0, 15, f"Max: {self.perex_contribs.max().item()}", color="red")
        plt.text(0, 20, f"cntrl: {len(self.zero_idxs) / len(self)}", color="red")
        plt.title(f"Nbr rep. for seed {self.i_exp}")
        plt.tight_layout()
        plt.savefig(f"results/plots/seed_{self.i_exp}.png", dpi=300)
        
    def check_pair_var_cate(self):
        """Checks the variance in the difference of y's across pairs"""
        pair_diffs = []
        for idx in range(len(self)):
            y = self.y[idx].view(-1, 1)
            b = self.beta[idx].view(-1, 1)
            yps = self.cf_pairs[idx]["yp"]
            pair_diffs.append((y - yps) * (2 * b - 1))
        pair_diffs = torch.cat(pair_diffs, dim=0)
        print(f"Pair diff variance: {pair_diffs.var().item()}")
        print(f"pair diff mean: {pair_diffs.mean().item()}")
        pass

    def __len__(self):
        return len(self.X)

    @abstractmethod
    def create_cf_idx(self, idx, num_pairs):
        """Creates a cache for the countrefactual set to be used while training.
        For each (z, b, y), we do the following:
            sample 10 (zp, bp, yp) from bp != b
            For each sampled (zp, bp, yp), sample 1 (zpe, bpe, ype) from bpp = bp
        """
        assert False, "Sub-class must provide the implementation for this function"
        
    def get_near_ids(self, idx, b, num_near, arbitrary_pairs=False):
        """Returns the indices of the nearest neighbors of the given idx with in the b group"""
        if arbitrary_pairs is False:
            p = self.zero_probs[idx] if b == 0 else self.one_probs[idx]
            assert not torch.isnan(p).any(), "p should not have any nan"
        else:
            # Sample from a uniform distribution
            p = (
                torch.ones_like(self.zero_probs[idx])
                if b == 0
                else torch.ones_like(self.one_probs[idx])
            )
            p = p / p.sum()

        if self.OT == False:
            # For ot such far-off pairs anyways won't be selected
            num_near = min(num_near, torch.nonzero(p).shape[0])

            if num_near == 0:
                # KNOWN: If there is no legal pair, select an arbitrary pair
                nids = torch.randperm(len(p))[:1].view(-1, 1)
                self.bad_indices.append(idx)
            else:
                nids = (
                    torch.topk(p, num_near, largest=True)[1]
                    if self.det
                    else torch.LongTensor(
                        np.random.choice(len(p), num_near, p=p.cpu().numpy(), replace=False)
                    )
                )
        elif self.OT == True:
            eps = 1e-12
            p = (p + eps) / (p + eps).sum()
            # print("near_p", p)
            nids =  torch.LongTensor(
                        np.random.choice(len(p), num_near, p=p.cpu().numpy(), replace=False)
                    )
        near_probs = p[nids] / p[nids].sum()
        nids = self.zero_idxs[nids] if b == 0 else self.one_idxs[nids]
        return nids, near_probs

    def __getitem__(self, idx):
        assert False, "This class is deprecated"

class PairDataset(BaseTorchDataset):
    def __init__(self, X, beta, y, xemb, **kwargs):
        self.arbitrary_pairs = kwargs.get("arbitrary_pairs", False)
        if self.arbitrary_pairs == True:
            print("Using arbitrary pairs")
        super().__init__(X, beta, y, xemb, **kwargs)

    def create_cf_idx(self, idx, num_pairs):
        """Creates a cache for the countreffactual set to be used while training.
        For each (z, b, y), we do the following:
            sample 10 (zp, bp, yp) from bp != b
        """
        x, b, y = self.X[idx], self.beta[idx], self.y[idx]
        xp, bp, yp = [], [], []

        nids = []

        # Sample bp
        # torch.multinomial seems inappropriate here. https://discuss.pytorch.org/t/trying-to-understand-the-torch-multinomial/71643
        nids, near_probs = self.get_near_ids(
            idx, 1 - b, num_near=num_pairs, arbitrary_pairs=self.arbitrary_pairs
        )

        selfids = torch.LongTensor([idx] * len(nids))

        x = self.X[selfids]
        xp = self.X[nids]

        b = self.beta[selfids]
        bp = self.beta[nids]

        y = self.y[selfids]
        yp = self.y[nids]

        nids = torch.LongTensor(nids)

        return {
            "x": x,
            "b": b,
            "y": y,
            "xp": xp,
            "bp": bp,
            "yp": yp,
            "nids": nids,
            "near_probs": near_probs,
        }

    def __getitem__(self, idx):
        item_dict = self.cf_pairs[idx]
        num_pairs = len(item_dict["x"])
        near_probs = item_dict["near_probs"]
        assert len(near_probs) == num_pairs, "near_probs and num_pairs should be same"
        # sample_ids = torch.LongTensor(np.random.choice(num_pairs, self.num_cfz, p=near_probs.cpu().numpy(), replace=False))
        sample_ids = torch.randperm(num_pairs)[: self.num_cfz]

        return {
            "x": item_dict["x"][sample_ids],
            "b": item_dict["b"][sample_ids],
            "y": item_dict["y"][sample_ids],
            "xp": item_dict["xp"][sample_ids],
            "bp": item_dict["bp"][sample_ids],
            "yp": item_dict["yp"][sample_ids],
            "nids": item_dict["nids"][sample_ids],
        }


def test_agreement_ds():
    # X = torch.repeat_interleave(torch.zeros(2).view(1, -1), 4, dim=0)
    # y = torch.arange(1, 5).view(-1, 1)
    # beta = torch.Tensor([0, 0, 1, 1]).view(-1, 1)
    # xemb = X.clone()

    # ds_args = {
    #     "det": True,
    #     "pcs_dist": False,
    #     "sm_temp": 1.0,
    #     "drop_frac": 0.0,
    # }
    # agr_ds = AgreementDataset(X=X, beta=beta, y=y, xemb=xemb, num_cfz=1, **ds_args)
    # print(agr_ds)

    import pandas as pd

    # dicts = [agr_ds[i] for i in range(len(agr_ds))]
    # df = pd.DataFrame(dicts)
    # df.to_csv("test_agreement_ds.csv")

    X = torch.randn(96, 2)
    beta = torch.Tensor([0, 0, 0, 0, 0, 0, 1, 1] * 12).view(-1, 1)
    y = torch.arange(len(X)).view(-1, 1)
    xemb = X.clone()
    ds_args = {
        "det": False,
        "pcs_dist": True,
        "sm_temp": 0.5,
        "drop_frac": 0.5,
        "num_cfz": 2,
    }
    agr_ds = BaseTorchDataset(X=X, beta=beta, y=y, xemb=xemb, **ds_args)
    df = pd.DataFrame([agr_ds[i] for i in range(len(agr_ds))])
    df.to_csv("test_agreement_ds2.csv")


if __name__ == "__main__":
    test_agreement_ds()
    print("Test passed!")
