# PairNet: Training with Observed Pairs to Estimate Individual Treatment Effect
Lokesh Nagalapatti, Pranava Singhal, Avishek Ghosh, Sunita Sarawagi

- Arxiv: https://arxiv.org/abs/2406.03864
- Accepted at ICML'24
- The code is adapted from the [CATENets](https://github.com/AliciaCurth/CATENets) library for Binary Treatment Effect Estimation. We extend its support for continuous treatments too.

# To install catenets, use the following commands:
```
conda create -n catenets numpy pandas python=3.9 scipy 
pip3 install torch
pip install jax
pip install jaxlib
pip install loguru
pip install pytest scikit_learn
pip install gdown
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install ott-jax
```
---

# Binary Experiments:

## IHDP
```
python run_experiments_benchmarks_NeurIPS.py --experiment ihdp --file_name pairnet --n_exp 100
```

## ACIC
```
python run_experiments_benchmarks_NeurIPS.py --experiment acic --file_name v2_pairnet --n_exp 10 --simu_num 2
``` 
use the above command for version 2


```
python run_experiments_benchmarks_NeurIPS.py --experiment acic --file_name v7_pairnet --n_exp 10 --simu_num 7
``` 
use the above command for version 7


```
python run_experiments_benchmarks_NeurIPS.py --experiment acic --file_name v26_pairnet --n_exp 10 --simu_num 26
```

use the above command for version 27

## Twins
```
python run_experiments_inductive_bias_NeurIPS.py --setup twins --file_name pairnet --n_exp 5
```

# Continuous experiments

We provide the continuous datasets in the following [datasets.zip](https://drive.google.com/file/d/1zQOVxesVg2WsKTA440HzTKbM6dXkJ2Na/view?usp=drivesdk) file.

To run the continuous experiments, you first need to run the baseline methods to dump the representation embeddings as follows. Edit the ```model_dict``` dictionary to specify which baselines to run amongst VCNet and DRNetC. Modify the ```dataset_names``` dictionary to select which datasets to perform experiments on.
```
python run_experiments_continuous.py --file_name baselines --dump_reps True
```
subsequently, run the PairNet methods by editing ```model_dict``` again to choose the desired continuous methods amongst VCNetPairNet and DRNetPairNet
```
python run_experiments_continuous.py --file_name pairnet --dump_reps False
```
to configure hyperparameters, edit [catenets/models/cont_constants.py](catenets/models/cont_constants.py). All results in paper are reported with default HyperParameters. Some important parameters are:
```
DEFAULT_STEP_SIZE = 1e-3 # learning rate
DEFAULT_PENALTY_L2 = 1 # L2 penalty for representation heads
# PairNet specific parameters
DEFAULT_X_DISTANCE_DELTA = 0.1 # 
DEFAULT_NUM_NEIGHBOURS = 3 #
DEFAULT_SAMPLING_TEMPERATURE = 1 # set 0 for uniform sampling
DEFAULT_STATIC_PHI = True # set False for dynamic phi
```
Modify these parameters before running ```run_experiments_continuous.py``` to perform ablation experiments.

---

# Ablations
The code for executing the sensitivity analysis experiments, as detailed in our paper, can be found in the file [ihdp_ablations.py](ihdp_ablations.py).

- $\delta_{\text{pair}}$: This parameter controls the exclusion of distant pairs during the pairing process. The default value in PairNet is 0.1.
- $\text{num}_{z'}$: It specifies the number of neighbors to be selected for pairing each example. The default setting in PairNet is 3.
- $\phi_{\text{fct}}$: This parameter determines the type of representations used for computing the embeddings. PairNet typically defaults to using representations obtained from a factual model, which is often TARNet.

> To ensure the availability of the $\phi$, it is important to run the factual models first, and then subsequently run PairNet so that it can use them to create pairs.

## $L_2$ penalty
> CATENets initially applies an $L_2$ penalty to the model weights with a default value of 1e-4. Howeve we observed that increasing this penalty can significantly improve the performance of representation learning-based baselines. We found this adjustment to be particularly crucial for the ACIC dataset, especially in the case of seed 2, where the Individual Treatment Effect (ITE) remains constant across individuals.


To run experiment with catenets default value for $L_2$ penalty, change the `DEFAULT_PENALTY_L2` in [constants.py](catenets/models/constants.py) file to 1e-4 for the baselines.
for `PairNet`, change the value in `model_hypers` dictionary in the corresponding files: [ihdp](experiments/experiments_benchmarks_NeurIPS21/ihdp_experiments_catenets.py), [acic](experiments/experiments_benchmarks_NeurIPS21/acic_experiments_catenets.py), [twins](experiments/experiments_inductivebias_NeurIPS21/experiments_twins.py)

---

# Generating the main table

For reproducing the results featured in the main table, which also involve conducting the $p$-test, you can refer to the code available in the provided  [Notebook](collate_csvs.ipynb). In the case of ACIC and Twins datasets, the initial step involves aggregating the CSV files generated from various configurations into a single CSV file. This consolidated CSV file is then utilized to perform the $p$-tests.

<hr style="border:2px solid red">

# CATENets - Conditional Average Treatment Effect Estimation Using Neural Networks

[![CATENets Tests](https://github.com/AliciaCurth/CATENets/actions/workflows/test.yml/badge.svg)](https://github.com/AliciaCurth/CATENets/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/catenets/badge/?version=latest)](https://catenets.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/AliciaCurth/CATENets/blob/main/LICENSE)


Code Author: Alicia Curth (amc253@cam.ac.uk)

This repo contains Jax-based, sklearn-style implementations of Neural Network-based Conditional
Average Treatment Effect (CATE) Estimators, which were used in the AISTATS21 paper
['Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning
Algorithms']( https://arxiv.org/abs/2101.10943) (Curth & vd Schaar, 2021a) as well as the follow up
NeurIPS21 paper ["On Inductive Biases for Heterogeneous Treatment Effect Estimation"](https://arxiv.org/abs/2106.03765) (Curth & vd
Schaar, 2021b) and the NeurIPS21 Datasets & Benchmarks track paper ["Really Doing Great at Estimating CATE? A Critical Look at ML Benchmarking Practices in Treatment Effect Estimation"](https://openreview.net/forum?id=FQLzQqGEAH) (Curth et al, 2021).

We implement the SNet-class we introduce in Curth & vd Schaar (2021a), as well as FlexTENet and
OffsetNet as discussed in Curth & vd Schaar (2021b), and re-implement a number of
NN-based algorithms from existing literature (Shalit et al (2017), Shi et al (2019), Hassanpour
& Greiner (2020)). We also provide Neural Network (NN)-based instantiations of a number of so-called
meta-learners for CATE estimation, including two-step pseudo-outcome regression estimators (the
DR-learner (Kennedy, 2020) and single-robust propensity-weighted (PW) and regression-adjusted (RA) learners), Nie & Wager (2017)'s R-learner and Kuenzel et al (2019)'s X-learner. The jax implementations in ``catenets.models.jax`` were used in all papers listed; additionally, pytorch versions of some models (``catenets.models.torch``) were contributed by [Bogdan Cebere](https://github.com/bcebere).

### Interface
The repo contains a package ``catenets``, which contains all general code used for modeling and evaluation, and a folder ``experiments``, in which the code for replicating experimental results is contained. All implemented learning algorithms in ``catenets`` (``SNet, FlexTENet, OffsetNet, TNet, SNet1 (TARNet), SNet2
(DragonNet), SNet3, DRNet, RANet, PWNet, RNet, XNet``) come with a sklearn-style wrapper,  implementing a ``.fit(X, y, w)`` and a ``.predict(X)`` method, where predict returns CATE by default. All hyperparameters are documented in detail in the respective files in ``catenets.models`` folder.

Example usage:

```python
from catenets.models.jax import TNet, SNet
from catenets.experiment_utils.simulation_utils import simulate_treatment_setup

# simulate some data (here: unconfounded, 10 prognostic variables and 5 predictive variables)
X, y, w, p, cate = simulate_treatment_setup(n=2000, n_o=10, n_t=5, n_c=0)

# estimate CATE using TNet
t = TNet()
t.fit(X, y, w)
cate_pred_t = t.predict(X)  # without potential outcomes
cate_pred_t, po0_pred_t, po1_pred_t = t.predict(X, return_po=True)  # predict potential outcomes too

# estimate CATE using SNet
s = SNet(penalty_orthogonal=0.01)
s.fit(X, y, w)
cate_pred_s = s.predict(X)

```

All experiments in Curth & vd Schaar (2021a) can be replicated using this repository; the necessary
code is in ``experiments.experiments_AISTATS21``. To do so from shell, clone the repo, create a new
virtual environment and run
```shell
pip install catenets # install the library from PyPI
# OR
pip install . # install the library from the local repository

# Run the experiments
python run_experiments_AISTATS.py
```
```shell
Options:
--experiment # defaults to 'simulation', 'ihdp' will run ihdp experiments
--setting # different simulation settings in synthetic experiments (can be 1-5)
--models # defaults to None which will train all models considered in paper,
         # can be string of model name (e.g 'TNet'), 'plug' for all plugin models,
         # 'pseudo' for all pseudo-outcome regression models

--file_name # base file name to write to, defaults to 'results'
--n_repeats # number of experiments to run for each configuration, defaults to 10 (should be set to 100 for IHDP)
```

Similarly, the experiments in Curth & vd Schaar (2021b) can be replicated using the code in
``experiments.experiments_inductivebias_NeurIPS21`` (or from shell using ```python
run_experiments_inductive_bias_NeurIPS.py```) and the experiments in Curth et al (2021) can be replicated using the code in ``experiments.experiments_benchmarks_NeurIPS21`` (the catenets experiments can also be run from shell using ``python run_experiments_benchmarks_NeurIPS``).

The code can also be installed as a python package (``catenets``). From a local copy of the repo, run ``python setup.py install``.

Note: jax is currently only supported on macOS and linux, but can be run from windows using WSL (the windows subsystem for linux).


### Citing

If you use this software please cite the corresponding paper(s):

```
@inproceedings{curth2021nonparametric,
  title={Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms},
  author={Curth, Alicia and van der Schaar, Mihaela},
    year={2021},
  booktitle={Proceedings of the 24th International Conference on Artificial
  Intelligence and Statistics (AISTATS)},
  organization={PMLR}
}

@article{curth2021inductive,
  title={On Inductive Biases for Heterogeneous Treatment Effect Estimation},
  author={Curth, Alicia and van der Schaar, Mihaela},
  booktitle={Proceedings of the Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}


@article{curth2021really,
  title={Really Doing Great at Estimating CATE? A Critical Look at ML Benchmarking Practices in Treatment Effect Estimation},
  author={Curth, Alicia and Svensson, David and Weatherall, James and van der Schaar, Mihaela},
  booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  year={2021}
}

```
