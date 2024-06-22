"""
Define some constants for initialisation of hyperparamters etc
"""
import numpy as np

# default model architectures
DEFAULT_LAYERS_OUT = 2
DEFAULT_LAYERS_OUT_T = 2
DEFAULT_LAYERS_R = 3
DEFAULT_LAYERS_R_T = 3

DEFAULT_UNITS_OUT = 100
DEFAULT_UNITS_R = 200
DEFAULT_UNITS_OUT_T = 100
DEFAULT_UNITS_R_T = 200

DEFAULT_NONLIN = "elu"

# other default hyperparameters
DEFAULT_STEP_SIZE = 1e-4
DEFAULT_STEP_SIZE_T = 0.0001
DEFAULT_N_ITER = 1000
DEFAULT_BATCH_SIZE = 100
DEFAULT_PENALTY_L2 = 1
DEFAULT_PENALTY_DISC = 0
DEFAULT_PENALTY_ORTHOGONAL = 1 / 100
DEFAULT_AVG_OBJECTIVE = True


# defaults for early stopping
DEFAULT_VAL_SPLIT = 0.3
DEFAULT_N_ITER_MIN = 200
DEFAULT_PATIENCE = 10

# Defaults for crossfitting
DEFAULT_CF_FOLDS = 2

# other defaults
DEFAULT_SEED = 42
DEFAULT_N_ITER_PRINT = 50
LARGE_VAL = np.iinfo(np.int32).max

DEFAULT_UNITS_R_BIG_S = 100
DEFAULT_UNITS_R_SMALL_S = 50

DEFAULT_UNITS_R_BIG_S3 = 150
DEFAULT_UNITS_R_SMALL_S3 = 50

N_SUBSPACES = 3
DEFAULT_DIM_S_OUT = 50
DEFAULT_DIM_S_R = 100
DEFAULT_DIM_P_OUT = 50
DEFAULT_DIM_P_R = 100
