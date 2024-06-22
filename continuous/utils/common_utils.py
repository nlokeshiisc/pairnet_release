import random
import torch
import numpy as np
from copy import deepcopy
import json
from pathlib import Path
import os



def set_seed(seed: int = 42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return "cpu"


def set_cuda_device(gpu_num: int):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)


def insert_kwargs(kwargs: dict, new_args: dict):
    assert type(new_args) == type(kwargs), "Please pass two dictionaries"
    merged_args = kwargs.copy()
    merged_args.update(new_args)
    return merged_args


def dict_print(d: dict):
    d_new = deepcopy(d)

    def cast_str(d_new: dict):
        for k, v in d_new.items():
            if isinstance(v, dict):
                d_new[k] = cast_str(v)
            d_new[k] = str(v)
        return d_new

    d_new = cast_str(d_new)

    pretty_str = json.dumps(d_new, sort_keys=False, indent=4)
    return pretty_str


def config_to_command(config: dict):
    cmd = ""

    for k, v in config.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                cmd += f"\t {k}.{k1}:{v1}"
        else:
            cmd += f"\t {k}:{v}"
    return cmd
