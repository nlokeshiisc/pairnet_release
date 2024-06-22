import os
import continuous.src.config as config

config = config.config
import continuous.src.constants as constants
import continuous.utils.common_utils as cu
import continuous.src.main_helper as main_helper

from pathlib import Path

this_dir = Path(".").absolute()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import argparse
import pickle as pkl


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", help="Use the correct argument", default="continuous/src/config.py"
)

config = cu.parse_args(parser)

if __name__ == "__main__":
    num_epochs = config[constants.EPOCHS]
    dataset_name = config[constants.DATASET]
    dataset_nums = config[constants.DATASET_NUM]
    enforce_diff = config[constants.ENFORCE_DIFF]
    file_suffix = config[constants.FILE_SUFFIX]

    baseline_args = config[constants.BASELINE_ARGS]
    diff_args = config[constants.DIFF_ARGS]

    if enforce_diff:
        main_helper.run_diff(
            dataset_name=dataset_name,
            dataset_nums=dataset_nums,
            num_epochs=num_epochs,
            suffix=file_suffix,
            **diff_args,
        )

    else:
        main_helper.run_baselines(
            dataset_name=dataset_name,
            dataset_nums=dataset_nums,
            num_epochs=num_epochs,
            suffix=file_suffix,
            **baseline_args,
        )

# Unit test for find_nbr_ids
# import continuous.utils.icte_utils as iu
# iu.unit_test_find_nbr_ids()