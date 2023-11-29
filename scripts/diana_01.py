import glob
import os
import random
import sys

import numpy as np
import pandas as pd

PRJCT_DIR = "/space/calico/1/users/Harsha/photo-reconstruction"
DATA_DIR = PRJCT_DIR + "/data"
HCP_DIR = DATA_DIR + "/4harshaHCP"
LEFT_MASK = DATA_DIR + "/chris_LEFTmask.nii.gz"
HCP_HEMI_DIR = DATA_DIR + "/4diana_hcp_left_hemi"


def list_t1_files():
    t1_files = sorted(glob.glob(os.path.join(HCP_DIR, "*.T1.nii.gz")))
    return t1_files


def list_t2_files():
    t2_files = sorted(glob.glob(os.path.join(HCP_DIR, "*.T2.nii.gz")))
    return t2_files


def sample_subjects(t1_files, t2_files, n=100):
    select_samples = random.sample(list(zip(t1_files, t2_files)), n)
    return select_samples


def apply_left_mask(samples):
    for sample in samples:
        for file in sample:
            file_name = os.path.basename(file)

            out_file = os.path.join(HCP_HEMI_DIR, file_name)
            command = f"mri_mask {file} {LEFT_MASK} {out_file}"

            os.system(command)
        print()


def main():
    t1_files = list_t1_files()
    t2_files = list_t2_files()
    select_samples = sample_subjects(t1_files, t2_files, 100)
    os.makedirs(HCP_HEMI_DIR, exist_ok=True)
    apply_left_mask(select_samples)


if __name__ == "__main__":
    main()