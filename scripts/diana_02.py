"""Contains code to replicate photos from HCP dataset"""
import glob
import logging
import multiprocessing
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform, distance_transform_edt
from skimage.transform import resize

from ext.hg_utils import zoom
from ext.lab2im import utils
from ext.utils import (
    get_git_revision_branch,
    get_git_revision_short_hash,
    get_git_revision_url,
)

# TODO: fix this function
# def get_min_max_idx(t2_file):
#     # 7. Open the T2
#     t2_vol = utils.load_volume(t2_file)

#     # scaling the entire volume
#     t2_vol = 255 * t2_vol / np.max(t2_vol)

#     N = np.zeros(len(spacings))
#     for n in range(len(spacings)):
#         N(n) = minimum used slice index for spacing (skip) n
#         s1= np.max(N)

#     for n in range(len(spacings)):
#         N(n) = maximum used slice index for spacing (skip) n
#         s1= np.min(N)


def get_nonzero_slice_ids(t2_vol):
    """find all non-zero slices"""
    slice_sum = np.sum(t2_vol, axis=(0, 1))
    return np.where(slice_sum > 0)[0]


def slice_ids_method1(args, t2_vol):
    """current method of selecting slices
    Example:
    slice_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    slices:    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,  1,  0,  0,  0]
    selected:  [0, 0, 0, 0, 1, 0, 1, 0, 1, 0,  1,  0,  0,  0] (skip = 2)
    selected:  [0, 0, 0, 1, 0, 0, 1, 0, 0, 1,  0,  0,  0,  0] (skip = 3)
    """
    non_zero_slice_ids = get_nonzero_slice_ids(t2_vol)

    first_nz_slice = non_zero_slice_ids[0]
    slice_ids_of_interest = np.where(non_zero_slice_ids % args["SKIP"] == 0)
    slice_ids_of_interest = slice_ids_of_interest[0] + first_nz_slice

    return slice_ids_of_interest


def slice_ids_method2(args, t2_vol):
    """method: skipping from start
    Example:
    slice_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    slices:    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,  1,  0,  0,  0]
    selected:  [0, 0, 0, 1, 0, 1, 0, 1, 0, 1,  0,  0,  0,  0] (skip = 2)
    selected:  [0, 0, 0, 1, 0, 0, 1, 0, 0, 1,  0,  0,  0,  0] (skip = 3)
    """
    # another method: skip 5 non zero slices on either end (good)
    non_zero_slice_ids = get_nonzero_slice_ids(t2_vol)

    first_nz_slice = non_zero_slice_ids[0] + 6
    last_nz_slice = non_zero_slice_ids[-1] - 6

    slice_ids_of_interest = np.arange(first_nz_slice, last_nz_slice + 1, args["SKIP"])
    return slice_ids_of_interest


def process_t1(args, t1_file, t1_name):
    """_summary_

    Args:
        args (_type_): _description_
        t1_file (_type_): _description_
        t1_name (_type_): _description_
    """
    # 1. Sample 3 rotations about the 3 axes, e.g., between -30 and 30 degrees.
    rotation = np.random.randint(-30, 31, 3)

    # 2. Sample 3 translations along the 3 axes, e.g., between 20 and 20 mm
    translation = np.random.randint(-20, 21, 3)

    # 3. Build a rigid 3D rotation + translation (4x4) matrix using the rotations and shifts
    t1_rigid_mat = utils.create_affine_transformation_matrix(
        3,
        scaling=None,
        rotation=rotation,
        shearing=None,
        translation=translation,
    )

    # t1_rigid_mat = np.eye(t1_rigid_mat.shape[0])

    t1_rigid_out = os.path.join(args["out_dir"], t1_name, f"{t1_name}.rigid.npy")
    np.save(t1_rigid_out, t1_rigid_mat)

    # 4. Open the T1, and premultiply the affine matrix of the header
    # (“vox2ras”) by the matrix from 3.
    volume, aff, hdr = utils.load_volume(t1_file, im_only=False)
    new_aff = np.matmul(t1_rigid_mat, aff)
    hdr.set_sform(new_aff)

    # 5. Binarize the T1 volume by thresholding at 0 and save it with the
    # new header, and call it “mri.mask.mgz”
    mask = volume > 0

    M = mask.astype("bool")
    # positive is outside, negative is inside
    D = distance_transform_edt(~M) - distance_transform_edt(M)
    Rsmall = np.random.uniform(low=-1.5, high=1.5, size=(5, 5, 5))
    R = resize(Rsmall, M.shape, order=3)
    mask = D < R

    t1_out_path = os.path.join(args["out_dir"], t1_name, f"{t1_name}.mri.mask.mgz")
    utils.save_volume(mask, new_aff, hdr, t1_out_path)


def create_slice_affine(affine_dir, t2_name, idx, curr_slice):
    """_summary_

    Args:
        idx (_type_): _description_
        curr_slice (_type_): _description_
    """
    # Sample a rotation eg between -20 and 20 degrees
    rotation = np.random.randint(-20, 21, 1)

    # Sample 2 translations along the 2 axes, eg, between -0.5 and 0.5 pixels
    translation = np.random.uniform(-0.5, 0.5, 2)

    # Sample 2 small shears about the 2 axes (eg between -0.1 and 0.1)
    shearing = np.random.uniform(-0.1, 0.1, 2)

    # Build a 2D (3x3) matrix with the rotation, translations, and shears
    translation_mat_1 = np.array(
        [
            [1, 0, -0.5 * curr_slice.shape[0]],
            [0, 1, -0.5 * curr_slice.shape[1]],
            [0, 0, 1],
        ]
    ).astype(float)
    translation_mat_2 = np.array(
        [
            [1, 0, 0.5 * curr_slice.shape[0]],
            [0, 1, 0.5 * curr_slice.shape[1]],
            [0, 0, 1],
        ]
    ).astype(float)
    aff_mat = utils.create_affine_transformation_matrix(
        2,
        scaling=None,
        rotation=rotation,
        shearing=shearing,
        translation=translation,
    )
    # to rotate around the center of the slice instead of the corner
    slice_aff_mat = np.matmul(translation_mat_2, np.matmul(aff_mat, translation_mat_1))

    # Save this matrix somewhere for evaluation later on eg as a numpy array
    slice_aff_out = os.path.join(affine_dir, f"{t2_name}.slice.{idx:03d}.npy")
    np.save(slice_aff_out, slice_aff_mat)

    return slice_aff_mat


def make_mask_from_deformed(photo_dir, t2_name, idx, deformed_slice):
    """_summary_

    Args:
        photo_dir (_type_): _description_
        t2_name (_type_): _description_
        idx (_type_): _description_
        deformed_slice (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Threshold the deformed slice at zero to get a mask (1 inside, 0 outside)
    mask = deformed_slice > 0

    M = mask.astype("bool")
    D = distance_transform_edt(~M) - distance_transform_edt(
        M
    )  # where positive is outside, negative is inside
    Rsmall = np.random.uniform(low=-1.5, high=1.5, size=(5, 5))
    R = resize(Rsmall, M.shape, order=3)
    mask = D < R

    # write it as photo_dir/image.[c].npy
    # (format the number c with 2 digits so they are in order when listed)
    out_file_name = os.path.join(photo_dir, f"{t2_name}.image.{idx:03d}.npy")
    np.save(out_file_name, mask)

    return mask


def create_corrupted_image(photo_dir, t2_name, idx, deformed_slice):
    """_summary_

    Args:
        photo_dir (_type_): _description_
        t2_name (_type_): _description_
        idx (_type_): _description_
        deformed_slice (_type_): _description_
        mask (_type_): _description_
    """

    mask = make_mask_from_deformed(photo_dir, t2_name, idx, deformed_slice)

    # add illumination field to the slice
    # Sample a random zero-mean gaussian tensor of size (5x5)
    # and multiply by a small standard deviation (eg 0.1)
    small_vol = 0.1 * np.random.normal(size=(5, 5))

    # Upscale the tensor to the size of the slice
    # edit_volumes.resample_volume(small)
    factors = np.divide(mask.shape, small_vol.shape)

    # pixel-wise exponential of the upsampled tensor to get an illumination field
    bias_result = zoom.scipy_zoom(small_vol, factors, mask.shape)

    # Multiply the deformed slice by the illumination field
    corrupted_image = np.multiply(deformed_slice, bias_result)

    # Write the corrupted image to photo_dir/image.[c].tif
    img_out = os.path.join(photo_dir, f"{t2_name}.image.{idx:03d}.png")
    corrupted_image = Image.fromarray(np.uint8(corrupted_image))
    corrupted_image.save(img_out, "PNG")


def slice_jitter(t2_vol, slice_id, jitter):
    """_summary_

    Args:
        t2_name (_type_): _description_
        jitter (_type_): _description_
        t2_vol (_type_): _description_
        slice_id (_type_): _description_
    """
    while True:
        rand_idx = random.randrange(-jitter, jitter)
        curr_slice = t2_vol[..., slice_id + rand_idx]

        if np.sum(curr_slice):
            return curr_slice


def process_t2(args, t2_file, t2_name):
    """_summary_

    Args:
        args (_type_): _description_
        t2_file (_type_): _description_
        t2_name (_type_): _description_
        jitter (int, optional): _description_. Defaults to 0.
    """
    jitter = args["jitter"]
    affine_dir = os.path.join(args["out_dir"], t2_name, "slice_affines")
    photo_dir = os.path.join(args["out_dir"], t2_name, "photo_dir")

    # 6. Create a directory “photo_dir"
    os.makedirs(affine_dir, exist_ok=True)
    os.makedirs(photo_dir, exist_ok=True)

    # 7. Open the T2 (and scale)
    t2_vol = utils.load_volume(t2_file)
    t2_vol = 255 * t2_vol / np.max(t2_vol)

    slice_ids = slice_ids_method2(args, t2_vol)  # see method 2

    for idx, slice_id in enumerate(slice_ids, 1):
        curr_slice = t2_vol[..., slice_id]

        if jitter:
            curr_slice = slice_jitter(t2_vol, slice_id, jitter)

        # NOTE: (in hindsight) this rotation was a bad idea
        curr_slice = np.pad(np.rot90(curr_slice), 25)

        slice_aff_mat = create_slice_affine(affine_dir, t2_name, idx, curr_slice)

        # Use this matrix to deform the slice
        deformed_slice = affine_transform(
            curr_slice, slice_aff_mat, mode="constant", order=1
        )

        create_corrupted_image(photo_dir, t2_name, idx, deformed_slice)


def pipeline(args, t1_file, t2_file):
    """_summary_

    Args:
        args (_type_): _description_
        t1_file (_type_): _description_
        t2_file (_type_): _description_
        jitter (int, optional): _description_. Defaults to 0.
    """
    # get file name
    t1_fname = os.path.split(t1_file)[1]
    t2_fname = os.path.split(t2_file)[1]

    # get subject ID
    t1_subject_name = t1_fname.split(".")[0]
    t2_subject_name = t2_fname.split(".")[0]

    assert t1_subject_name == t2_subject_name, "Incorrect Subject Name"

    # make output directory for subject
    out_subject_dir = os.path.join(args["out_dir"], t1_subject_name)

    if not os.path.isdir(out_subject_dir):
        os.makedirs(out_subject_dir, exist_ok=True)

    # create symlinks to source files (T1, T2)
    t1_dst = os.path.join(out_subject_dir, t1_fname)
    t2_dst = os.path.join(out_subject_dir, t2_fname)

    if not os.path.exists(t1_dst):
        os.symlink(t1_file, t1_dst)

    if not os.path.exists(t2_dst):
        os.symlink(t2_file, t2_dst)

    # work on T1 and T2 volumes
    process_t1(args, t1_file, t1_subject_name)
    process_t2(args, t2_file, t2_subject_name)


def get_t1_t2_pairs(args):
    """_summary_

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    t1_files = utils.list_files(args["in_dir"], True, "T1.nii.gz")
    t2_files = utils.list_files(args["in_dir"], True, "T2.nii.gz")

    assert len(t1_files) == len(t2_files), "Subject Mismatch"

    return list(zip(t1_files, t2_files))


def submit_pipeline(args):
    """_summary_

    Args:
        args (_type_): _description_
        jitter (int, optional): _description_. Defaults to 0.

    Raises:
        Exception: _description_
    """
    t1_t2_pairs = get_t1_t2_pairs(args)
    file_count = len(t1_t2_pairs)

    input_ids = np.random.choice(range(file_count), file_count, replace=False)
    input_ids = input_ids[:100]

    n_procs = 1 if args["DEBUG"] else multiprocessing.cpu_count()

    with Pool(processes=n_procs) as pool:
        pool.starmap(
            pipeline,
            [(args, *t1_t2_pairs[idx]) for idx in input_ids],
        )


def make_specific_args(args, skip, jitter):
    """_summary_

    Args:
        skip (_type_): _description_
        jitter (_type_): _description_

    Returns:
        _type_: _description_
    """

    results_dir = args["results_dir"]
    out_dir = os.path.join(results_dir, f"skip-{skip:02d}-r{jitter}")

    os.makedirs(out_dir, exist_ok=True)

    args["out_dir"] = out_dir
    args["SKIP"] = skip
    args["jitter"] = jitter

    return args


def info_logger(args):
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(args["results_dir"], "log.txt"),
        filemode="a",
        format="%(asctime)s - %(message)s",
    )

    logging.info(os.path.basename(__file__))
    logging.info(f"The Git Branch is: {get_git_revision_branch()}")
    logging.info(f"The Git Commit is: {get_git_revision_short_hash()}")
    # log the git url
    logging.info(f"The Git URL is: {get_git_revision_url()}")


def make_main_args():
    """Specify project specific directory paths and other parameters

    Returns:
        dict: dictionary of project specific parameters
    """
    PRJCT_DIR = "/space/calico/1/users/Harsha/photo-reconstruction"
    in_dir = os.path.join(PRJCT_DIR, "data/4diana_hcp_left_hemi")
    results_dir = os.path.join(PRJCT_DIR, "results/4diana-hcp-recons")

    os.makedirs(results_dir, exist_ok=True)

    gettrace = getattr(sys, "gettrace", None)
    DEBUG = True if gettrace() else False

    return dict(in_dir=in_dir, results_dir=results_dir, DEBUG=DEBUG)


def main():
    """Simulate photos (from T2) and mask from (T1) of th HCP dataset"""
    args = make_main_args()
    info_logger(args)

    print("t1_rigid_mat is not set to identity")
    print("slice extraction using method 2")
    MIN_SKIP, MAX_SKIP = 14, 14
    MIN_JITTER, MAX_JITTER = 0, 0

    # for skip in range(MIN_SKIP, MAX_SKIP + 1, 2):
    for skip in [14]:
        for jitter in range(MIN_JITTER, MAX_JITTER + 1):
            np.random.seed(0)  # reset seed for reproducibility
            logging.info(f"Running Skip {skip:02d}, Jitter {jitter}")

            args = make_specific_args(args, skip, jitter)
            submit_pipeline(args)


if __name__ == "__main__":
    main()
