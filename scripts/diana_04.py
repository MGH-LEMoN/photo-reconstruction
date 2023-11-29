import glob
import json
import multiprocessing
import os
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from PIL import Image
from scipy.ndimage import map_coordinates

from ext.lab2im import utils

rcParams.update({"figure.autolayout": True})


sns.set(
    style="whitegrid",
    rc={
        "text.usetex": True,
        "font.family": "serif",
    },
)

# CUSTOM = 'subject_133'

SOME_SUFFIX = ""
# {curtailed | identity | test | | divisible} }

# SOME_SUFFIX_LIST = ["divisible"]
SOME_SUFFIX_LIST = ["curtailed", "identity"]


def get_nonzero_slice_ids(t2_vol):
    """find all non-zero slices"""
    slice_sum = np.sum(t2_vol, axis=(0, 1))
    return np.where(slice_sum > 0)[0]


# write function to filter a list of numbers to only those that are divisible by
# all numbers in the list simultaneously
def filter_divisible_by_all(numbers, divisors):
    """_summary_

    Args:
        numbers (_type_): _description_
        divisors (_type_): _description_
    """
    return [n for n in numbers if all(n % d == 0 for d in divisors)]


def slice_ids_method1(skip, t2_vol):
    """current method of selecting slices
    Example:
    slice_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    slices:    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,  1,  0,  0,  0]
    selected:  [0, 0, 0, 0, 1, 0, 1, 0, 1, 0,  1,  0,  0,  0] (skip = 2)
    selected:  [0, 0, 0, 1, 0, 0, 1, 0, 0, 1,  0,  0,  0,  0] (skip = 3)
    """
    non_zero_slice_ids = get_nonzero_slice_ids(t2_vol)

    first_nz_slice = non_zero_slice_ids[0]
    slice_ids_of_interest = np.where(non_zero_slice_ids % skip == 0)
    slice_ids_of_interest = slice_ids_of_interest[0] + first_nz_slice

    return slice_ids_of_interest


def slice_ids_method2(skip, t2_vol):
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

    slice_ids_of_interest = np.arange(first_nz_slice, last_nz_slice + 1, skip)
    return slice_ids_of_interest


def get_middle_elements(test_list, K):
    # using list slicing
    return test_list[
        int(len(test_list) / 2) - int(K / 2) : int(len(test_list) / 2) + int(K / 2) + 1
    ]


def get_error_optimized(results_dir):
    """_summary_

    Args:
        skip (_type_): _description_
        jitter (_type_): _description_
        sub_id (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    PAD = 3  # for reconstruction

    print(os.path.basename(results_dir))

    skip = get_skip(results_dir)
    sub_id = os.path.basename(results_dir).split("_")[-1]

    rigid_transform = np.load(utils.list_files(results_dir, True, "rigid.npy")[0])

    photo_affines = utils.list_files(
        os.path.join(results_dir, "slice_affines"), True, "npy"
    )
    photo_affine_matrix = np.stack([np.load(item) for item in photo_affines], axis=2)

    recon_matrix = np.load(
        os.path.join(
            results_dir, f"ref_soft_mask_skip_{skip:02d}", "slice_matrix_M.npy"
        )
    )
    recon_matrix = recon_matrix[
        :, :, PAD:-PAD
    ]  # removing matrices corresponding to padding
    recon_matrix = recon_matrix[
        :, [0, 1, 3], :
    ]  # excluding rotation/translation in z-axis
    recon_matrix = recon_matrix[[0, 1, 3], :, :]

    if SOME_SUFFIX == "identity":
        num_repeats = recon_matrix.shape[-1]
        recon_matrix = np.concatenate([np.eye(3)[..., None]] * num_repeats, axis=-1)

    all_paddings = np.load(
        os.path.join(results_dir, f"ref_soft_mask_skip_{skip:02d}", "all_paddings.npy")
    )

    assert (
        photo_affine_matrix.shape[-1] == recon_matrix.shape[-1]
    ), "Slice count mismatch"

    t1_path = utils.list_files(results_dir, True, "T1.nii.gz")[0]
    t2_path = utils.list_files(results_dir, True, "T2.nii.gz")[0]

    _, t1_aff, t1_hdr = utils.load_volume(t1_path, im_only=False)
    t2_vol, _, _ = utils.load_volume(t2_path, im_only=False)

    recon = os.path.join(
        results_dir, f"ref_soft_mask_skip_{skip:02d}", "photo_recon.mgz"
    )
    _, recon_aff, recon_hdr = utils.load_volume(recon, im_only=False)

    error_norms_slices = []
    error_norms_slices_xy = []
    error_norms_slices_z = []
    slice_ids_of_interest = slice_ids_method2(skip, t2_vol)

    # # get the number of slices in it's highest spacing counterpart
    # # (for equal comparison with the reconstruction)
    if SOME_SUFFIX == "curtailed":
        high_subject_dir = results_dir.replace(f"skip-{skip:02d}", "skip-14")
        num_slices = len(
            utils.list_files(
                os.path.join(high_subject_dir, "slice_affines"), False, ".npy"
            )
        )
        keep_slice_list = get_middle_elements(slice_ids_of_interest, num_slices)

    if SOME_SUFFIX == "divisible":
        divisors = [2, 3, 4, 6, 8]
        keep_slice_list = filter_divisible_by_all(slice_ids_of_interest, divisors)

    # creating empty arrays for error norms (volumes)
    errors_vol = np.zeros_like(t2_vol)
    errors_vol_xy = np.zeros_like(t2_vol)
    errors_vol_z = np.zeros_like(t2_vol)

    for z, slice_id in enumerate(slice_ids_of_interest):
        if SOME_SUFFIX == "curtailed" or SOME_SUFFIX == "divisible":
            if slice_id not in keep_slice_list:
                continue

        # skipping the first slice
        if z == 0:
            continue

        curr_slice = t2_vol[..., slice_id]
        io, jo = np.where(curr_slice > 0)

        # load D1
        D1_path = utils.list_files(
            os.path.join(results_dir, "niftreg_outputs"), True, "D1.nii.gz"
        )[0]
        D1 = utils.load_volume(D1_path, im_only=True)

        # get D1 at v_woz coordinates
        D1_at_gt = D1[io, jo, slice_id, :]

        # rotate and pad
        i = curr_slice.shape[1] + 25 - jo - 1
        j = io + 25

        v = np.stack([i, j, np.ones(i.shape)])
        v2 = np.matmul(np.linalg.inv(photo_affine_matrix[:, :, z]), v)

        zp = len(all_paddings) - z - 1

        P = np.eye(3)
        P[:-1, -1] = all_paddings[zp]

        v3 = np.matmul(P, v2)
        v4 = np.matmul(np.linalg.inv(recon_matrix[:, :, zp]), v3)

        v4_3d = np.stack([v4[0, :], v4[1, :], (zp + PAD) * v4[-1, :], v4[-1, :]])

        ras_new = np.matmul(recon_aff, v4_3d)

        # v4_3d coordinates of gt in recon space
        # applying vox2ras gives ras_new
        # applying inverse vox2ras of synthsr

        synthsr_file = utils.list_files(results_dir, True, "SynthSR")[0]
        _, synthsr_aff, _ = utils.load_volume(synthsr_file, im_only=False)
        synthsr_coords = np.matmul(np.linalg.inv(synthsr_aff), ras_new)

        # getting D2 at those coordinates
        D2_path = utils.list_files(
            os.path.join(results_dir, "niftreg_outputs"), True, "D2.nii.gz"
        )[0]
        D2 = utils.load_volume(D2_path, im_only=True)

        D2_at_gt = np.stack(
            [
                map_coordinates(D2[..., i], synthsr_coords[:3], order=1)
                for i in range(3)
            ],
            -1,
        )

        errors_slice = D1_at_gt - D2_at_gt

        error_norms_slice = np.sqrt(np.sum(errors_slice**2, axis=1))
        error_norms_slice_xy = np.sqrt(np.sum(errors_slice[:, [0, -1]] ** 2, axis=1))
        error_norms_slice_z = np.abs(errors_slice[:, 2])

        error_norms_slices.append(error_norms_slice)
        error_norms_slices_xy.append(error_norms_slice_xy)
        error_norms_slices_z.append(error_norms_slice_z)

        # putting errors in a volume
        errors_vol[io, jo, slice_id] = error_norms_slice
        errors_vol_xy[io, jo, slice_id] = error_norms_slice_xy
        errors_vol_z[io, jo, slice_id] = error_norms_slice_z

    os.makedirs(
        os.path.join(results_dir, "-".join(["error_vols", SOME_SUFFIX]).strip("-")),
        exist_ok=True,
    )

    utils.save_volume(
        errors_vol,
        t1_aff,
        t1_hdr,
        os.path.join(
            results_dir,
            "-".join(["error_vols", SOME_SUFFIX]).strip("-"),
            "errors_xyz.mgz",
        ),
    )
    utils.save_volume(
        errors_vol_xy,
        t1_aff,
        t1_hdr,
        os.path.join(
            results_dir,
            "-".join(["error_vols", SOME_SUFFIX]).strip("-"),
            "errors_xy.mgz",
        ),
    )
    utils.save_volume(
        errors_vol_z,
        t1_aff,
        t1_hdr,
        os.path.join(
            results_dir,
            "-".join(["error_vols", SOME_SUFFIX]).strip("-"),
            "errors_z.mgz",
        ),
    )

    return (
        sub_id,
        (
            error_norms_slices,
            np.nanmean(np.concatenate(error_norms_slices)),
            np.nanstd(np.concatenate(error_norms_slices)),
            np.nanmedian(np.concatenate(error_norms_slices)),
        ),
        (
            error_norms_slices_xy,
            np.nanmean(np.concatenate(error_norms_slices_xy)),
            np.nanstd(np.concatenate(error_norms_slices_xy)),
            np.nanmedian(np.concatenate(error_norms_slices_xy)),
        ),
        (
            error_norms_slices_z,
            np.nanmean(np.concatenate(error_norms_slices_z)),
            np.nanstd(np.concatenate(error_norms_slices_z)),
            np.nanmedian(np.concatenate(error_norms_slices_z)),
        ),
    )


def save_errors(results_dir, corr, idx):
    """_summary_

    Args:
        skip (_type_): _description_
        jitter (_type_): _description_
        corr (_type_): _description_
    """
    idx_flag = [None, "all", "xy", "z"]
    out_strings = ["errors", "means", "stds", "medians"]

    jitter_val = get_jitter(results_dir)
    skip_val = get_skip(results_dir)

    file_suffix = f"skip-{skip_val:02d}-r{jitter_val}"

    head_dir = os.path.join(
        os.path.dirname(results_dir),
        "-".join(["hcp-errors", SOME_SUFFIX]).strip("-"),
        idx_flag[idx],
    )
    os.makedirs(head_dir, exist_ok=True)

    # FIXME: clean this function
    subject_ids = [int(item[0]) for item in corr]

    for str_idx, out_string in enumerate(out_strings):
        all_vals = [item[idx][str_idx] for item in corr]
        all_vals = dict(zip(subject_ids, all_vals))

        if str_idx == 0:
            np.save(os.path.join(head_dir, f"hcp-errors-{file_suffix}"), all_vals)

            all_mean_of_means = dict()
            for k, v in all_vals.items():
                try:
                    all_mean_of_means[k] = np.mean([np.mean(slice) for slice in v])
                except:
                    all_mean_of_means[k] = None

            with open(
                os.path.join(head_dir, f"hcp-mean-of-means-{file_suffix}"), "w"
            ) as write_file:
                json.dump(all_mean_of_means, write_file, indent=4)

        else:
            with open(
                os.path.join(head_dir, f"hcp-{out_string}-{file_suffix}"),
                "w",
            ) as write_file:
                json.dump(all_vals, write_file, indent=4)


def subject_selector(results_dir, n_size=None):
    """_summary_

    Args:
        skip (_type_): _description_
        jitter (_type_): _description_

    Returns:
        _type_: _description_
    """
    file_count = len(utils.list_subfolders(results_dir))

    subject_ids = np.random.choice(range(file_count), n_size, replace=False)
    # subjects = [item for item in subjects if CUSTOM in item]
    return subject_ids


def get_error_wrapper(sub_id, results_dir):
    """_summary_

    Args:
        sub_id (_type_): _description_
        skip (_type_): _description_
        jitter (_type_): _description_

    Returns:
        _type_: _description_
    """
    subject_dir = utils.list_subfolders(results_dir, True, "subject_")[sub_id]

    try:
        return get_error_optimized(subject_dir)
    except:
        subject_id = os.path.basename(subject_dir)
        print(f"Failed: {subject_id}")
        return (
            subject_id.split("_")[-1],
            (None, None, None, None),
            (None, None, None, None),
            (None, None, None, None),
        )


def main_mp(results_dir, sample_size=None):
    """_summary_

    Args:
            skip (_type_): _description_
            jitter (_type_): _description_
    """
    subjects = subject_selector(results_dir, sample_size)

    gettrace = getattr(sys, "gettrace", None)
    n_procs = 1 if gettrace() else multiprocessing.cpu_count()

    with Pool(processes=n_procs) as pool:
        corr = pool.starmap(
            get_error_wrapper, [(subject, results_dir) for subject in subjects]
        )

    save_errors(results_dir, corr, 1)
    save_errors(results_dir, corr, 2)
    save_errors(results_dir, corr, 3)


def get_jitter(dir_name):
    """_summary_

    Args:
        dir_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    return int(dir_name[-1])


def get_skip(file_path):
    """_summary_

    Args:
        file_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(file_path, str):
        return int(file_path.split("-")[-2])
    else:
        return sorted(list({int(item.split("-")[-2]) for item in file_path}))


def some_function(dir_path, error_type="means"):
    # TODO: name this function
    file_list = utils.list_files(dir_path, True, "hcp-" + error_type + "-skip")

    r3_means = [pd.read_json(file, orient="index") for file in file_list]

    col_item1 = lambda x: np.round(np.array(get_skip(x)) * 0.7, 1)
    columns = [str(col_item1(file)) + "_" + str(get_jitter(file)) for file in file_list]

    r3_mean_df = pd.concat(r3_means, axis=1)
    print(f"Before removing NaN rows: {r3_mean_df.shape[0]}")

    r3_mean_df = r3_mean_df.dropna(axis=0, how="any")
    print(f"After removing NaN rows: {r3_mean_df.shape[0]}")

    if r3_mean_df.empty:
        return None

    r3_mean_df.columns = columns

    return r3_mean_df


def get_just_means_for_filtering(dir_path):
    r3_mean_df = some_function(dir_path)
    return r3_mean_df["1.4_0"] > 0.8


def get_means_and_stds(dir_path, error_type="mean"):
    """_summary_

    Args:
        jitter (_type_): _description_

    Returns:
        _type_: _description_
    """
    r3_mean_df = some_function(dir_path, error_type)

    # filter subjects on the two clusters found in means
    # r3_mean_df = r3_mean_df[get_just_means_for_filtering(dir_path)]

    r3_mean_df = r3_mean_df.stack().reset_index().drop(labels=["level_0"], axis=1)
    r3_mean_df.columns = ["Skip", f"{error_type.capitalize()} Error"]

    r3_mean_df[["Spacing", "Jitter"]] = pd.DataFrame(
        r3_mean_df["Skip"].str.split("_").tolist()
    )
    r3_mean_df["Spacing"] = r3_mean_df["Spacing"].astype(float)
    r3_mean_df = r3_mean_df.drop(labels=["Skip"], axis=1)

    return r3_mean_df


def make_error_boxplot(data_frame, out_file, x_col, y_col, hue_col):
    """_summary_

    Args:
        df (_type_): _description_
        out_file (_type_): _description_
        x_col (_type_): _description_
        hue_col (_type_): _description_
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    bp_ax = sns.boxplot(
        x=x_col,
        y=y_col,
        hue=hue_col,
        data=data_frame,
        palette="viridis",
        linewidth=0.5,
    )
    # fig = bp_ax.get_figure()
    # lgnd = bp_ax.legend(ncol=10, edgecolor="white", framealpha=0.25, )
    # lgnd = plt.legend(frameon=False, fontsize=13)
    ylabel = (
        bp_ax.get_ylabel()
        .replace("Means", "Mean")
        .replace("Medians", "Median")
        .replace("Stds", "Std. Dev")
    )

    bp_ax.set_xlabel(xlabel=bp_ax.get_xlabel(), fontweight="bold").set_fontsize("15")
    bp_ax.set_ylabel(ylabel=ylabel, fontweight="bold").set_fontsize("15")

    fig.savefig(out_file, dpi=1200, bbox_inches="tight")
    plt.clf()


def plot_file_name(results_dir, type, plot_idx):
    """_summary_

    Args:
        results_dir (_type_): _description_
        plot_idx (_type_): _description_
    """
    out_file = f"{SOME_SUFFIX}-{type}_{plot_idx:02d}.png".strip("-")
    out_file = os.path.join(results_dir, out_file)
    # os.makedirs(os.path.dirname(out_file), exist_ok=True)
    return out_file


def plot_registration_error(results_dir, error_str="mean"):
    """_summary_

    Args:
        jitters (_type_): _description_
    """
    # for sub_folder in ["all", "xy", "z"]:
    for sub_folder in ["all"]:
        errors_dir = os.path.join(
            results_dir,
            "-".join(["hcp-errors", SOME_SUFFIX]).strip("-"),
            sub_folder,
        )

        final_df = get_means_and_stds(errors_dir, error_str)

        if final_df is not None:
            y_col = f"{error_str.capitalize()} Error"

            out_file = plot_file_name(errors_dir, error_str, 1)
            make_error_boxplot(final_df, out_file, "Spacing", y_col, "Jitter")

            out_file = plot_file_name(errors_dir, error_str, 2)
            make_error_boxplot(final_df, out_file, "Jitter", y_col, "Spacing")
        else:
            print(f"Empty DataFrame for {error_str}")


def calculate_registration_error(results_dir, n_subjects=None):
    """_summary_"""
    recon_folders = utils.list_subfolders(results_dir, True, "skip-")

    for recon_folder in recon_folders:
        if "ex" in os.path.basename(recon_folder):
            continue
        np.random.seed(0)
        print(os.path.basename(recon_folder))
        main_mp(recon_folder, n_subjects)


def collect_images_into_pdf(results_dir):
    """[summary]
    Args:
        target_dir_str ([str]): string relative to RESULTS_DIR
    """
    errors_dir = os.path.join(
        results_dir, "-".join(["hcp-errors", SOME_SUFFIX]).strip("-")
    )

    model_dirs = sorted(os.listdir(errors_dir))

    for model_dir in model_dirs:
        out_file = os.path.join(errors_dir, f"{model_dir}_plots-{SOME_SUFFIX}.pdf")
        model_dir = os.path.join(errors_dir, model_dir)

        pdf_img_list = []
        images = sorted(glob.glob(os.path.join(model_dir, "*.png")))

        for image in images:
            img = Image.open(image)
            img = img.convert("RGB")
            pdf_img_list.append(img)

        pdf_img_list[0].save(out_file, save_all=True, append_images=pdf_img_list[1:])

    return


if __name__ == "__main__":
    PRJCT_DIR = "/space/calico/1/users/Harsha/photo-reconstruction/results"

    FOLDER = "4diana-hcp-recons"

    # set this to a high value if you want to run all subjects
    # there are nearly 897 subjects in the dataset
    M = 100

    full_results_path = os.path.join(PRJCT_DIR, FOLDER)

    if not os.path.exists(full_results_path):
        raise Exception("Folder does not exist")

    calculate_registration_error(full_results_path, M)

    # for stat_key in ["means", "stds", "medians", "mean-of-means"]:
    #     plot_registration_error(full_results_path, stat_key)

    # collect_images_into_pdf(full_results_path)
