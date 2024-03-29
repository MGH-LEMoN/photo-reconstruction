import csv
import glob
import inspect
import os
import re
from distutils.dir_util import copy_tree

import numpy as np
from PIL import Image, ImageDraw
from tabulate import tabulate

import ext.my_functions as my

PRJCT_KEY = "uw_photo/Photo_data"
PRJCT_DIR = os.getenv("PYTHONPATH")
DATA_DIR = os.path.join(PRJCT_DIR, "data", PRJCT_KEY)
RESULTS_DIR = os.path.join(PRJCT_DIR, "results", PRJCT_KEY)


def grab_diff_photos(ref_string=None):
    """_summary_

    Args:
        ref_string (str): options include 'image', 'hard', 'soft'
    """
    prjct_key = "uw_photo/Photo_data"
    prjct_dir = os.getenv("PYTHONPATH")
    data_dir = os.path.join(prjct_dir, "data", prjct_key)
    results_dir = os.path.join(prjct_dir, "results")

    subject_list = sorted(glob.glob(os.path.join(data_dir, "*")))
    subject_list = [item for item in subject_list if os.path.isdir(item)]

    im_list = []
    dst_pdf = os.path.join(results_dir, "uw_photo_difference_images" + ".pdf")
    for ref_string in ["image", "hard", "soft"]:
        for skip_val in [1, 2, 3, 4]:
            for subject in subject_list:
                ref_folder_string = f"ref_{ref_string}_skip_{skip_val}"
                ref_folder = os.path.join(subject, ref_folder_string)
                diff_file = glob.glob(
                    os.path.join(
                        ref_folder, "propagated_labels", "*difference*"
                    )
                )

                if len(diff_file) == 0:
                    continue
                src_file = diff_file[0]

                print_text = os.path.basename(subject) + "_" + ref_folder_string
                image = Image.open(src_file)
                draw = ImageDraw.Draw(image)
                draw.text(
                    (image.size[0] // 2, 10),
                    print_text,
                    fill="white",
                    align="right",
                )
                # imagelist is the list with all image filenames
                im_list.append(image)

    im_list[0].save(
        dst_pdf,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=im_list[1:],
    )


def grab_diff_photos_main():
    """_summary_"""
    for ref in ["hard", "soft", "image"]:
        grab_diff_photos(ref)


def return_common_subjects(*args):
    """_summary_

    Returns:
        _type_: _description_
    """
    args = [
        {
            os.path.split(input_file)[-1][:7]: input_file
            for input_file in file_list
        }
        for file_list in args
    ]

    lst = [set(lst.keys()) for lst in args]

    # One-Liner to intersect a list of sets
    common_names = sorted(lst[0].intersection(*lst))

    args = [[lst[key] for key in common_names] for lst in args]

    return args


def get_first_non_empty_slice(volume_file):
    """_summary_

    Args:
        volume_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    volume = my.MRIread(volume_file, im_only=True)
    if len(volume.shape) == 4:
        non_zero_slice = np.min(np.where(volume[..., 0].sum(0).sum(0) > 0))
    else:
        non_zero_slice = np.argmax((volume > 1).sum(0).sum(0))
    return non_zero_slice


def print_slice_idx_all():
    """Print the index of the GT slice (excludes padding)
    Note: This info is relevant for simulating skips in reconstruction
    """
    global DATA_DIR
    henry_results = os.path.join(DATA_DIR, "recons/henry_results/")
    henry_results_hard = os.path.join(henry_results, "Results_hard")
    henry_results_soft = os.path.join(henry_results, "Results_soft")

    hard_subject_list = sorted(
        glob.glob(os.path.join(henry_results_hard, "*-*"))
    )
    soft_subject_list = sorted(
        glob.glob(os.path.join(henry_results_soft, "*-*"))
    )

    common_subject_list = return_common_subjects(
        hard_subject_list, soft_subject_list
    )

    subject_gt_idx = []
    for hard_subject, soft_subject in zip(*common_subject_list):
        subject_id = os.path.basename(hard_subject)

        # soft_seg_gt = f"soft/{subject_id}_manualLabel.mgz"
        soft_seg_unmerged_gt = f"soft/{subject_id}_soft_manualLabel.mgz"
        soft_seg_merged_gt = f"soft/{subject_id}_soft_manualLabel_merged.mgz"

        hard_seg_unmerged_str = f"{subject_id}_hard_manualLabel.mgz"
        hard_seg_merged_str = f"{subject_id}_hard_manualLabel_merged.mgz"

        hard_recon_str = f"{subject_id}.hard.recon.mgz"
        soft_recon_str = f"soft/{subject_id}_soft.mgz"

        hard_recon = os.path.join(hard_subject, hard_recon_str)
        soft_recon = os.path.join(soft_subject, soft_recon_str)

        ref_seg1 = os.path.join(hard_subject, hard_seg_unmerged_str)
        ref_seg2 = os.path.join(hard_subject, hard_seg_merged_str)

        # ref_seg3 = os.path.join(soft_subject, soft_seg_gt)
        ref_seg4 = os.path.join(soft_subject, soft_seg_unmerged_gt)
        ref_seg5 = os.path.join(soft_subject, soft_seg_merged_gt)

        subject_row = []
        for file in [
            hard_recon,
            ref_seg1,
            ref_seg2,
            soft_recon,
            ref_seg4,
            ref_seg5,
        ]:
            if os.path.isfile(file):
                subject_row.append(my.MRIread(file, im_only=True).shape)
                subject_row.append(get_first_non_empty_slice(file))

        # idx2 = get_slice_idx(ref_seg2, hard_recon)
        # idx1 = get_slice_idx(ref_seg1)

        # idx4 = get_slice_idx(ref_seg4)
        # idx5 = get_slice_idx(ref_seg5)

        # temp = [os.path.basename(hard_subject), idx1, idx2, idx4, idx5]
        # subject_gt_idx.append(temp)
        subject_gt_idx.append([subject_id] + subject_row)

    global RESULTS_DIR
    with open(os.path.join(RESULTS_DIR, "gt_slice_idx.txt"), "w") as f:
        f.write(
            tabulate(
                subject_gt_idx,
                headers=[
                    "subject",
                    "h_recon_shape",
                    "h_recon_nz",
                    "h_unmerge_shape",
                    "h_unmerge_nz_slice",
                    "h_merge_shape",
                    "h_merge_nz_slice",
                    "s_recon_shape",
                    "s_recon_nz",
                    "s_unmerge_shape",
                    "s_unmerge_nz_slice",
                    "s_merge_shape",
                    "s_merge_nz_slice",
                ],
                colalign="center",
            )
        )


def get_gt_slice_idx(seg_vol, recon_vol=None):
    """_summary_

    Args:
        seg_vol (_type_): _description_
        recon_vol (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    try:
        seg_vol = my.MRIread(seg_vol, im_only=True)

        # gt slice (includes padding)
        slice_idx = np.argmax((seg_vol > 1).sum(0).sum(0))

        if recon_vol is not None:
            recon_vol = my.MRIread(recon_vol, im_only=True)

            # index of first nonzero slice (or # of padded slices)
            slice_idx_recon = np.min(
                np.where(recon_vol[..., 0].sum(0).sum(0) > 1.0)
            )
            slice_idx -= slice_idx_recon
    except Exception:
        slice_idx = "DNE"

    return slice_idx


def print_uw_gt_map():
    """Print the index of the GT slice (excludes padding)
    Note: This info is relevant for simulating skips in reconstruction
    """
    global DATA_DIR
    henry_results = os.path.join(DATA_DIR, "recons/henry_results/")
    henry_results_hard = os.path.join(henry_results, "Results_hard")

    # list all folders/subjects
    hard_subject_list = sorted(
        glob.glob(os.path.join(henry_results_hard, "*-*"))
    )

    subject_gt_idx = []
    for hard_subject in hard_subject_list:
        # get subject_id
        subject_id = os.path.basename(hard_subject)

        # read reconstruction and merged segmentation volumes
        hard_seg_merged_str = f"{subject_id}_hard_manualLabel_merged.mgz"
        hard_recon_str = f"{subject_id}.hard.recon.mgz"

        hard_recon = os.path.join(hard_subject, hard_recon_str)
        hard_seg = os.path.join(hard_subject, hard_seg_merged_str)

        idx2 = get_gt_slice_idx(hard_seg, hard_recon)
        subject_gt_idx.append([subject_id, idx2])

    global RESULTS_DIR
    with open(os.path.join(RESULTS_DIR, "uw_gt_map.csv"), "w") as f:
        f.write(
            tabulate(
                subject_gt_idx,
                headers=[
                    "subject",
                    "gt_slice_idx",
                ],
            )
        )

    with open(os.path.join(RESULTS_DIR, "uw_gt_map.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(subject_gt_idx)


def get_hcp_subject_list():
    """_summary_"""
    t1_files = glob.glob(
        os.path.join(
            "/space/calico/1/users/Harsha/SynthSeg/data/4harshaHCP", "*.T1.*"
        )
    )
    t2_files = glob.glob(
        os.path.join(
            "/space/calico/1/users/Harsha/SynthSeg/data/4harshaHCP", "*.T2.*"
        )
    )

    t1_files = [
        re.findall("[0-9]+", os.path.basename(file))[0] for file in t1_files
    ]
    t2_files = [
        re.findall("[0-9]+", os.path.basename(file))[0] for file in t2_files
    ]

    common_subjects = sorted(set(t1_files).intersection(t2_files))

    with open(
        os.path.join(
            "/space/calico/1/users/Harsha/photo-reconstruction/results/hcp_subject_list.csv"
        ),
        "w",
    ) as f:
        f.write("\n".join(common_subjects))
        f.write("\n")

    return


def put_skip_recons_in_main_dir():
    """_summary_"""
    global DATA_DIR
    global RESULTS_DIR

    subject_list = sorted(glob.glob(os.path.join(RESULTS_DIR, "*-*")))
    subject_list = [item for item in subject_list if os.path.isdir(item)]

    for subject in subject_list:
        subject_id = os.path.basename(subject)
        print(subject_id)
        src = subject
        dst = os.path.join(DATA_DIR, "Photo_data", subject_id)
        copy_tree(src, dst)


def recon_ref_image():
    """python version of the make target 'recon_ref_image'"""
    prjct_dir = "/space/calico/1/users/Harsha/photo-reconstruction"
    OUT_DIR = f"{prjct_dir}/data/uw_photo/Photo_data"
    data_dir = f"{prjct_dir}/data/uw_photo"

    subjects = [
        "17-0333",
        "18-0086",
        "18-0444",
        "18-0817",
        "18-1045",
        "18-1132",
        "18-1196",
        "18-1274",
        "18-1327",
        "18-1343",
        "18-1470",
        "18-1680",
        "18-1690",
        "18-1704",
        "18-1705",
        "18-1724",
        "18-1754",
        "18-1913",
        "18-1930",
        "18-2056",
        "18-2128",
        "18-2259",
        "18-2260",
        "19-0019",
        "19-0037",
        "19-0100",
        "19-0138",
        "19-0148",
    ]

    subjects = ["19-0019"]
    for skip in range(1, 5):
        for p in subjects:
            command = f"python scripts/3d_photo_reconstruction.py \
            --input_photo_dir {data_dir}/Photo_data/{p}/{p}_MATLAB \
            --input_segmentation_dir {data_dir}/Photo_data/{p}/{p}_MATLAB \
            --ref_mask {data_dir}/FLAIR_Scan_Data/{p}.rotated_masked.mgz \
            --photos_of_posterior_side --allow_z_stretch --slice_thickness 4 --photo_resolution 0.1 \
            --output_directory {OUT_DIR}/{p}/ref_image_skip_{skip} \
            --gpu 0 \
            --skip  \
            --multiply_factor {skip}"

            os.system(command)


def hcp_recon(skip=6):
    """python version of the make target 'hcp_recon'"""
    prjct_dir = f"/space/calico/1/users/Harsha/SynthSeg/results/4harshaHCP-skip-{skip:02d}-r3"
    subjects = os.listdir(prjct_dir)

    for subject in subjects:
        input_dir = os.path.join(prjct_dir, subject)
        input_photo_dir = os.path.join(input_dir, "photo_dir")
        ref_mask = os.path.join(input_dir, f"{subject}.mri.mask.mgz")
        command = f"sbatch --job-name=$$subid \
        submit.sh scripts/3d_photo_reconstruction.py \
        --input_photo_dir {input_photo_dir} \
        --input_segmentation_dir {input_photo_dir} \
        --ref_mask {ref_mask} \
        --photos_of_posterior_side \
        --allow_z_stretch \
        --order_posterior_to_anterior \
        --slice_thickness 8.4 \
        --photo_resolution 0.7 \
        --output_directory {input_dir} \
        --gpu 0"

        print(command)


def hcp_recon_group(skip=6, n_jobs=5):
    """python version of the make target 'hcp_recon' to group multiple reconstructions at once"""
    prjct_dir = f"/space/calico/1/users/Harsha/SynthSeg/results/hcp-results/4harshaHCP-skip-{skip:02d}-r3"
    subjects = sorted(os.listdir(prjct_dir))
    log_dir = f"/space/calico/1/users/Harsha/photo-reconstruction/logs/hcp-recon/test-skip-{skip:02d}-r3"

    os.makedirs(log_dir, exist_ok=True)

    if not n_jobs:
        raise Exception("Number of jobs grouped cannot be None or Zero")

    for idx, ptr in enumerate(range(0, len(subjects) - n_jobs + 1, n_jobs)):
        script_file = os.path.join(
            os.getcwd(), "submit_scripts", f"submit_group_{idx:02d}.sh"
        )
        os.makedirs(os.path.dirname(script_file), exist_ok=True)
        batch = subjects[ptr : ptr + n_jobs]

        all_commands = []
        for subject in batch:
            log_file = os.path.join(log_dir, subject + ".out")
            input_dir = os.path.join(prjct_dir, subject)
            input_photo_dir = os.path.join(input_dir, "photo_dir")
            ref_mask = os.path.join(input_dir, f"{subject}.mri.mask.mgz")
            command = f"python scripts/3d_photo_reconstruction.py \
                --input_photo_dir {input_photo_dir} \
                --input_segmentation_dir {input_photo_dir} \
                --ref_mask {ref_mask} \
                --photos_of_posterior_side \
                --allow_z_stretch \
                --order_posterior_to_anterior \
                --slice_thickness {0.7*skip:.1f} \
                --photo_resolution 0.7 \
                --output_directory /tmp/\
                --gpu 0 > {log_file} &"

            command = " ".join(command.split())
            all_commands.append(command)

        sbatch_top = """#!/bin/bash
                #SBATCH --account=lcnrtx
                #SBATCH --partition=rtx6000
                #SBATCH --nodes=1
                #SBATCH --ntasks=5
                #SBATCH --gpus=1
                #SBATCH --time=0-01:30:00
                #SBATCH --output="./logs/hcp-recon/test/%x.out"
                #SBATCH --error="./logs/hcp-recon/test/%x.err"
                #SBATCH --mail-user=hvgazula@umich.edu
                #SBATCH --mail-type=FAIL

                source /space/calico/1/users/Harsha/venvs/recon-venv/bin/activate
                export PYTHONPATH=/space/calico/1/users/Harsha/photo-reconstruction

                echo 'Start time:' `date`
                echo 'Node:' $HOSTNAME
                echo "$@"
                start=$(date +%s)
                """

        sbatch_bottom = """end=$(date +%s)
                        echo 'End time:' `date`
                        echo "Elapsed Time: $(($end-$start)) seconds"
                        """

        with open(script_file, "w+") as fn:
            fn.write(inspect.cleandoc(sbatch_top))
            fn.write("\n" * 2)
            fn.write("\n".join(all_commands))
            fn.write("\n")
            fn.write("wait")
            fn.write("\n" * 2)
            fn.write(inspect.cleandoc(sbatch_bottom))


if __name__ == "__main__":
    # grab_diff_photos()
    # put_skip_recons_in_main_dir()
    # print_uw_gt_map()
    # get_hcp_subject_list()
    hcp_recon_group(skip=12, n_jobs=5)
