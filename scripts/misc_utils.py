import glob
import csv
import os
import re

import ext.my_functions as my
import numpy as np
from PIL import Image
from tabulate import tabulate

PRJCT_KEY = 'UW_photo_recon'
PRJCT_DIR = '/space/calico/1/users/Harsha/photo-reconstruction/'
DATA_DIR = os.path.join(PRJCT_DIR, 'data', PRJCT_KEY)
RESULTS_DIR = os.path.join(PRJCT_DIR, 'results', PRJCT_KEY)


def grab_diff_photos(ref_string):
    """_summary_

    Args:
        ref_string (str): options include 'image', 'hard', 'soft'
    """
    # photo_path = os.path.join(DATA_DIR, 'Photo_data')
    subject_list = sorted(glob.glob(os.path.join(RESULTS_DIR, '*')))

    for skip_val in [1, 2, 3, 4]:
        ref_folder_string = f'ref_{ref_string}_skip_{skip_val}'
        dst_pdf = os.path.join(RESULTS_DIR, ref_folder_string + '.pdf')

        im_list = []
        for subject in subject_list:
            ref_folder = os.path.join(subject, ref_folder_string)
            diff_file = glob.glob(
                os.path.join(ref_folder, 'propagated_labels', '*difference*'))

            if len(diff_file) == 0:
                continue
            src_file = diff_file[0]

            # imagelist is the list with all image filenames
            im_list.append(Image.open(src_file))

        im_list[0].save(dst_pdf,
                        "PDF",
                        resolution=100.0,
                        save_all=True,
                        append_images=im_list)


def grab_diff_photos_main():
    for ref in ['hard', 'soft', 'image']:
        grab_diff_photos(ref)


def return_common_subjects(*args):
    args = [{
        os.path.split(input_file)[-1][:7]: input_file
        for input_file in file_list
    } for file_list in args]

    lst = [set(lst.keys()) for lst in args]

    # One-Liner to intersect a list of sets
    common_names = sorted(lst[0].intersection(*lst))

    args = [[lst[key] for key in common_names] for lst in args]

    return args


def get_first_non_empty_slice(volume_file):
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
    HENRY_RESULTS = os.path.join(DATA_DIR, 'recons/results_Henry/')
    HARD_FOLDER = os.path.join(HENRY_RESULTS, 'Results_hard')
    SOFT_FOLDER = os.path.join(HENRY_RESULTS, 'Results_soft')

    hard_subject_list = sorted(glob.glob(os.path.join(HARD_FOLDER, '*-*')))
    soft_subject_list = sorted(glob.glob(os.path.join(SOFT_FOLDER, '*-*')))

    common_subject_list = return_common_subjects(hard_subject_list,
                                                 soft_subject_list)

    subject_gt_idx = []
    for hard_subject, soft_subject in zip(*common_subject_list):
        subject_id = os.path.basename(hard_subject)

        soft_seg_gt = f'soft/{subject_id}_manualLabel.mgz'
        soft_seg_unmerged_gt = f'soft/{subject_id}_soft_manualLabel.mgz'
        soft_seg_merged_gt = f'soft/{subject_id}_soft_manualLabel_merged.mgz'

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
                hard_recon, ref_seg1, ref_seg2, soft_recon, ref_seg4, ref_seg5
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

    with open(os.path.join(RESULTS_DIR, 'gt_slice_idx.txt'), 'w') as f:
        f.write(
            tabulate(subject_gt_idx,
                     headers=[
                         'subject',
                         'h_recon_shape',
                         'h_recon_nz',
                         'h_unmerge_shape',
                         'h_unmerge_nz_slice',
                         'h_merge_shape',
                         'h_merge_nz_slice',
                         's_recon_shape',
                         's_recon_nz',
                         's_unmerge_shape',
                         's_unmerge_nz_slice',
                         's_merge_shape',
                         's_merge_nz_slice',
                     ],
                     colalign='center'))


def get_gt_slice_idx(seg_vol, recon_vol=None):
    try:
        seg_vol = my.MRIread(seg_vol, im_only=True)

        # gt slice (includes padding)
        slice_idx = np.argmax((seg_vol > 1).sum(0).sum(0))

        if recon_vol is not None:
            recon_vol = my.MRIread(recon_vol, im_only=True)

            # index of first nonzero slice (or # of padded slices)
            slice_idx_recon = np.min(
                np.where(recon_vol[..., 0].sum(0).sum(0) > 1.))
            slice_idx -= slice_idx_recon
    except:
        slice_idx = 'DNE'

    return slice_idx


def print_uw_gt_map():
    """Print the index of the GT slice (excludes padding)
    Note: This info is relevant for simulating skips in reconstruction
    """
    HENRY_RESULTS = os.path.join(DATA_DIR, 'recons/results_Henry/')
    HARD_FOLDER = os.path.join(HENRY_RESULTS, 'Results_hard')

    # list all folders/subjects
    hard_subject_list = sorted(glob.glob(os.path.join(HARD_FOLDER, '*-*')))

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

    with open(os.path.join(RESULTS_DIR, 'uw_gt_map.csv'), 'w') as f:
        f.write(tabulate(subject_gt_idx, headers=[
            'subject',
            'gt_slice_idx',
        ]))

    with open(os.path.join(RESULTS_DIR, 'uw_gt_map.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(subject_gt_idx)


def get_hcp_subject_list():
    T1_files = glob.glob(
        os.path.join('/space/calico/1/users/Harsha/SynthSeg/data/4harshaHCP',
                     '*.T1.*'))
    T2_files = glob.glob(
        os.path.join('/space/calico/1/users/Harsha/SynthSeg/data/4harshaHCP',
                     '*.T2.*'))

    T1_files = [
        re.findall('[0-9]+', os.path.basename(file))[0] for file in T1_files
    ]
    T2_files = [
        re.findall('[0-9]+', os.path.basename(file))[0] for file in T2_files
    ]

    common_subjects = sorted(set(T1_files).intersection(T2_files))

    with open(
            os.path.join(
                '/space/calico/1/users/Harsha/photo-reconstruction/results/hcp_subject_list.csv'
            ),
            'w',
    ) as f:
        f.write('\n'.join(common_subjects))
        f.write('\n')

    return


if __name__ == '__main__':
    grab_diff_photos_main()
    # print_uw_gt_map()
    # get_hcp_subject_list()
