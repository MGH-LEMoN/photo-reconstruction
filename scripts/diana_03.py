import argparse
import glob
import os

HOME_DIR = "/space/calico/1/users/Harsha"
NIFTYREG_DIR = HOME_DIR + "/niftyreg"
PRJCT_DIR = HOME_DIR + "/photo-reconstruction"
DATA_DIR = HOME_DIR + "/data"
SCRIPTS_DIR = PRJCT_DIR + "/scripts"
MNI_TMPLT = (
    DATA_DIR
    + "/prob_atlases/mni_icbm152_t1_tal_nlin_sym_09c.rigidlyAlignedToAtlas.left_hemi.masked.nii.gz"
)
RESULTS_DIR = PRJCT_DIR + "/results/4diana-hcp-recons/skip-14-r0"
SYNTHSR_MODEL = DATA_DIR + "/SynthSR_left_hemi_3d.h5"


def main(subject_results):
    # get path for reconstructions
    # subject_results = sorted(glob.glob(RESULTS_DIR + "/*"))

    # for each reconstruction
    # wrap in list to avoid modifying the code a lot
    subject_results = [subject_results]

    for subject in subject_results:
        # create folder for niftyreg command outputs
        NIFTY_OUTPUT_DIR = subject + "/niftreg_outputs"
        os.makedirs(NIFTY_OUTPUT_DIR, exist_ok=True)

        # get t1, t2 and recon files
        t1_file = glob.glob(subject + "/*T1.nii.gz")[0]
        t2_file = glob.glob(subject + "/*T2.nii.gz")[0]
        recon_file = glob.glob(subject + "/*photo_recon.mgz")[0]

        t1_name = os.path.basename(subject)

        # run synthsr on the reconstruction
        # get the command for this step from Diana
        synthsr_file = subject + f"/{t1_name}.SynthSR.nii.gz"
        CMD_SynthSR = (
            f"mri_synthsr --i {recon_file} --o {synthsr_file} --model {SYNTHSR_MODEL}"
        )
        os.system(CMD_SynthSR)

        print()

        # t1 to mni and synthsr to mni
        for idx, file_type in enumerate(["T1", "SynthSR"], 1):
            print(f"Running for {file_type}")

            if idx == 1:
                ref_file = t1_file
            else:
                ref_file = synthsr_file

            # setting up command 01: reg_aladin
            lin_res_file = NIFTY_OUTPUT_DIR + f"/{t1_name}.{file_type}.linear.nii.gz"
            lin_aff_file = (
                NIFTY_OUTPUT_DIR + f"/{t1_name}.{file_type}.linear.affine.txt"
            )
            CMD_01 = f"{NIFTYREG_DIR}/build/reg-apps/reg_aladin -ref {ref_file} -flo {MNI_TMPLT} -res {lin_res_file} -aff {lin_aff_file} -omp 4"
            os.system(CMD_01)
            

            # setting up command 02: reg_f3d
            nonlin_res_file = (
                NIFTY_OUTPUT_DIR + f"/{t1_name}.{file_type}.nonlinear.nii.gz"
            )
            nonlin_aff_file = (
                NIFTY_OUTPUT_DIR + f"/{t1_name}.{file_type}.nonlinear.affine.txt"
            )
            cpp_file = NIFTY_OUTPUT_DIR + f"/{t1_name}.{file_type}.CPP.nii.gz"
            CMD_02 = f"{NIFTYREG_DIR}/build/reg_apps/reg_f3d -ref {ref_file} -flo {MNI_TMPLT} -res {nonlin_res_file} -aff {lin_aff_file} -cpp {cpp_file} -omp 4 -sx 15 -vel --lnccw 4.0"
            os.system(CMD_02)

            # setting up command 03: reg_transform
            out_file = NIFTY_OUTPUT_DIR + f"/D{idx}.nii.gz"
            CMD_03 = f"{NIFTYREG_DIR}/build/reg-apps/reg_transform -ref {ref_file} -def {cpp_file} nonlin_aff_file {out_file}"
            os.system(CMD_03)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, nargs="*", required=True)
    options = parser.parse_args()
    main(options.input_dir)

