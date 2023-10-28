import glob
import os

HOME_DIR = '/space/calico/1/users/Harsha'  
NIFTYREG_DIR = HOME_DIR + '/niftyreg'  
PRJCT_DIR = HOME_DIR + '/photo-reconstruction'
SCRIPTS_DIR = PRJCT_DIR + '/scripts'
MNI_TMPLT = '/tmp'  # from Diana/Eugenio
RESULTS_DIR = PRJCT_DIR + '/results/4diana-hcp-recons/skip-14-r0'

def main():
    # get path for reconstructions
    subject_results = sorted(glob.glob(RESULTS_DIR + '/*'))

    # for each reconstruction
    for subject in subject_results:
        # create folder for niftyreg command outputs
        NIFTY_OUTPUT_DIR = subject + '/niftreg_outputs'
        os.makedirs(NIFTY_OUTPUT_DIR, exist_ok=True)

        # get t1, t2 and recon files
        t1_file = glob.glob(subject + '/*T1.nii.gz')[0]
        t2_file = glob.glob(subject + '/*T2.nii.gz')[0]
        recon_file = glob.glob(subject + '/*photo_recon.mgz')[0]

        t1_name = os.path.basename(subject)

        # run synthsr on the reconstruction
        # get the command for this step from Diana
        synthsr_file = subject + f'/{t1_name}.SynthSR.nii.gz'
        CMD_SynthSR = f"mri_synthsr -i {recon_file} -o {synthsr_file}"
        print(CMD_SynthSR)

        print()

        # t1 to mni and synthsr to mni
        for idx, file_type in enumerate(['T1', 'SynthSR'], 1):
            print(f"Running for {file_type}")

            if idx == 1:
                ref_file = t1_file
            else:
                ref_file = synthsr_file
            
            # setting up command 01: reg_aladin
            lin_res_file = NIFTY_OUTPUT_DIR + f"/{t1_name}.{file_type}.linear.nii.gz"
            lin_aff_file = NIFTY_OUTPUT_DIR + f"/{t1_name}.{file_type}.linear.affine.txt"
            CMD_01 = f'{NIFTYREG_DIR}/build/reg-apps/reg_aladin -ref {ref_file} -flo {MNI_TMPLT} -res {lin_res_file} -aff {lin_aff_file} -omp 4'
            print(CMD_01)
        
            # setting up command 02: reg_f3d
            nonlin_res_file = NIFTY_OUTPUT_DIR + f"/{t1_name}.{file_type}.nonlinear.nii.gz"
            nonlin_aff_file = NIFTY_OUTPUT_DIR + f"/{t1_name}.{file_type}.nonlinear.affine.txt"
            cpp_file = NIFTY_OUTPUT_DIR + f"/{t1_name}.{file_type}.CPP.nii.gz"
            sx, vel, lnccw = 0, 0, 0  # check with Diana
            CMD_02 = f"{NIFTYREG_DIR}/build/reg_apps/reg_f3d -ref {ref_file} -flo {MNI_TMPLT} -res {nonlin_res_file} -aff {nonlin_aff_file} -cpp {cpp_file} -omp 4 -sx {sx} -vel {vel} -lnccw {lnccw}"
            print(CMD_02)
        
            # setting up command 03: reg_transform
            out_file = NIFTY_OUTPUT_DIR + f"/D{idx}.nii.gz"
            CMD_03 = f"{NIFTYREG_DIR}/build/reg-apps/reg_transform -ref {ref_file} -def {nonlin_aff_file} {out_file}"
            print(CMD_03) 
            print()
            
        break

if __name__ == '__main__':
    main()

