# Run all commands in one shell
.ONESHELL:

# Default target
.DEFAULT_GOAL := help

.PHONY : help
## help: run 'make help" at commandline
help : Makefile
	@sed -n 's/^##//p' $<

.PHONY: list
## list: list all targets in the current make file
list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

## uw_recon: Run reconstructions on the UW data
# For more info: https://github.com/hvgazula/photo-reconstruction/blob/main/README.md
# Data is at /cluster/vive/UW_photo_data
uw_recon: SID = 17-0333 18-0086 18-0444 18-0817 18-1045 18-1132 18-1196 18-1274 18-1327 18-1343 18-1470 18-1680 18-1690 18-1704 18-1705 18-1724 18-1754 18-1913 18-1930 18-2056 18-2128 18-2259 18-2260 19-0019 19-0037 19-0100 19-0138 19-0148
uw_%: SKIP_SLICE = $(shell seq 1 4)
uw_recon: PRJCT_DIR = /space/calico/1/users/Harsha/photo-reconstruction
uw_recon: OUT_DIR = $(PRJCT_DIR)/data/uw_photo
uw_recon: DATA_DIR = $(PRJCT_DIR)/data/uw_photo
uw_recon: REF_HARD = --ref_mask $(DATA_DIR)/FLAIR_Scan_Data/$$sid.rotated_cerebrum.mgz
uw_recon: REF_SOFT = --ref_soft_mask $(DATA_DIR)/prob_atlases/onlyCerebrum.nii.gz
uw_recon: REF_IMAGE = --ref_image $(DATA_DIR)/FLAIR_Scan_Data/$$sid.rotated_masked.mgz
uw_recon: REF_KEY = soft
uw_recon: REF_VALUE = $(REF_SOFT)
# REF_KEY: REF_VALUE options are {hard: $(REF_MASK) | soft: $(REF_SOFT_MASK) | image: $(REF_IMAGE)}
uw_%: CMD = sbatch --job-name=$(REF_KEY)-$$skip-$$sid submit.sh
# {echo | python | sbatch --job-name=$(REF_KEY)-$$skip-$$p submit.sh} 
uw_recon:
	for sid in $(SID); do \
		for skip in $(SKIP_SLICE); do \
			$(CMD) scripts/3d_photo_reconstruction.py \
			--input_photo_dir $(DATA_DIR)/Photo_data/$$sid/$$sid\_MATLAB \
			--input_segmentation_dir $(DATA_DIR)/Photo_data/$$sid/$$sid\_MATLAB \
			$(REF_VALUE) \
			--photos_of_posterior_side \
			--allow_z_stretch \
			--slice_thickness 4 \
			--photo_resolution 0.1 \
			--output_directory $(OUT_DIR)/$$sid/ref_$(REF_KEY)_skip_$$skip \
			--gpu 0 \
			--skip \
			--multiply_factor $$skip; \
		done; \
	done;
	
## mgh_recon: Run reconstructions on the MGH data
# For more info: https://github.com/hvgazula/photo-reconstruction/blob/main/README.md
mgh_recon: SID = #SUBJ_ID

#left = 2711_left 2708_left 2629_left 2605_left 2607_left 2614_left 2615_left 2619_left 2621_left 2624_left 2629_left 2630_left 2635_left 2657_left 2663_left 2668_left 2687_left 2689_left 
#whole = 2604_whole 2628_whole 2644_whole 2661_whole 2675_whole
#right = 2618_right 2623_right 2627_right 2637_right 2638_right 2639_right 2642_right 2648_right 2683_right 2691_right
mgh_%: SKIP_SLICE = $(shell seq 1 1)
mgh_recon: PRJCT_DIR = /autofs/cluster/vive/
mgh_recon: OUT_DIR = $(PRJCT_DIR)/MGH_photo_recon/
mgh_recon: DATA_DIR = $(PRJCT_DIR)/MGH_photo_recon/

# REF_KEY: REF_VALUE options are {hard: $(REF_MASK) | soft: $(REF_SOFT_MASK) | image: $(REF_IMAGE)}
mgh_recon: SAMPLE_LEFT = /autofs/cluster/vive/UW_photo_recon/prob_atlases/onlyCerebrum.left_hemi.smoothed.nii.gz
mgh_recon: SAMPLE_RIGHT = /autofs/cluster/vive/UW_photo_recon/prob_atlases/onlyCerebrum.right_hemi.smoothed.nii.gz
mgh_recon: SAMPLE_WHOLE = /autofs/cluster/vive/UW_photo_recon/prob_atlases/onlyCerebrum.smoothed.nii.gz

mgh_%: CMD = jobsubmit -p rtx8000 -A lcnrtx -m 128G -t 0-4:00:00 -c 3 -G 1
# {echo | python | sbatch --job-name=$(REF_KEY)-$$skip-$$p submit.sh} 
mgh_recon:
	for sid in $(SID); do \
		for skip in $(SKIP_SLICE); do \
			$(CMD) python /autofs/cluster/vive/tmp/old_gui_versions/photo-reconstruction-main/scripts/3d_photo_reconstruction.py \
			--input_photo_dir $(DATA_DIR)/$$sid/deformed \
			--input_segmentation_dir $(DATA_DIR)/$$sid/connected_components \
			--ref_surface $(DATA_DIR)/$$sid/mesh/$$sid.stl \
			--mesh_autoalign_target $(SAMPLE_WHOLE) \
			--photos_of_posterior_side \
			--allow_z_stretch \
			--slice_thickness 10 \
			--photo_resolution 0.1 \
			--output_directory $(OUT_DIR)/$$sid/recon_new \
			--gpu 0; \
		done; \
	done;

# propagate_gt: Propagate ground truth labels to reconstruction space
uw_gt_propagate: REF_DIR=/space/calico/1/users/Harsha/photo-reconstruction/data/uw_photo/recons/results_Henry/Results_hard
uw_gt_propagate:
	while IFS=, read -r sid gt_idx _
	do
		for skip in $(SKIP_SLICE); do \
			reference_intensities=$(REF_DIR)/$$sid/$$sid.hard.recon.mgz
			reference_segmentation=$(REF_DIR)/$$sid/$$sid\_hard_manualLabel_merged.mgz
			target_intensities=/space/calico/1/users/Harsha/photo-reconstruction/data/uw_photo/Photo_data/$$sid/ref_$(REF_KEY)_skip_$$skip/photo_recon.mgz
			output_segmentation=$$sid\_seg_output.mgz
			output_QC_prefix=$$sid\_seg_output_QC
			$(RUN_CMD) matlab -nodisplay -nosplash -r "cd('scripts'); propagate_manual_segs_slices_elastix_smart('$$reference_intensities', '$$reference_segmentation', '$$target_intensities', '$$output_segmentation', '$$output_QC_prefix', '$$skip', $$gt_idx); exit"
		done; \
	done < ./results/uw_gt_map.csv

# PICS = 17-0333
# propagate_slices:
# 	for p in $(PICS); do \
# 		for skip in $(SKIP_SLICE); do \
# 			sbatch --job-name=prop-$$p-$$skip --export=ALL,sid=$$p,skip=$$skip submit.sh
# 		done; \
# 	done;

## gt_slice_idx: print ground truth slice idx
# For more info: https://github.com/hvgazula/photo-reconstruction/wiki/Index-of-GT-slices
gt_slice_idx:
	python -c "from scripts import misc_utils; misc_utils.print_gt_slice_idx()"

hcp_%: SKIP=04
hcp_%: THICK=2.8
hcp_%: JITTER=$(shell seq 3 -1 1)
hcp_recon:
	COUNTER=0
	for jitter in $(JITTER); do \
		PRJCT_DIR=/space/calico/1/users/Harsha/SynthSeg/results/hcp-results-20220615/4harshaHCP-skip-$(SKIP)-r$$jitter
		for item in `ls -d $$PRJCT_DIR/*`; do
			subid=`basename $$item`
			sbatch --job-name=skip-$(SKIP)-r$$jitter/$$subid submit.sh scripts/3d_photo_reconstruction.py \
				--input_photo_dir $$item/photo_dir \
				--input_segmentation_dir $$item/photo_dir \
				--ref_mask $$item/$$subid.mri.mask.mgz \
				--photos_of_posterior_side \
				--allow_z_stretch \
				--order_posterior_to_anterior \
				--slice_thickness $(THICK) \
				--photo_resolution 0.7 \
				--output_directory $$item/ref_mask_skip_$(SKIP) \
				--gpu 0;
			let COUNTER=COUNTER+1
			@if (( $$COUNTER % 100 == 0 )); then\
				sleep 15m;\
			fi
		done; \
	done

## run recon given a file with failed subjects
hcp_fail1:
		while IFS= read -r subid
		do
			for jitter in $(JITTER); do \
				PRJCT_DIR=/space/calico/1/users/Harsha/SynthSeg/results/hcp-results-20220613/4harshaHCP-skip-$(SKIP)-r$$jitter
				sbatch --job-name=skip-$(SKIP)-r$$jitter/subject_$$subid submit.sh scripts/3d_photo_reconstruction.py \
				--input_photo_dir $$PRJCT_DIR/subject_$$subid/photo_dir \
				--input_segmentation_dir $$PRJCT_DIR/subject_$$subid/photo_dir \
				--ref_mask $$PRJCT_DIR/subject_$$subid/subject_$$subid.mri.mask.mgz \
				--photos_of_posterior_side \
				--allow_z_stretch \
				--order_posterior_to_anterior \
				--slice_thickness $(THICK) \
				--photo_resolution 0.7 \
				--output_directory $$PRJCT_DIR/subject_$$subid/ref_mask_skip_$(SKIP) \
				--gpu 0
			done;
		done < /space/calico/1/users/Harsha/SynthSeg/test_csv.csv

hcp_%: SKIP=02
hcp_%: THICK=1.4
hcp_%: JITTER=1
hcp_fail_new:
	PRJCT_DIR=/space/calico/1/users/Harsha/SynthSeg/results/hcp-results-20220615/4harshaHCP-skip-$(SKIP)-r$(JITTER)
	for item in `find ./logs/hcp-recon-20220615/skip-$(SKIP)-r$(JITTER)/ -name "*.out" -exec grep -L -e "freeview" {} +`; do \
		subid=`basename $$item`
		IFS='_.'
		read -r a subid c <<< $$subid
		IFS=' '
		sbatch --job-name=skip-$(SKIP)-r$(JITTER)/subject_$$subid submit.sh scripts/3d_photo_reconstruction.py \
				--input_photo_dir $$PRJCT_DIR/subject_$$subid/photo_dir \
				--input_segmentation_dir $$PRJCT_DIR/subject_$$subid/photo_dir \
				--ref_mask $$PRJCT_DIR/subject_$$subid/subject_$$subid.mri.mask.mgz \
				--photos_of_posterior_side \
				--allow_z_stretch \
				--order_posterior_to_anterior \
				--slice_thickness $(THICK) \
				--photo_resolution 0.7 \
				--output_directory $$PRJCT_DIR/subject_$$subid/ref_mask_skip_$(SKIP) \
				--gpu 0
	done

propagate_gt: SKIP_SLICE := $(shell seq 1 4)
propagate_gt: REF_DIR=/space/calico/1/users/Harsha/photo-reconstruction/data/uw_photo/recons/results_Henry/Results_hard
propagate_gt: REF_KEY := image
# {hard | soft | image}
propagate_gt: RUN_CMD := sbatch --job-name=$(REF_KEY)-$$skip-$$sid submit.sh
# {sbatch --job-name=hard-$$skip-$$sid submit.sh | pbsubmit -m hg824 -c | echo}
propagate_gt:
	while IFS=, read -r sid gt_idx _
	do
		for skip in $(SKIP_SLICE); do \
			reference_intensities=$(REF_DIR)/$$sid/$$sid.hard.recon.mgz
			reference_segmentation=$(REF_DIR)/$$sid/$$sid\_hard_manualLabel_merged.mgz
			target_intensities=/space/calico/1/users/Harsha/photo-reconstruction/data/uw_photo/Photo_data/$$sid/ref_$(REF_KEY)_skip_$$skip/photo_recon.mgz
			output_segmentation=$$sid\_seg_output.mgz
			output_QC_prefix=$$sid\_seg_output_QC
			$(RUN_CMD) matlab -nodisplay -nosplash -r "cd('scripts'); propagate_manual_segs_slices_elastix_smart('$$reference_intensities', '$$reference_segmentation', '$$target_intensities', '$$output_segmentation', '$$output_QC_prefix', '$$skip', $$gt_idx); exit"
		done; \
	done < ./results/uw_gt_map.csv

## test-mlsc: Test running matlab on mlsc
test-mlsc:
	sbatch submit.sh matlab -nodisplay -nosplash -nojvm -r "cd('misc'); fact('5')"

## test-launchpad: Test running matlab on launchpad
# Notice the use \" in this compared to the mlsc command
test-launchpad:
	pbsubmit -q matlab -n 2 -O fact1.out -E fact1.err -m hvgazula@umich.edu -e -c "matlab -nodisplay -nosplash -nojvm -r \"cd('misc'); fact('5')\""

# this is exclusvely for r2 cases for the time being
hcpcpu_%: SKIP=10
hcpcpu_%: THICK=7.0
hcpcpu_%: PRJCT_DIR=/space/calico/1/users/Harsha/SynthSeg/results/hcp-results/4harshaHCP-skip-$(SKIP)-r1
hcpcpu_recon:
	COUNTER=0
	for item in `ls -d $(PRJCT_DIR)/*`; do
		subid=`basename $$item`
		sbatch --job-name=$$subid submit-cpu.sh scripts/3d_photo_reconstruction.py \
		--input_photo_dir $$item/photo_dir \
		--input_segmentation_dir $$item/photo_dir \
		--ref_mask $$item/$$subid.mri.mask.mgz \
		--photos_of_posterior_side \
		--allow_z_stretch \
		--order_posterior_to_anterior \
		--slice_thickness $(THICK) \
		--photo_resolution 0.7 \
		--output_directory $$item/ref_mask_skip_$(SKIP)
		let COUNTER=COUNTER+1
		@if (( $$COUNTER % 100 == 0 )); then\
    		sleep 0;\
		fi
	done

hcpcpu_fail:
	for item in `find ./logs/hcp-recon/skip-08-r2 -name "*.err" ! -size 0  | sort`; do \
		subid=`basename $$item`
		IFS='_.'
		read -r a subid c <<< $$subid
		IFS=' '
		sbatch --job-name=subject_$$subid submit-cpu.sh scripts/3d_photo_reconstruction.py \
		--input_photo_dir $(PRJCT_DIR)/subject_$$subid/photo_dir \
		--input_segmentation_dir $(PRJCT_DIR)/subject_$$subid/photo_dir \
		--ref_mask $(PRJCT_DIR)/subject_$$subid/subject_$$subid.mri.mask.mgz \
		--photos_of_posterior_side \
		--allow_z_stretch \
		--order_posterior_to_anterior \
		--slice_thickness $(THICK) \
		--photo_resolution 0.7 \
		--output_directory $(PRJCT_DIR)/subject_$$subid/ref_mask_skip_$(SKIP) \
		--gpu 0
	done

