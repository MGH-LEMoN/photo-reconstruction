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

hcp_%: PRJCT_DIR=/space/calico/1/users/Harsha/SynthSeg/results/4harshaHCP_extracts
hcp_recon:
	for item in `ls -d $(PRJCT_DIR)/*`; do \
		subid=`basename $$item`
		sbatch --job-name=$$subid submit.sh scripts/3d_photo_reconstruction.py \
		--input_photo_dir $$item/photo_dir \
		--input_segmentation_dir $$item/photo_dir \
		--ref_mask $$item/$$subid.mri.mask.mgz \
		--photos_of_posterior_side \
		--allow_z_stretch \
		--order_posterior_to_anterior \
		--slice_thickness 4.2 \
		--photo_resolution 0.7 \
		--output_directory $$item/ref_mask_skip_6 \
		--gpu 0
	done

hcp_test:
	for item in `ls -d $(PRJCT_DIR)/*`; do \
		subid=`basename $$item`
		IFS='_'
		read -r a b <<< $$subid
		echo $$b
		IFS=' '
	done
