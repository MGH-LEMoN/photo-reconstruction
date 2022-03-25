# Run all commands in one shell
.ONESHELL:

# Default target
.DEFAULT_GOAL := help

.PHONY : help
## help: run 'make help" at commandline
help : Makefile
	@sed -n 's/^##//p' $<

.PHONY: list
list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'


CMD := sbatch --job-name=$$skip-$$p submit.sh 
# {echo | python | sbatch --job-name=$$skip-$$p submit.sh}
PICS = 17-0333 18-0086 18-0444 18-0817 18-1045 18-1132 18-1196 18-1274 18-1327 18-1343 18-1470 18-1680 18-1690 18-1704 18-1705 18-1724 18-1754 18-1913 18-1930 18-2056 18-2128 18-2259 18-2260 19-0019 19-0037 19-0100 19-0138 19-0148
SKIP_SLICE := $(shell seq 4 4)

## ref_mask: Run reconstruction with hard reference
recon_%: PRJCT_DIR=/space/calico/1/users/Harsha/photo-reconstruction
recon_%: OUT_DIR=$(PRJCT_DIR)/results/uw_recons
recon_%: DATA_DIR=$(PRJCT_DIR)/data/UW_photo_recon
recon_mask:
	for p in $(PICS); do \
		for skip in $(SKIP_SLICE); do \
			sbatch --job-name=hard-$$skip-$$p submit.sh scripts/3d_photo_reconstruction.py \
			--input_photo_dir $(DATA_DIR)/Photo_data/$$p/$$p\_MATLAB \
			--input_segmentation_dir $(DATA_DIR)/Photo_data/$$p/$$p\_MATLAB \
			--ref_mask $(DATA_DIR)/FLAIR_Scan_Data/NP$$p.rotated.binary.mgz \
			--photos_of_posterior_side \
			--allow_z_stretch \
			--slice_thickness 4 \
			--photo_resolution 0.1 \
			--output_directory $(OUT_DIR)/$$p/ref_mask_skip_$$skip \
			--gpu 0 \
			--skip \
			--multiply_factor $$skip; \
		done; \
	done;

## ref_image: Run reconstruction with image/volume reference
recon_image:
	for p in $(PICS); do \
		for skip in $(SKIP_SLICE); do \
			sbatch --job-name=image-$$skip-$$p submit.sh scripts/3d_photo_reconstruction.py \
			--input_photo_dir $(DATA_DIR)/Photo_data/$$p/$$p\_MATLAB \
			--input_segmentation_dir $(DATA_DIR)/Photo_data/$$p/$$p\_MATLAB \
			--ref_image $(DATA_DIR)/FLAIR_Scan_Data/NP$$p.rotated.masked.mgz \
			--photos_of_posterior_side \
			--allow_z_stretch \
			--slice_thickness 4 \
			--photo_resolution 0.1 \
			--output_directory $(OUT_DIR)/$$p/ref_image_skip_$$skip \
			--gpu 0 \
			--skip \
			--multiply_factor $$skip; \
		done; \
	done;

## ref_soft_mask: Run reconstruction with soft/probabilistic reference
recon_soft_mask:
	for p in $(PICS); do \
		for skip in $(SKIP_SLICE); do \
			sbatch --job-name=soft-$$skip-$$p submit.sh scripts/3d_photo_reconstruction.py \
			--input_photo_dir $(DATA_DIR)/Photo_data/$$p/$$p\_MATLAB \
			--input_segmentation_dir $(DATA_DIR)/Photo_data/$$p/$$p\_MATLAB \
			--ref_soft_mask $(DATA_DIR)/prob_atlases/onlyCerebrum.nii.gz \
			--photos_of_posterior_side \
			--allow_z_stretch \
			--slice_thickness 4 \
			--photo_resolution 0.1 \
			--output_directory $(OUT_DIR)/$$p/ref_soft_mask_skip_$$skip \
			--gpu 0 \
			--skip \
			--multiply_factor $$skip; \
		done; \
	done;

# PICS = 17-0333
# propagate_slices:
# 	for p in $(PICS); do \
# 		for skip in $(SKIP_SLICE); do \
# 			sbatch --job-name=prop-$$p-$$skip --export=ALL,sid=$$p,skip=$$skip submit.sh
# 		done; \
# 	done;

## gt_slice_idx: print ground truth slice idx
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