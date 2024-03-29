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

# Generic Variables
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d")

## uw_recon: Run reconstructions on the UW data
# For more info: https://github.com/hvgazula/photo-reconstruction/blob/main/README.md
# Data is at /cluster/vive/UW_photo_data
uw_recon: SID = `ls -d1 /cluster/vive/UW_photo_recon/Photo_data/*-*/ | xargs -n 1 basename`
uw_%: SKIP_SLICE = $(shell seq 1 4)
uw_recon: PRJCT_DIR = /space/calico/1/users/Harsha/photo-reconstruction
uw_recon: OUT_DIR = $(PRJCT_DIR)/data/uw_photo
uw_recon: DATA_DIR = $(PRJCT_DIR)/data/uw_photo
uw_recon: REF_KEY = hard
# {hard | soft | image}
uw_%: CMD = sbatch --job-name=$(REF_KEY)-$$skip-$$sid submit.sh
# {echo | python | sbatch --job-name=$(REF_KEY)-$$skip-$$p submit.sh} 
uw_recon:
	for sid in $(SID); do \
		for skip in $(SKIP_SLICE); do \
			if [[ $(REF_KEY) == "hard" ]]; then
				REF_FLAG="--ref_mask $(DATA_DIR)/FLAIR_Scan_Data/$$sid.rotated_cerebrum.mgz"
			elif [[ $(REF_KEY) == "soft" ]]; then
				REF_FLAG="--ref_soft_mask $(DATA_DIR)/prob_atlases/onlyCerebrum.nii.gz"
			elif [[ $(REF_KEY) == "image" ]]; then
				REF_FLAG="--ref_image $(DATA_DIR)/FLAIR_Scan_Data/$$sid.rotated_masked.mgz"
			else
				echo "Invalid REF_KEY"
				exit 0
			fi

			$(CMD) python scripts/3d_photo_reconstruction.py \
			--input_photo_dir $(DATA_DIR)/Photo_data/$$sid/$$sid\_MATLAB \
			--input_segmentation_dir $(DATA_DIR)/Photo_data/$$sid/$$sid\_MATLAB \
			$$REF_FLAG \
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
	
## mgh_recon: Run reconstructions on MGH data (with surface/atlas reference)
mgh_recon: SID = `ls -d1 /cluster/vive/MGH_photo_recon/2*_{whole,left,right}/ | xargs -n 1 basename`
mgh_recon: CODE_DIR = /space/calico/1/users/Harsha/photo-reconstruction
mgh_recon: DATA_DIR = /cluster/vive/MGH_photo_recon
mgh_recon: OUT_DIR = $(DATA_DIR)
mgh_recon: REF_KEY = surface
# {surface | atlas}
mgh_recon: MESH_COORDINATES = /cluster/vive/MGH_photo_recon/mgh_mesh_coordinates_ratings.csv
mgh_recon: CMD = sbatch --job-name=$$sid-$(REF_KEY) submit.sh
# {echo | python | sbatch --job-name=$(REF_KEY)-$$sid submit.sh} 
mgh_recon:
	mkdir -p ./logs/mgh-recon-$(DT)
	for sid in $(SID); do \
		VERTICES=`cat $(MESH_COORDINATES) | grep $$sid | cut -d _ -f2 | cut -d , -f2-4`
		
		if [[ -z "$$VERTICES" ]] || [[ "$$VERTICES" == "0,0,0" ]] ; then
			echo "$$sid - empty vertices"
			continue
		fi

		if [[ $(REF_KEY) == "surface" ]]; then
			REF_FLAG="--ref_surface $(DATA_DIR)/$$sid/mesh/$$sid.stl --mesh_reorient_with_indices $$VERTICES"
		elif [[ $(REF_KEY) == "atlas" ]]; then
			if [[ "$$sid" == *_"whole" ]]; then
				REF_FLAG="--ref_soft_mask /cluster/vive/prob_atlases/onlyCerebrum.nii.gz"
			elif [[ "$$sid" == *_"left" ]]; then
				REF_FLAG="--ref_soft_mask /cluster/vive/prob_atlases/onlyCerebrum.left_hemi.nii.gz"
			elif [[ "$$sid" == *_"right" ]]; then
				REF_FLAG="--ref_soft_mask /cluster/vive/prob_atlases/onlyCerebrum.right_hemi.nii.gz"
			fi
		fi

		$(CMD) fspython $(CODE_DIR)/scripts/mri_3d_photo_recon \
		--input_photo_dir $(DATA_DIR)/$$sid/deformed \
		--input_segmentation_dir $(DATA_DIR)/$$sid/connected_components \
		$$REF_FLAG \
		--photos_of_posterior_side \
		--allow_z_stretch \
		--slice_thickness 10 \
		--photo_resolution 0.1 \
		--output_directory $(OUT_DIR)/$$sid/recon_$(REF_KEY)_$(DT) \
		--gpu 0; \
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

# ## mask-to-surface: Convert binary mask to atlas
mask-to-surface:
	for sid in `ls -d1 /cluster/vive/UW_photo_recon/Photo_data/*-*/ | xargs -n 1 basename`; do
	FILE=/cluster/vive/UW_photo_recon/FLAIR_Scan_Data/$$sid.rotated_cerebrum.mgz
	if [ -f $(FILE) ]; then
		echo mri_mc $$FILE 128 /tmp/tmp.surf;
		echo mris_smooth /tmp/tmp.surf /cluster/vive/UW_photo_recon/FLAIR_Scan_Data/$$sid.smooth.surf;
	fi
	done;
