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
SKIP_SLICE := $(shell seq 2 3)

## ref_mask: Run reconstruction with hard reference
ref_mask:
	for p in $(PICS); do \
		for skip in $(SKIP_SLICE); do \
			$(CMD) scripts/3d_photo_reconstruction.py \
			--input_photo_dir /autofs/cluster/vive/UW_photo_recon/Photo_data/$$p/$$p\_MATLAB \
			--input_segmentation_dir /autofs/cluster/vive/UW_photo_recon/Photo_data/$$p/$$p\_MATLAB \
			--ref_mask /autofs/cluster/vive/UW_photo_recon/FLAIR_Scan_Data/NP$$p.rotated.binary.mgz \
			--photos_of_posterior_side \
			--allow_z_stretch \
			--slice_thickness 4 \
			--photo_resolution 0.1 \
			--output_directory /autofs/cluster/vive/UW_photo_recon/Photo_data/$$p/ref_mask_skip_$$skip \
			--gpu 0 \
			--skip \
			--multiply_factor $$skip; \
		done; \
	done;

## ref_image: Run reconstruction with image/volume reference
ref_image:
	for p in $(PICS); do \
		for skip in $(SKIP_SLICE); do \
			sbatch --job-name=$$skip-$$p submit.sh scripts/3d_photo_reconstruction.py \
			--input_photo_dir /autofs/cluster/vive/UW_photo_recon/Photo_data/$$p/$$p\_MATLAB \
			--input_segmentation_dir /autofs/cluster/vive/UW_photo_recon/Photo_data/$$p/$$p\_MATLAB \
			--ref_image /autofs/cluster/vive/UW_photo_recon/FLAIR_Scan_Data/NP$$p.rotated.masked.mgz \
			--photos_of_posterior_side \
			--allow_z_stretch \
			--slice_thickness 4 \
			--photo_resolution 0.1 \
			--output_directory /autofs/cluster/vive/UW_photo_recon/Photo_data/$$p/ref_image_skip_$$skip \
			--gpu 0 \
			--skip \
			--multiply_factor $$skip; \
		done; \
	done;

## ref_soft_mask: Run reconstruction with soft/probabilistic reference
ref_soft_mask:
	for p in $(PICS); do \
		for skip in $(SKIP_SLICE); do \
			sbatch --job-name=$$skip-$$p submit.sh scripts/3d_photo_reconstruction.py \
			--input_photo_dir /autofs/cluster/vive/UW_photo_recon/Photo_data/$$p/$$p\_MATLAB \
			--input_segmentation_dir /autofs/cluster/vive/UW_photo_recon/Photo_data/$$p/$$p\_MATLAB \
			--ref_soft_mask /autofs/cluster/vive/UW_photo_recon/prob_atlases/onlyCerebrum.nii.gz \
			--photos_of_posterior_side \
			--allow_z_stretch \
			--slice_thickness 4 \
			--photo_resolution 0.1 \
			--output_directory /autofs/cluster/vive/UW_photo_recon/Photo_data/$$p/ref_soft_mask_skip_$$skip \
			--gpu 0 \
			--skip \
			--multiply_factor $$skip; \
		done; \
	done;
