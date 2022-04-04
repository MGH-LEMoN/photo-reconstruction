clear; clc; close all;

diary propagate_image.out
subjects = {'17-0333', '18-0086', '18-0444', '18-0817', '18-1045', '18-1132', '18-1196', '18-1274', '18-1327', '18-1343', '18-1470', '18-1680', '18-1690', '18-1704', '18-1705', '18-1724', '18-1754', '18-1913', '18-1930', '18-2056', '18-2128', '18-2259', '18-2260', '19-0019', '19-0037', '19-0100', '19-0138', '19-0148'};
skips = {'1', '2', '3', '4'};
labeled_slice = {20, 21, 18, 19, 20, 20, 18, 19, 19, 20, 17, 17, 21, 21, 19, 22, 18, 19, 18, 20, 20, 18, 20, 0, 21, 19, 21, 22};

for id = 1:length(subjects)
    for skip_id = 1:length(skips)
        reference_intensities = ['/space/calico/1/users/Harsha/photo-reconstruction/data/UW_photo_recon/recons/results_Henry/Results_hard/' subjects{id} '/' subjects{id} '.hard.recon.mgz'];
        reference_segmentation = ['/space/calico/1/users/Harsha/photo-reconstruction/data/UW_photo_recon/recons/results_Henry/Results_hard/' subjects{id} '/' subjects{id} '_hard_manualLabel_merged.mgz'];
        target_intensities = ['/space/calico/1/users/Harsha/photo-reconstruction/results/uw_recons/' subjects{id} '/ref_image_skip_' skips{skip_id} '/photo_recon.mgz'];
        output_segmentation = [subjects{id} '_seg_output.mgz'];
        output_QC_prefix = [subjects{id} '_seg_output_QC'];
        try
            propagate_manual_segs_slices_elastix_smart(reference_intensities, reference_segmentation, target_intensities, output_segmentation, output_QC_prefix, skips{skip_id}, labeled_slice{id})
            disp(['SUCCESS: ' subjects{id} '-' skips{skip_id}])
        catch
            disp(['FAILED: ' subjects{id} '-' skips{skip_id}])
        end
    end
end

% matlab -nodisplay -nosplash -r "propagate_manual_segs_slices_elastix_smart('/autofs/cluster/vive/UW_photo_recon/results_Henry/Results_hard/17-0333/17-0333.hard.recon.mgz', '/autofs/cluster/vive/UW_photo_recon/results_Henry/Results_hard/17-0333/17-0333_hard_manualLabel_merged.mgz', '/autofs/cluster/vive/UW_photo_recon/Photo_data/17-0333/ref_image_skip_2/photo_recon.mgz', '17-0333_seg_output.mgz', '17-0333_seg_output_QC'); exit"

% 18-0817 - 4, 18-1705 - 3, 18-1754 - 4, 19-0019 - 2,3,4
