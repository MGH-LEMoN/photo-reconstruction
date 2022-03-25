function propagate_manual_segs_slices_elastix_smart(reference_intensities, reference_segmentation, target_intensities, output_segmentation, output_QC_prefix, skip, labeled_slice)

skip = str2double(skip);

% clear
% export reference_intensities='/autofs/cluster/vive/UW_photo_recon/recons/results_Henry/Results_hard/18-0086/18-0086.hard.recon.mgz'
% export reference_segmentation='/autofs/cluster/vive/UW_photo_recon/recons/results_Henry/Results_hard/18-0086/18-0086_hard_manualLabel_merged.mgz'
% export target_intensities='/autofs/cluster/vive/UW_photo_recon/Photo_data/18-0086/ref_image_skip_2/photo_recon.mgz'
% export output_segmentation='test_seg_output.mgz'
% export output_QC_prefix='test_seg_output_QC'

% matlab -nodisplay -nosplash -r "propagate_manual_segs_slices_elastix_smart('$reference_intensities', '$reference_segmentation', '$target_intensities', '$output_segmentation', '$output_QC_prefix'); exit"

%disp(reference_intensities)
%disp(reference_segmentation)
%disp(target_intensities)
%disp(output_segmentation)
%disp(output_QC_prefix)

%disp(fsgettmppath)

[parent_dir, ~, ~] = fileparts(target_intensities);
output_dir = fullfile(parent_dir, 'propagated_labels');
tempdir = output_dir;
mkdir(tempdir)

output_segmentation = fullfile(output_dir, output_segmentation);
output_QC_prefix = fullfile(output_dir, output_QC_prefix);

% add path to I/O functions if needed
if isempty(which('MRIread'))
    addpath /usr/local/freesurfer/dev/matlab
end


% niftyreg executables
ELASTIX = '/autofs/cluster/vive/UW_photo_recon/elastix46/run_elastix.sh ';
TRANSFORMIX = '/autofs/cluster/vive/UW_photo_recon/elastix46/run_transformix.sh ';
PARAMFILE_RIGID = '/autofs/cluster/vive/UW_photo_recon/elastix46/rigid_ssd.txt ';
PARAMFILE_SIMILARITY = '/autofs/cluster/vive/UW_photo_recon/elastix46/similarity_ssd.txt ';
PARAMFILE_AFFINE = '/autofs/cluster/vive/UW_photo_recon/elastix46/affine_ssd.txt ';

% Read in images + segmentation
% disp('Reading in volumes');
refI = MRIread(reference_intensities);
refS = MRIread(reference_segmentation);
sizI = size(refI.vol);
sizS = size(refS.vol);
if any(sizI(1:3)~=sizS)
    error('reference image and segmentation do not have the same size');
end
tarI = MRIread(target_intensities);


% find slice with segmentation
nl = squeeze(sum(sum(refS.vol>1,1),2));
[maxi, z] = max(nl);
if maxi==0
    errors('could not find a slice with segmentations');
end

% Prepare slices to register
F = uint8(mean(refI.vol(:,:,z,:),4));
FL = uint8(refS.vol(:,:,z,:));

disp(z)
if false,
    % OK now we need to find the corresponding slice in the target volume
    % We used this with simple statistics
    % disp('Finding corresponding slice in target volume');
    nims = size(tarI.vol,3);
    costs = zeros(1, nims);
    M = F > 0;
    Fhist = hist(F(M),1:255);
    Fhist = Fhist / sum(Fhist);
    for i = 1 : nims
        I = uint8(mean(squeeze(tarI.vol(:,:,i,:)),3));
        M = I > 0;
        Ihist = hist(I(M),1:255);
        Ihist = Ihist / sum(Ihist);
        costs(i) = sum(abs(Ihist-Fhist));
    end
    costs(isnan(costs))=1;
    [tmp, z] = min(costs);
elseif false
    first = min(find(sum(sum(tarI.vol(:,:,:,1),1),2) > 0));
    pad = first - 1;    %first = min(find(sum(sum(tarI.vol(:,:,:,1),1),2) > 0));
    if skip == 1
        z = (z-2) + pad;
    else
        z = 1 + pad + ((z-2) - mod(z-2,skip )) / skip;
    end

else
    pad = 3;
    if skip == 1
        z = pad + labeled_slice;
    else
        z = 1 + pad + (labeled_slice - mod(labeled_slice,skip )) / skip;
    end
end

disp(z)

%%%%%%%%%%5
R = uint8(mean(tarI.vol(:,:,z,:),4));
imwrite(R,[tempdir '/ref.png']);


% OK we now need to register. The problem is that there may be flips and
% rotations, so we try 8 different alternatives and keep the best
best_rmse = inf;
best_labs = [];
best_im = [];
% disp('Registering with 8 different initializations');
c = 0;
for flip = 0:1
    for rotation  = 0:90:270
        c = c + 1;
        % disp(['   ' num2str(c) ' of 8']);
        im = F;
        if flip, im = fliplr(im); end
        im = imrotate(im,rotation);
        imwrite(im,[tempdir '/flo.png']);
        im = FL;
        if flip, im = fliplr(im); end
        im = imrotate(im,rotation);
        imwrite(im,[tempdir '/flo-labs.png']);
        
        a = system([ELASTIX ' -f ' tempdir '/ref.png ' ...
            ' -m ' tempdir '/flo.png ' ...
            ' -out ' tempdir ' ' ...
            ' -p ' PARAMFILE_RIGID ' -p ' PARAMFILE_SIMILARITY ' -p ' PARAMFILE_AFFINE  ...
            ' -threads 4 >/dev/null']);
        if a, error('error in elastix'); end
        mri = MRIread([tempdir '/result.2.nii.gz']);
        REG = uint8(mri.vol);
        rmse = sqrt(mean(mean((double(R)-double(mri.vol)).^2)));

        if rmse<best_rmse
            best_rmse = rmse;
            a = system([TRANSFORMIX ' -in ' tempdir '/flo-labs.png ' ...
                                    ' -out ' tempdir ' ' ...
                                    ' -tp ' tempdir '/TransformParameters.2.txt ' ...
                                    ' -threads 4 >/dev/null']);
            if a, error('error in transformix'); end
            mri = MRIread([tempdir '/result.nii.gz']);
            best_labs = mri.vol;
            best_im = REG;
            
        end
    end
end
% disp('Writing results to disk');
mri = tarI;
mri.vol(:) = 0;
mri.vol = mri.vol(:,:,:,1);
mri.vol(:,:,z) = best_labs;
MRIwrite(mri,output_segmentation);

imwrite(R, [output_QC_prefix '.target.png']);
imwrite(best_im, [output_QC_prefix '.registered.png']);
D = uint8(3 * abs(double(R) - double(best_im)));
imwrite(D, [output_QC_prefix '.difference.png']);

cmd = ['freeview -v ' target_intensities ':rgb=1 -v ' output_segmentation ...
    ':colormap=lut -slice 100 100 ' num2str(z-1) ' -layout 4 -viewport coronal &'];
disp(cmd);
%system(cmd); 
