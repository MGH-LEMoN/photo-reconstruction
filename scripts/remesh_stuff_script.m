% NOTE: This script is intended to run for cases 2604 and 2605 where the
% cerebellum and brain stem were not removed during surface scanning.

% In the end we had a better idea: since rasterization did actually work, we:
% 1. rasterized into a volume;
% 2. smoothed a bit;
% 3. kept the largest connected component;
% 4. remeshed with marching cubes (topologically correct),
% 5. and smoothed the mesh a bit. 

clear

addpath ~/matlab/myFunctions/

input_mesh = '/autofs/cluster/vive/MGH_photo_recon/2605_left/mesh/2605_left.stl';
output_mesh = '/cluster/scratch/monday/2605_left_repaired.stl';

system(['mris_copy_header ' input_mesh ' /usr/local/freesurfer/dev/subjects/bert/surf/rh.white /tmp/input_mesh_with_header.surf']);
disp('header copied');

system('mris_fill -r .25 /tmp/input_mesh_with_header.surf /tmp/filled.mgz');
disp('mesh filled into volume');

a=MRIread('/tmp/filled.mgz');
disp('filled volume read');
pad = 10;
A=a.vol;
B=zeros(size(A)+2*pad);
B(pad+1:end-pad,pad+1:end-pad,pad+1:end-pad)=A;
w=4;
B=B>0;
B=imdilate(B,createSphericalStrel(w));
disp('filled volume dilated');
B=imfill(B,'holes');
disp('holes filled');
B=imerode(B,createSphericalStrel(w));
disp('eroded');
B=getLargestCC(B);
disp('largest CC');
a.vol = B(pad+1:end-pad,pad+1:end-pad,pad+1:end-pad);
MRIwrite(a,'/tmp/repaired.mgz');
disp('repaired volume written');

system('mri_mc /tmp/repaired.mgz 1 /tmp/repaired.surf');
disp('tessellated');

system('mris_smooth /tmp/repaired.surf /tmp/smoothed.surf');
disp('smnoothed');

system(['mris_smooth /tmp/smoothed.surf ' output_mesh]);
disp('converted; all done');

system(['chmod 777 ' output_mesh]);
disp('permissions changed');