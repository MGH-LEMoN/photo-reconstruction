# Imports
import argparse
import glob
import os
import sys

import cv2
import nibabel as nib
import numpy as np
import scipy.io
import scipy.ndimage
import torch

torch.set_default_dtype(torch.float64)
# import trimesh

import ext.my_functions as my
import photo_reconstruction.LBFGS as LBFGS
import photo_reconstruction.versatile_reg_network as PRnets
from ext.utils import get_git_revision_short_hash, seed_all

seed_all(0)
print(f"Git Commit Hash: {get_git_revision_short_hash()}")


########################################################
# Parse arguments
parser = argparse.ArgumentParser(
    description="Code for 3D photo reconstruction (Tregidgo, ..., & Iglesias, MICCAI 2020"
)

parser.add_argument(
    "--input_photo_dir",
    type=str,
    nargs="*",
    help="Directory with input photos (required)",
    required=True,
)
parser.add_argument(
    "--input_segmentation_dir",
    type=str,
    nargs="*",
    help="Directory with input slab masks / segmentations (required)",
    required=True,
)

parser.add_argument(
    "--ref_mask", type=str, help="Reference binary mask", default=None
)
parser.add_argument(
    "--ref_surface", type=str, help="Reference surface file", default=None
)
parser.add_argument(
    "--ref_image", type=str, help="Reference image file", default=None
)
parser.add_argument(
    "--ref_soft_mask", type=str, help="Reference soft mask", default=None
)

parser.add_argument(
    "--DL_synthesis",
    type=str,
    help="Model for deep learning synthesis (highly experimental!)",
    default=None,
)

# parser.add_argument(
#     "--mesh_autoalign_target",
#     type=str,
#     help="Probabilistic atlas to globally initialize mesh rotation",
#     default=None,
# )
# parser.add_argument(
#     "--mesh_manually_oriented",
#     dest="mesh_manually_oriented",
#     action="store_true",
#     help="Use this flag if you manually oriented the filled mesh (please see manual)",
# )
# parser.set_defaults(mesh_manually_oriented=False)

parser.add_argument(
    "--mesh_reorient_with_indices",
    type=str,
    help="Vertex indices of frontal pole, occipital pole, and top of central sulcus, separated with commas, for mesh alignment",
    default=None,
)

parser.add_argument(
    "--photos_of_posterior_side",
    dest="posterior_side",
    action="store_true",
    help="Use when photos are taken of posterior side of slabs (default is anterior side)",
)
parser.set_defaults(posterior_side=False)

parser.add_argument(
    "--order_posterior_to_anterior",
    dest="posterior_to_anterior",
    action="store_true",
    help="Use when photos are ordered from posterior to anterior (default is anterior to posterior)",
)
parser.set_defaults(posterior_to_anterior=False)

parser.add_argument(
    "--allow_z_stretch",
    dest="allow_z_stretch",
    action="store_true",
    help="Use to adjust the slice thickness to best match the reference."
    + " You should probably *never* use this with soft references",
)
parser.set_defaults(allow_z_stretch=False)

parser.add_argument(
    "--rigid_only_for_photos",
    dest="rigid_only_for_photos",
    action="store_true",
    help="Switch on if you want photos to deform only rigidly (not affine)",
)
parser.set_defaults(rigid_only_for_photos=False)

parser.add_argument(
    "--slice_thickness", type=float, help="Slice thickness in mm", required=True
)
parser.add_argument(
    "--photo_resolution",
    type=float,
    help="Resolution of the photos in mm",
    required=True,
)
parser.add_argument(
    "--n_cp_nonlin",
    type=int,
    help="number of control points for within slice nonlinear deformation of "
    + " the photos, along largest dimension of image. You should probably "
    + " *never* use this with soft references",
    default=0,
)

parser.add_argument(
    "--stiffness_nonlin",
    type=float,
    help="stiffness of the nonlinear deformation",
    default=0.33333333333333,
)

parser.add_argument(
    "--output_directory",
    type=str,
    help="Output directory with reconstructed photo volume and reference",
    required=True,
)

parser.add_argument("--gpu", type=int, help="Index of GPU to use", default=None)

# TODO: Harsha, bear in mind that this needs to go in the public version...
parser.add_argument("--skip", action="store_true", dest="skip_flag") 
parser.add_argument(
    "--multiply_factor",
    type=int,
    help="Multiplication Factor for thickness",
    default=1,
)

parser.add_argument(
    "--alternate_optimization",
    dest="alternate_optimization",
    action="store_true",
    help="Switch on if you want perform alternate training",
)
parser.set_defaults(alternate_optimization=False)

options = parser.parse_args()

########################################################
# Set to true for fast processing for debugging etc
FAST = False

########################################################
# Set the GPU if needed
if options.gpu is None:
    print("Using the CPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu)
    if torch.cuda.is_available():
        print("Using GPU device " + str(options.gpu))
        device = torch.device("cuda:0")
    else:
        print(
            "Tried to use GPU device "
            + str(options.gpu)
            + " but failed; using CPU instead"
        )
        device = torch.device("cpu")

########################################################
# Input data

# First, make sure that you specify one and only one reference!
n_refs = (
    (options.ref_mask is not None)
    + (options.ref_soft_mask is not None)
    + (options.ref_surface is not None)
    + (options.ref_image is not None)
)
if n_refs != 1:
    raise Exception(
        "You should provide 1 and only 1 reference: binary mask, soft mask, surface, or image"
    )

if options.ref_mask is not None:
    input_reference = options.ref_mask
    ref_type = "mask"
elif options.ref_soft_mask is not None:
    input_reference = options.ref_soft_mask
    ref_type = "soft_mask"
elif options.ref_image is not None:
    input_reference = options.ref_image
    ref_type = "image"
else:
    input_reference = options.ref_surface
    ref_type = "surface"

DL_synthesis_model = options.DL_synthesis

if not os.path.exists(input_reference):
    sys.exit(f"DNE: {input_reference}")

# Some directory names have spaces... oh well
input_photo_dir = options.input_photo_dir[0]
for i in range(len(options.input_photo_dir) - 1):
    input_photo_dir = input_photo_dir + " " + options.input_photo_dir[i + 1]
input_photo_dir = input_photo_dir + "/"

input_segmentation_dir = options.input_segmentation_dir[0]
for i in range(len(options.input_segmentation_dir) - 1):
    input_segmentation_dir = (
        input_segmentation_dir + " " + options.input_segmentation_dir[i + 1]
    )
input_segmentation_dir = input_segmentation_dir + "/"

reverse_lr = options.posterior_side
reverse_ap = options.posterior_to_anterior
slice_thickness = options.slice_thickness
photo_res = options.photo_resolution
allow_z_stretch = options.allow_z_stretch
n_cp_nonlin = options.n_cp_nonlin
if n_cp_nonlin > 0 and n_cp_nonlin < 4:
    raise Exception(
        "If you are using the nonlinear mode, the minimum number of control points is 5"
    )
# mesh_autoalign_target = options.mesh_autoalign_target
# mesh_manually_oriented = options.mesh_manually_oriented
stiffness_nonlin = options.stiffness_nonlin

if options.skip_flag:
    slice_thickness = slice_thickness * options.multiply_factor
    subject_id = os.path.basename(os.path.dirname(options.input_photo_dir[0]))
    ref_seg = os.path.join(
        os.getcwd(),
        "data",
        "uw_photo/recons/results_Henry/Results_hard",
        subject_id,
        f"{subject_id}_hard_manualLabel_merged.mgz",
    )

    if os.path.exists(ref_seg):
        x = my.MRIread(ref_seg, im_only=True)
    else:
        sys.exit(f"Ground Truth doesn't exist for subject {subject_id}")

    slice_idx = (
        np.argmax((x > 1).sum(0).sum(0)) - 2
    )  # subtracting 2 for padding
    start_idx = slice_idx % options.multiply_factor
else:
    start_idx = 0

# Outputs
output_directory = options.output_directory + "/"
output_photo_recon = output_directory + "photo_recon.mgz"
if ref_type == "surface":
    output_registered_reference = output_directory + "registered_reference.surf"
else:
    output_registered_reference = None

if os.path.isdir(output_directory) is False:
    os.makedirs(output_directory, exist_ok=True)

# Training
alternate_optimization = options.alternate_optimization

########################################################

# Constants
if DL_synthesis_model is None:
    RESOLUTIONS = [4, 2, 1, 0.5]
    STEPS = [300, 300, 300, 300]
    if FAST:
        STEPS = [10, 2, 2, 2]

    if RESOLUTIONS[-1] < photo_res:
        RESOLUTIONS[-1] = photo_res
else:
    RESOLUTIONS = [4, 2, 1]
    STEPS = [25, 25, 25]
    if FAST:
        STEPS = [10, 2, 2]

LR = 1.0
TOL = 1e-6

if ref_type == "mask":
    K_DICE_MRI = 0.95
    K_DICE_SLICES = 0.025
    K_NCC_SLICES = 0.025
    K_REGULARIZER = 0.00025
    K_NCC_INTERMODALITY = None
    K_SURFACE_TERM = None
    K_NONLINEAR = stiffness_nonlin
elif ref_type == "soft_mask":
    K_DICE_MRI = 0.8
    K_DICE_SLICES = 0.1
    K_NCC_SLICES = 0.1
    K_REGULARIZER = 0.1
    K_NCC_INTERMODALITY = None
    K_SURFACE_TERM = None
    K_NONLINEAR = stiffness_nonlin
elif ref_type == "image":
    K_NCC_INTERMODALITY = 0.95
    K_DICE_SLICES = 0.025
    K_NCC_SLICES = 0.0125
    K_REGULARIZER = 0.0025
    K_DICE_MRI = K_NCC_INTERMODALITY
    K_SURFACE_TERM = None
    K_NONLINEAR = stiffness_nonlin
else:  # surface
    K_SURFACE_TERM = 1.0
    K_DICE_SLICES = 0.025
    K_NCC_SLICES = 0.025
    K_REGULARIZER = 0.0025
    K_DICE_MRI = 0.95
    K_NCC_INTERMODALITY = None
    K_NONLINEAR = stiffness_nonlin

########################################################

print("Extracting slices from photographs")
d_i = glob.glob(input_photo_dir + "/*.jpg")
if len(d_i) == 0:
    d_i = glob.glob(input_photo_dir + "/*.tif")
if len(d_i) == 0:
    d_i = glob.glob(input_photo_dir + "/*.tiff")
if len(d_i) == 0:
    d_i = glob.glob(input_photo_dir + "/*.JPG")
if len(d_i) == 0:
    d_i = glob.glob(input_photo_dir + "/*.png")
d_i = sorted(d_i)

d_s = glob.glob(input_segmentation_dir + "/*.mat")
if len(d_s) == 0:
    d_s = glob.glob(input_segmentation_dir + "/*.npy")
d_s = sorted(d_s)

if reverse_ap:
    d_s = d_s[::-1]
    d_i = d_i[::-1]

Nphotos = len(d_i)

Iorig = []
Morig = []

all_croppings = []
total_slice_count = 0
for n in np.arange(Nphotos):
    X = np.flip(cv2.imread(d_i[n]), axis=-1)  # convert to RGB

    if d_s[n][-3:] == "mat":
        Y = scipy.io.loadmat(d_s[n])["LABELS"]
    else:
        Y = np.load(d_s[n])
    print(f"Photo {n + 1} has {len(np.unique(Y))-1} slices (CCs)") # Eugenio added -1 to account for zero
    total_slice_count += (len(np.unique(Y)) - 1)

    for l in 1 + np.arange(np.max(Y)):
        mask, cropping = my.cropLabelVol(Y == l, np.round(5 / photo_res))
        all_croppings.append(cropping)
        cropping[2] = 0
        cropping[5] = 3
        image = my.applyCropping(X, cropping)
        image = image * mask
        Iorig.append(image)
        Morig.append(mask)

print(f"Found {total_slice_count} slices in {Nphotos} photos")

########################################################

print(
    "Resampling to highest target resolution: " + str(RESOLUTIONS[-1]) + " mm"
)

Nslices0 = len(Iorig)
select_slices = np.arange(start_idx, Nslices0, options.multiply_factor)

if options.skip_flag:
    if slice_idx in select_slices:
        print(
            f"NOTE: {subject_id}, GT Slice {slice_idx}, Skip {options.multiply_factor}, Selected"
        )
    else:
        print(
            f"NOTE: {subject_id}, GT Slice {slice_idx}, Skip {options.multiply_factor}, NOT Selected"
        )

Nslices = len(select_slices)
Nscales = len(RESOLUTIONS)
I = []
M = []
for n in np.arange(start_idx, Nslices0, options.multiply_factor):
    Isl = cv2.resize(
        Iorig[n],
        None,
        fx=photo_res / RESOLUTIONS[-1],
        fy=photo_res / RESOLUTIONS[-1],
        interpolation=cv2.INTER_AREA,
    )
    Msl = cv2.resize(
        Morig[n].astype("float"),
        None,
        fx=photo_res / RESOLUTIONS[-1],
        fy=photo_res / RESOLUTIONS[-1],
        interpolation=cv2.INTER_AREA,
    )
    Isl[Msl == 0] = 0
    I.append(Isl)
    M.append(Msl)

########################################################

print("Coarse alignment and padding")

PAD_AP = 3  # Padding in A-P axis ensures that mask remains 100% in FOV
sizes = np.zeros([Nslices, 2])
for n in np.arange(Nslices):
    sizes[n, :] = M[n].shape
siz = np.round(1.5 * np.max(sizes, axis=0)).astype("int")
n_cp_nonlin = np.round(n_cp_nonlin * siz / np.max(siz)).astype("int")
Is = []
Ms = []
Affs = []

aff = np.array(
    [
        [0, -RESOLUTIONS[-1], 0, 0],
        [0, 0, -slice_thickness, 0],
        [-RESOLUTIONS[-1], 0, 0, 0],
        [0, 0, 0, 1],
    ]
)
if reverse_lr:
    aff[0, 1] = -aff[0, 1]
im = np.zeros([*siz, Nslices + 2 * PAD_AP, 3])
mask = np.zeros([*siz, Nslices + 2 * PAD_AP])
all_paddings = []
for n in np.arange(Nslices):
    idx1 = np.ceil(0.5 * (np.array(siz) - sizes[n, :])).astype("int")
    idx2 = (idx1 + sizes[n, :]).astype("int")
    im[idx1[0] : idx2[0], idx1[1] : idx2[1], n + PAD_AP, :] = I[n]
    mask[idx1[0] : idx2[0], idx1[1] : idx2[1], n + PAD_AP] = M[n]
    all_paddings.append(idx1[0:2] - all_croppings[n][0:2])

Is.append(im)
Ms.append(mask)
Affs.append(aff)
Nslices = Nslices + 2 * PAD_AP

########################################################

print("Building resolution pyramid")

for s in np.arange(Nscales - 2, -1, -1):
    for n in np.arange(Nslices):
        Isl = cv2.resize(
            Is[-1][:, :, n, :],
            None,
            fx=RESOLUTIONS[-1] / RESOLUTIONS[s],
            fy=RESOLUTIONS[-1] / RESOLUTIONS[s],
            interpolation=cv2.INTER_AREA,
        )
        Msl = cv2.resize(
            Ms[-1][:, :, n],
            None,
            fx=RESOLUTIONS[-1] / RESOLUTIONS[s],
            fy=RESOLUTIONS[-1] / RESOLUTIONS[s],
            interpolation=cv2.INTER_AREA,
        )
        if n == 0:
            im = np.zeros([*Msl.shape, Nslices, 3])
            mask = np.zeros([*Msl.shape, Nslices])

        im[:, :, n, :] = Isl
        mask[:, :, n] = Msl

    aff = np.zeros([4, 4])
    aff[0, 1] = -RESOLUTIONS[s]
    aff[1, 2] = -slice_thickness
    aff[2, 0] = -RESOLUTIONS[s]
    aff[3, 3] = 1
    if reverse_lr:
        aff[0, 1] = -aff[0, 1]
    aux = np.array(
        [
            [RESOLUTIONS[s] / RESOLUTIONS[-1]],
            [RESOLUTIONS[s] / RESOLUTIONS[-1]],
            [1],
        ]
    )
    aff[0:3, 3] = np.matmul(Affs[-1][0:3, 0:3], (0.5 * aux - 0.5))[:, 0]

    Is.insert(0, im)
    Ms.insert(0, mask)
    Affs.insert(0, aff)

# Switch from intensities to gradients, if we are using surfaces
Is_copy = np.copy(Is[-1])
if ref_type == "surface":
    for s in range(Nscales):
        erode_its = np.ceil(1.0 / RESOLUTIONS[s]).astype("int")
        for z in range(Nslices):
            M_ERODED = scipy.ndimage.binary_erosion(
                Ms[s][:, :, z] > 0.5, iterations=erode_its
            )
            for c in range(3):
                Is[s][:, :, z, c] = my.grad2d(Is[s][:, :, z, c]) * M_ERODED

########################################################

print("Reading and preprocessing reference")

if ref_type != "surface":
    REF, REFaff = my.MRIread(input_reference)
    if np.isnan(REF).any():
        print("There are NaNs here")
        REF[np.isnan(REF)] = 0
    REF = np.squeeze(REF)

    # Eugenio: added padding here; usefull when using hard masks from mris_fill
    pad = 10
    REF_padded = np.zeros(np.array(REF.shape) + 2 * pad)
    REF_padded[pad:-pad, pad:-pad, pad:-pad] = REF
    REFaff_padded = np.copy(REFaff)
    REFaff_padded[:-1, -1] = REFaff_padded[:-1, -1] - np.matmul(
        REFaff_padded[:-1, :-1], np.array([pad, pad, pad])
    )
    REF = REF_padded
    REFaff = REFaff_padded

    REF_orig = np.copy(REF)
    REFaff_orig = np.copy(REFaff)

    REF = np.squeeze(REF) / np.max(REF)
    if ref_type == "mask" or ref_type == "image":
        REF = (REF > 0).astype("float")

# Surfaces require quite a bit of extra work
else:

    input_mesh_converted = output_directory + "/input_mesh.surf"
    input_mesh_reoriented = output_directory + "/input_mesh_reoriented.surf"
    reoriented_mask_vol = output_directory + "/input_mesh_reoriented.filled.mgz"

    # fs_home = os.getenv("FREESURFER_HOME")
    # if fs_home is None:
    #     raise Exception(
    #         "FREESURFER_HOME variable not found; is FreeSurfer sourced?"
    #     )

    print("Converting reference mesh to FreeSurfer format")
    a = os.system(
        "mris_convert "
        + input_reference
        + " "
        + input_mesh_converted
        + " >/dev/null"
    )
    if a > 0:
        raise Exception(
            "error in mris_convert... is FreeSurfer sourced?"
        )

    print()
    # Read in and fill in missing metadata if needed (eg if STL file)
    P, T, meta = nib.freesurfer.read_geometry(input_mesh_converted, read_metadata=True)
    if meta['valid'][0]=='0':
        meta['valid'] = '1  # volume info valid'
        meta['filename'] = ''
        meta['volume'] = np.array([256,256,256]).astype(int)
        meta['voxelsize'] = np.array([1.0, 1.0, 1.0])
        meta['xras'] = np.array([-1.0, 0.0, 0.0])
        meta['yras'] = np.array([0.0, 0.0, -1.0])
        meta['zras'] = np.array([0.0, 1.0, 0.0])
        meta['cras'] = np.array([0.0, 0.0, 0.0])


    # Apply rotation using provided key vertices, if provided
    # https://towardsdatascience.com/the-definitive-procedure-for-aligning-two-sets-of-3d-points-with-the-kabsch-algorithm-a7ec2126c87e
    if options.mesh_reorient_with_indices is None:
        print('No indices were provided to reorient mesh; just copying over...')
        a = os.system(
            "cp "
            + input_mesh_converted
            + " "
            + input_mesh_reoriented
            + " >/dev/null "
        )
        if a > 0:
            raise Exception(
                "error copying mesh"
            )
    else:
        print('Reorienting mesh with provided vertices')
        idx = np.zeros(3).astype(int)
        aux = options.mesh_reorient_with_indices.split(",")
        for i in range(len(idx)):
            idx[i] = int(aux[i])
        K = P[idx,:]
        K = K - np.mean(K, axis=0)

        if True: # rough RAS aligment, already demeaned!
            Kref = np.array([[0, 85, -20],
                             [0, -80, -25],
                             [0, -5, 45]]).astype(float)
        else: # precomputed from rh.white
            Kref = np.array([[5.64194918,  77.57227325, 10.32956219],
                             [1.60726917, -90.65991211, -0.76444769],
                             [3.86025476, -13.81834793, 69.90812683]])
            Kref = Kref - np.mean(Kref, axis=0)

        H = np.transpose(Kref) @ K
        U, S, Vt = np.linalg.svd(H)
        if np.linalg.det(np.transpose(Vt) @ U) > 0:
            R = np.transpose(Vt) @ np.transpose(U)
        else:
            E = np.eye(3)
            E[2, 2] = -1
            R = np.transpose(Vt) @ (E @ np.transpose(U))

        P = P - np.mean(P, axis=0)
        P = P @ R
        meta["cras"][:] = 0
        nib.freesurfer.write_geometry(input_mesh_reoriented, P, T, volume_info=meta)

    # Fill in the mesh
    print("Filling in mesh to obtain binary volume")
    a = os.system(
        "mris_fill -r 1 "
        + input_mesh_reoriented
        + " "
        + output_directory
        + "/temp.mgz >/dev/null"
    )
    if a > 0:
        raise Exception("error in mris_fill... is FreeSurfer sourced?")
    # We pad a bit
    [img, aff] = my.MRIread(output_directory + "/temp.mgz")
    pad = 8
    img2 = np.zeros(np.array(img.shape) + 2 * pad)
    img2[pad:-pad, pad:-pad, pad:-pad] = img
    aff2 = aff
    aff2[:-1, -1] = aff2[:-1, -1] - np.squeeze(
        np.matmul(aff[:-1, :-1], pad * np.ones([3, 1]))
    )
    my.MRIwrite(img2, aff2, reoriented_mask_vol)
    os.system("rm -rf " + output_directory + "/temp.mgz >/dev/null")

    # original_mesh_decimated = (
    #     output_directory + "/input_mesh_with_header.decimated.surf"
    # )
    # original_mesh_full = output_directory + "/input_mesh_with_header.surf"
    # original_mask_vol = output_directory + "/input_mesh_with_header.filled.mgz"
    # reoriented_mask_vol = (
    #     output_directory + "/input_mesh_with_header.filled.reoriented.mgz"
    # )
    # reoriented_mesh_decimated = (
    #     output_directory + "/input_mesh_with_header.decimated.reoriented.surf"
    # )
    # reoriented_mesh_full = (
    #     output_directory + "/input_mesh_with_header.reoriented.surf"
    # )
    #
    # if mesh_manually_oriented:
    #     if not os.path.isfile(original_mesh_decimated):
    #         raise Exception(
    #             "Decimated mesh not found in output directory. Did you try running the code in automatic mode first? (see manual)"
    #         )
    #     if not os.path.isfile(original_mesh_full):
    #         raise Exception(
    #             "Full-res mesh not found in output directory. Did you try running the code in automatic mode first? (see manual)"
    #         )
    #     if not os.path.isfile(original_mask_vol):
    #         raise Exception(
    #             "Original mask (filled mesh) volume not found in output directory. Did you try running the code in automatic mode first? (see manual)"
    #         )
    #     if not os.path.isfile(reoriented_mask_vol):
    #         raise Exception(
    #             "Reoriented mask (filled mesh) volumes not found in output directory. Did you try running the code in automatic mode first? (see manual)"
    #         )
    #
    #     print("Estimating transform from original and rotated volumes")
    #     _, aff1 = my.MRIread(original_mask_vol)
    #     _, aff2 = my.MRIread(reoriented_mask_vol)
    #     T = np.matmul(aff2, np.linalg.inv(aff1))
    #
    #     Pfull, Tfull, meta_full = nib.freesurfer.read_geometry(
    #         original_mesh_full, read_metadata=True
    #     )
    #     Pfull += meta_full["cras"]  # ** CRUCIAL **
    #     meta_full["cras"][:] = 0
    #     Pfull_oriented = np.matmul(
    #         np.concatenate([Pfull, np.ones([Pfull.shape[0], 1])], axis=1),
    #         T.transpose(),
    #     )[:, :-1]
    #     nib.freesurfer.write_geometry(
    #         reoriented_mesh_full, Pfull_oriented, Tfull, volume_info=meta_full
    #     )
    #
    #     Pdec, Tdec, meta_dec = nib.freesurfer.read_geometry(
    #         original_mesh_decimated, read_metadata=True
    #     )
    #     Pdec += meta_dec["cras"]  # ** CRUCIAL **
    #     meta_dec["cras"][:] = 0
    #     Pdec_oriented = np.matmul(
    #         np.concatenate([Pdec, np.ones([Pdec.shape[0], 1])], axis=1),
    #         T.transpose(),
    #     )[:, :-1]
    #     nib.freesurfer.write_geometry(
    #         reoriented_mesh_decimated, Pdec_oriented, Tdec, volume_info=meta_dec
    #     )
    #
    # else:
    #     fs_home = os.getenv("FREESURFER_HOME")
    #     if fs_home is None:
    #         raise Exception(
    #             "FREESURFER_HOME variable not found; is FreeSurfer sourced?"
    #         )
    #
    #     # Load reference mesh and apply header, unless already there
    #     if os.path.isfile(output_directory + "/input_mesh_with_header.surf"):
    #         print(
    #             "Reference mesh with header already found in output directory; skipping computation"
    #         )
    #     else:
    #         print("Loading reference mesh and applying existing header")
    #         a = os.system(
    #             "mris_copy_header "
    #             + input_reference
    #             + " "
    #             + fs_home
    #             + "/subjects/bert/surf/rh.white "
    #             + output_directory
    #             + "/input_mesh_with_header.surf >/dev/null"
    #         )
    #         if a > 0:
    #             raise Exception(
    #                 "error in mris_copy_header... is FreeSurfer sourced?"
    #             )
    #
    #     # Fill the mesh to get a binary volume (useful in first iteration)
    #     if os.path.isfile(
    #         output_directory + "/input_mesh_with_header.filled.mgz"
    #     ):
    #         print(
    #             "Filled in mesh volume already found in output directory; skipping computation"
    #         )
    #     else:
    #         print("Filling in mesh to obtain binary volume")
    #         a = os.system(
    #             "mris_fill -r 1 "
    #             + output_directory
    #             + "/input_mesh_with_header.surf "
    #             + output_directory
    #             + "/temp.mgz >/dev/null"
    #         )
    #         if a > 0:
    #             raise Exception("error in mris_fill... is FreeSurfer sourced?")
    #         # We pad a bit
    #         [img, aff] = my.MRIread(output_directory + "/temp.mgz")
    #         pad = 8
    #         img2 = np.zeros(np.array(img.shape) + 2 * pad)
    #         img2[pad:-pad, pad:-pad, pad:-pad] = img
    #         aff2 = aff
    #         aff2[:-1, -1] = aff2[:-1, -1] - np.squeeze(
    #             np.matmul(aff[:-1, :-1], pad * np.ones([3, 1]))
    #         )
    #         my.MRIwrite(
    #             img2,
    #             aff2,
    #             output_directory + "/input_mesh_with_header.filled.mgz",
    #         )
    #         os.system("rm -rf " + output_directory + "/temp.mgz >/dev/null")
    #
    #     # Decimate mesh so coarse alignment doesn't take like a year...
    #     if os.path.isfile(
    #         output_directory + "/input_mesh_with_header.decimated.surf"
    #     ):
    #         print(
    #             "Decimated mesh volume already found in output directory; skipping computation"
    #         )
    #     else:
    #         print(
    #             "Decimating mesh - useful for some operations that would otherwise take forever"
    #         )
    #         a = os.system(
    #             "mris_remesh  -i "
    #             + output_directory
    #             + "/input_mesh_with_header.surf  -o "
    #             + output_directory
    #             + "/input_mesh_with_header.decimated.surf --nvert 10000 >/dev/null"
    #         )
    #         if a > 0:
    #             raise Exception("mris_remesh failed")
    #
    #     # OK now we are ready for automated  alignment
    #     if (
    #         os.path.isfile(reoriented_mesh_decimated)
    #         and os.path.isfile(reoriented_mesh_full)
    #         and os.path.isfile(reoriented_mask_vol)
    #     ):
    #         print(
    #             "Reoriented mesh and filled volume volume already found in output directory; skipping alignment"
    #         )
    #     else:
    #         print(
    #             "Aligning mesh to provided probabilistic atlas for automatic reorientation (can be a bit slow...)"
    #         )
    #
    #         # Read autoalign target reference, get pixels around boundary, and convert to ras
    #         target, target_aff = my.MRIread(mesh_autoalign_target)
    #         target = np.squeeze(target)
    #         idx = np.where((target > 0.49) & (target < 0.51))
    #         idx = [*idx, np.ones_like(idx[0])]
    #         b = np.stack(idx)
    #         ras = np.matmul(target_aff, b)[:-1, :]
    #         shift = np.mean(ras, axis=1)
    #         ras = ras - shift[:, np.newaxis]
    #         target_aff[0:-1, -1] = target_aff[0:-1, -1] - shift
    #         target_pc = np.array(ras).transpose()
    #
    #         # And the actual registration
    #         Pdec, Tdec, meta_dec = nib.freesurfer.read_geometry(
    #             output_directory + "input_mesh_with_header.decimated.surf",
    #             read_metadata=True,
    #         )
    #         Pdec += meta_dec["cras"]  # ** CRUCIAL **
    #         meta_dec["cras"][
    #             :
    #         ] = 0  # We can now easily write surfaces in stl or surf
    #         mesh_dec = trimesh.Trimesh(Pdec, Tdec, process=False)
    #         T, cost = trimesh.registration.mesh_other(
    #             mesh_dec,
    #             target_pc,
    #             samples=500,
    #             scale=False,
    #             icp_first=10,
    #             icp_final=25,
    #         )
    #
    #         # Write registered meshes at full and decimated resolution
    #         Pdec_oriented = np.matmul(
    #             np.concatenate([Pdec, np.ones([Pdec.shape[0], 1])], axis=1),
    #             T.transpose(),
    #         )[:, :-1]
    #         nib.freesurfer.write_geometry(
    #             reoriented_mesh_decimated,
    #             Pdec_oriented,
    #             Tdec,
    #             volume_info=meta_dec,
    #         )
    #
    #         Pfull, Tfull, meta_full = nib.freesurfer.read_geometry(
    #             output_directory + "input_mesh_with_header.surf",
    #             read_metadata=True,
    #         )
    #         Pfull += meta_full["cras"]  # ** CRUCIAL **
    #         meta_full["cras"][:] = 0
    #         Pfull_oriented = np.matmul(
    #             np.concatenate([Pfull, np.ones([Pfull.shape[0], 1])], axis=1),
    #             T.transpose(),
    #         )[:, :-1]
    #         nib.freesurfer.write_geometry(
    #             reoriented_mesh_full,
    #             Pfull_oriented,
    #             Tfull,
    #             volume_info=meta_full,
    #         )
    #
    #         # Write registered binary mask
    #         img, aff = my.MRIread(
    #             output_directory + "/input_mesh_with_header.filled.mgz"
    #         )
    #         aff2 = np.matmul(T, aff)
    #         my.MRIwrite(img, aff2, reoriented_mask_vol)

    # Read deformed surface and corresponding reference volume
    REF, REFaff = my.MRIread(reoriented_mask_vol)
    REF_orig = np.copy(REF)
    REFaff_orig = np.copy(REFaff)
    REF = np.squeeze(REF) / np.max(REF)
    REF = (REF > 0.5).astype("float")

    Pmesh, Tmesh, meta_mesh = nib.freesurfer.read_geometry(
        input_mesh_reoriented, read_metadata=True
    )
    Pmesh += meta_mesh["cras"]  # ** CRUCIAL **
    meta_mesh["cras"][:] = 0


    # And finally, take the gradient of the photos
    for s in range(Nscales):
        erode_its = np.ceil(1.0 / RESOLUTIONS[s]).astype("int")
        for z in range(Nslices):
            # M_ERODED = scipy.ndimage.binary_erosion(Ms[s][:, :, z] > .5, iterations=erode_its)
            for c in range(3):
                Is[s][:, :, z, c] = (
                    my.grad2d(Is[s][:, :, z, c]) / 255.0
                )  # * M_ERODED

########################################################

print("Center the centers of gravity in the origin")

if ref_type == "surface":
    cog_mesh_ras = np.mean(Pmesh, axis=0)
    Pmesh -= cog_mesh_ras
    REFaff[:-1, -1] = REFaff[:-1, -1] - cog_mesh_ras

else:
    idx = np.where(REF > 0.1)
    cog_mri_vox = np.array(
        [[np.mean(idx[0])], [np.mean(idx[1])], [np.mean(idx[2])]]
    )
    cog_mri_ras = my.vox2ras(cog_mri_vox, REFaff)
    REFaff[:-1, -1] = REFaff[:-1, -1] - np.squeeze(cog_mri_ras)

idx = np.where(Ms[-1] > 0)
cog_photo_vox = np.array(
    [[np.mean(idx[0])], [np.mean(idx[1])], [np.mean(idx[2])]]
)
cog_photo_ras = my.vox2ras(cog_photo_vox, Affs[-1])
for s in np.arange(Nscales):
    Affs[s][:-1, -1] = Affs[s][:-1, -1] - np.squeeze(cog_photo_ras)

########################################################

print("Optimization")

# Initialize
t = None
theta = None
shear = None
scaling = None
sz = None
t_reference = None
theta_reference = None
s_reference = None
field = None

# Go over resolutions / modes
if n_cp_nonlin[0] > 0:
    n_modes = 3
    print("We will be running 3 modes: rigid, affine, and nonlinear")
else:
    n_modes = 2
    print("We will be running 2 modes: rigid, and affine (skipping nonlinear)")

if options.rigid_only_for_photos:
    n_modes = 1
    print("We will only be running 1 mode only: rigid for everything")

for mode_idx in range(n_modes):

    if mode_idx == 0:
        allow_scaling_and_shear = False
        allow_nonlin = False
        print("########################################################")
        print("###    First pass: no scaling / shearing allowed     ###")
        print("########################################################")

        # If we're doing images,  we switch the mode to masks in the first iteration...'
        if ref_type == "image":
            ref_type_iteration = "mask"
            k_dice_mri_iteration = K_DICE_MRI
            k_ncc_intermodality_iteration = None
            k_surface_term_iteration = None
        else:
            ref_type_iteration = ref_type
            k_dice_mri_iteration = K_DICE_MRI
            k_ncc_intermodality_iteration = K_NCC_INTERMODALITY
            k_surface_term_iteration = K_SURFACE_TERM

    elif mode_idx == 1:
        allow_scaling_and_shear = True
        allow_nonlin = False
        print("########################################################")
        print("###    Second pass: scaling / shearing is allowed    ###")
        print("########################################################")

        # If we're doing images  we compute gradients and switch back to original data term
        if ref_type == "image":
            ref_type_iteration = ref_type
            k_dice_mri_iteration = None
            k_ncc_intermodality_iteration = K_NCC_INTERMODALITY
            k_surface_term_iteration = K_SURFACE_TERM

            if DL_synthesis_model is None:
                # Standard mode: we use NCC on gradient maps
                REF = my.grad3d(np.copy(REF_orig))

                for s in range(Nscales):
                    erode_its = np.ceil(1.0 / RESOLUTIONS[s]).astype("int")
                    for z in range(Nslices):
                        # M_ERODED = scipy.ndimage.binary_erosion(Ms[s][:, :, z] > .5, iterations=erode_its)
                        for c in range(3):
                            Is[s][:, :, z, c] = my.grad2d(
                                Is[s][:, :, z, c]
                            )  # * M_ERODED
                    if ref_type == "surface":
                        Is[s] = Is[s] / 255.0  # Otherwise loss goes bananas...
            else:
                # Experimental mode: deep learning based synthesis of MRI-like slices from the photos
                # (note that this requires providing a reference processed with SynthSR)
                REF = np.copy(REF_orig)
                tempdir = output_directory + "/temp/"
                os.system("rm -rf " + tempdir)
                os.mkdir(tempdir)
                tempdir2 = output_directory + "/temp2/"
                os.system("rm -rf " + tempdir2)
                os.mkdir(tempdir2)
                for z in range(Nslices):
                    cv2.imwrite(
                        tempdir + str(z) + ".png",
                        np.mean(Is[-1][:, :, z], axis=-1).astype("B"),
                    )
                scriptdir = os.path.dirname(
                    os.path.dirname(os.path.abspath(sys.argv[0]))
                )
                os.system(
                    scriptdir
                    + "/run_synthesis.sh "
                    + tempdir
                    + " "
                    + tempdir2
                    + " "
                    + DL_synthesis_model
                )
                for z in range(Nslices):
                    aux = cv2.imread(tempdir2 + str(z) + "_SynthSR.png").astype(
                        float
                    )
                    aux[Is[s][:, :, z, :] == 0] = 0
                    Is[s][:, :, z, :] = aux
                os.system("rm -rf " + tempdir)
                os.system("rm -rf " + tempdir2)
                # Build pyramid
                for s in np.arange(Nscales - 2, -1, -1):
                    for z in range(Nslices):
                        Isl = cv2.resize(
                            Is[-1][:, :, z, :],
                            None,
                            fx=RESOLUTIONS[-1] / RESOLUTIONS[s],
                            fy=RESOLUTIONS[-1] / RESOLUTIONS[s],
                            interpolation=cv2.INTER_AREA,
                        )
                        Isl[Ms[s][:, :, z] == 0] = 0
                        Is[s][:, :, z, :] = Isl

        else:
            ref_type_iteration = ref_type
            k_dice_mri_iteration = K_DICE_MRI
            k_ncc_intermodality_iteration = K_NCC_INTERMODALITY
            k_surface_term_iteration = K_SURFACE_TERM

    else:
        allow_scaling_and_shear = True
        allow_nonlin = True
        field = np.zeros([2, Nslices, *n_cp_nonlin])
        print("########################################################")
        print("###    Third pass: scaling / shearing / nonlinear    ###")
        print("########################################################")

        ref_type_iteration = ref_type
        k_dice_mri_iteration = K_DICE_MRI
        k_ncc_intermodality_iteration = K_NCC_INTERMODALITY
        k_surface_term_iteration = K_SURFACE_TERM

    for res in range(len(RESOLUTIONS)):

        print(
            "Working on resolution %d of %d (%.2f mm): %d iterations "
            % (res + 1, len(RESOLUTIONS), RESOLUTIONS[res], STEPS[res])
        )

        if ref_type == "surface":
            allow_s_reference = False
            if FAST:
                ratio = int(Pmesh.shape[0] / 100000)
                ref_surface = Pmesh[::ratio, :]
            else:
                ref_surface = Pmesh
        else:
            ref_surface = None

        volres = np.sqrt(np.sum(REFaff[:, :-1] ** 2, axis=0))
        sigmas = 0.5 * RESOLUTIONS[res] / volres
        REFsmooth = scipy.ndimage.gaussian_filter(REF, sigmas)
        allow_s_reference = ref_type == "soft_mask"

        model = PRnets.PhotoAligner(
            Is[res],
            Ms[res],
            Affs[res],
            REFsmooth,
            REFaff,
            ref_surface,
            t_ini=t,
            theta_ini=theta,
            shear_ini=shear,
            scaling_ini=scaling,
            sz_ini=sz,
            allow_sz=allow_z_stretch,
            t_reference_ini=t_reference,
            theta_reference_ini=theta_reference,
            s_reference_ini=s_reference,
            allow_s_reference=allow_s_reference,
            ref_type=ref_type_iteration,
            field_ini=field,
            allow_nonlin=allow_nonlin,
            k_dice_mri=k_dice_mri_iteration,
            k_ncc_intermodality=k_ncc_intermodality_iteration,
            k_surface_term=k_surface_term_iteration,
            k_dice_slices=K_DICE_SLICES,
            k_ncc_slices=K_NCC_SLICES,
            k_regularizer=K_REGULARIZER,
            pad_ignore=PAD_AP,
            device=device,
            allow_scaling_and_shear=allow_scaling_and_shear,
            k_nonlinear=K_NONLINEAR,
        )

        if FAST:
            optimizer = torch.optim.SGD(model.parameters(), lr=10 * LR)
        else:
            if alternate_optimization:
                optimizer2d = LBFGS.FullBatchLBFGS(model.parameters2d())
                optimizer3d = LBFGS.FullBatchLBFGS(model.parameters3d())
            else:
                optimizer = LBFGS.FullBatchLBFGS(model.parameters())

        loss_old = 1e10

        trigger_times = 0
        for epoch in range(STEPS[res]):

            # Compute loss with forward pass
            loss = model()[0]

            # optimize with BFGS
            if alternate_optimization:

                def closure2d():
                    optimizer2d.zero_grad()
                    loss = model()[0]
                    return loss

                def closure3d():
                    optimizer3d.zero_grad()
                    loss = model()[0]
                    return loss

            else:

                def closure():
                    optimizer.zero_grad()
                    loss = model()[0]
                    return loss

            # optimizer.step(closure)
            if epoch == 1:
                loss.backward()

            if alternate_optimization:
                options2d = {
                    "closure": closure2d,
                    "current_loss": loss,
                    "max_ls": 75,
                }
                options3d = {
                    "closure": closure3d,
                    "current_loss": loss,
                    "max_ls": 75,
                }

                if epoch % 10 < 5:
                    (
                        loss,
                        _,
                        lr,
                        _,
                        F_eval,
                        G_eval,
                        _,
                        fail_flag,
                    ) = optimizer2d.step(options2d)
                else:
                    (
                        loss,
                        _,
                        lr,
                        _,
                        F_eval,
                        G_eval,
                        _,
                        fail_flag,
                    ) = optimizer3d.step(options3d)
            else:
                options = {
                    "closure": closure,
                    "current_loss": loss,
                    "max_ls": 75,
                }
                loss, _, lr, _, F_eval, G_eval, _, fail_flag = optimizer.step(
                    options
                )

            if fail_flag:
                print("Line search failed")
                break

            # print step info
            loss = loss.cpu().detach().numpy()
            print("   Step %d, loss = %.10f" % (epoch + 1, loss), flush=True)

            if ((loss_old - loss) < TOL) and not FAST:
                trigger_times += 1

                if trigger_times >= 25:
                    print(
                        "   Decrease in loss below tolerance limit for the last 25 steps"
                    )
                    break
            else:
                trigger_times = 0

            loss_old = loss

        # Retrieve model parameters
        t = model.t.cpu().detach().numpy()
        theta = model.theta.cpu().detach().numpy()
        shear = model.shear.cpu().detach().numpy()
        scaling = model.scaling.cpu().detach().numpy()
        sz = model.sz.cpu().detach().numpy()
        t_reference = model.t_reference.cpu().detach().numpy()
        theta_reference = model.theta_reference.cpu().detach().numpy()
        s_reference = model.s_reference.cpu().detach().numpy()
        if allow_nonlin:
            field = model.field.cpu().detach().numpy()

        # In the last resolution level, retrieve results before deleting the model
        if res == (len(RESOLUTIONS) - 1) and mode_idx == n_modes - 1:

            model.photo_vol = torch.Tensor(Is_copy).to(
                device
            )  # TODO: I had a division by np.max(Is_copy) but got rid of it...
            model.photo_rearranged = torch.unsqueeze(
                model.photo_vol.permute(3, 0, 1, 2), dim=0
            ).to(model.device)

            if ref_type == "surface":
                (
                    _,
                    photo_resampled,
                    photo_aff,
                    mri_aff_combined,
                    Rt,
                    TvoxPhotos,
                ) = model()
                Rt = Rt.cpu().detach().numpy()
            else:
                (
                    _,
                    photo_resampled,
                    photo_aff,
                    mri_aff_combined,
                    _,
                    TvoxPhotos,
                ) = model()

            TvoxPhotos = TvoxPhotos.cpu().detach().numpy()
            mri_aff_combined = mri_aff_combined.cpu().detach().numpy()
            photo_resampled = photo_resampled.cpu().detach().numpy()
            photo_aff = photo_aff.cpu().detach().numpy()

        # Free up memory
        if alternate_optimization:
            del optimizer2d
            del optimizer3d
        else:
            del optimizer
        del model

########################################################

print("Writing results to disk")

if ref_type == "surface":

    my.MRIwrite(photo_resampled, photo_aff, output_photo_recon)

    Pmesh_rotated = np.matmul(
        np.concatenate([Pmesh, np.ones([Pmesh.shape[0], 1])], axis=1),
        Rt.transpose(),
    )[:, :-1]
    nib.freesurfer.write_geometry(
        output_registered_reference, Pmesh_rotated, Tmesh, volume_info=meta_mesh
    )

    reg_mask = output_directory + "registered_reference.mgz"
    my.MRIwrite(REF_orig, mri_aff_combined, reg_mask)

    print(
        "freeview -v %s -v %s -f %s"
        % (
            output_photo_recon,
            reg_mask,
            output_registered_reference,
        )
    )

else:
    # Unless reference is soft, go back to original RAS space of reference before writing photo volume
    if ref_type == "soft_mask":
        my.MRIwrite(photo_resampled, photo_aff, output_photo_recon)
        reg_mask = output_directory + "registered_reference.mgz"
        my.MRIwrite(REF_orig, mri_aff_combined, reg_mask)
        print("freeview %s %s" % (output_photo_recon, reg_mask))

    else:
        T = np.matmul(mri_aff_combined, np.linalg.inv(REFaff_orig))
        Tinv = np.linalg.inv(T)
        my.MRIwrite(
            photo_resampled, np.matmul(Tinv, photo_aff), output_photo_recon
        )
        print("freeview %s %s" % (output_photo_recon, input_reference))

if "TvoxPhotos" in locals():
    try:
        np.save(output_directory + "slice_matrix_M.npy", TvoxPhotos)
        np.save(output_directory + "all_paddings.npy", all_paddings)
    except:
        print("FAIL: TvoxPhotos could not be saved")
else:
    print("DNE: TvoxPhotos does not exist")

print("All done!")
