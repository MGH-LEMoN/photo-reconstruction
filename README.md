# photo-reconstruction
Photo Reconstruction Project

#### Installation Instructions

1. `python3 -m venv /path/to/new/virtual/environment` [Instructions](https://docs.python.org/3/library/venv.html)
2. `source /path/to/new/virtual/environment/bin/activate`
3. `git clone https://github.com/hvgazula/photo-reconstruction.git`
4. `pip install --upgrade pip`
5. `cd photo-reconstruction`
6. `pip install -r requirements.txt`

#### Usage Instructions
`PYTHONPATH=/path/to/photo-reconstruction/repo python scripts/3d_photo_reconstruction.py -h`

```
usage: 3d_photo_reconstruction.py [-h] --input_photo_dir
                                  [INPUT_PHOTO_DIR [INPUT_PHOTO_DIR ...]]
                                  --input_segmentation_dir
                                  [INPUT_SEGMENTATION_DIR [INPUT_SEGMENTATION_DIR ...]]
                                  [--ref_mask REF_MASK]
                                  [--ref_surface REF_SURFACE]
                                  [--ref_image REF_IMAGE]
                                  [--ref_soft_mask REF_SOFT_MASK]
                                  [--mesh_autoalign_target MESH_AUTOALIGN_TARGET]
                                  [--mesh_manually_oriented]
                                  [--photos_of_posterior_side]
                                  [--order_posterior_to_anterior]
                                  [--allow_z_stretch] --slice_thickness
                                  SLICE_THICKNESS --photo_resolution
                                  PHOTO_RESOLUTION [--n_cp_nonlin N_CP_NONLIN]
                                  [--stiffness_nonlin STIFFNESS_NONLIN]
                                  --output_directory OUTPUT_DIRECTORY
                                  [--gpu GPU]

Code for 3D photo reconstruction (Tregidgo, ..., & Iglesias, MICCAI 2020

optional arguments:
  -h, --help            show this help message and exit
  --input_photo_dir [INPUT_PHOTO_DIR [INPUT_PHOTO_DIR ...]]
                        Directory with input photos (required)
  --input_segmentation_dir [INPUT_SEGMENTATION_DIR [INPUT_SEGMENTATION_DIR ...]]
                        Directory with input slab masks / segmentations
                        (required)
  --ref_mask REF_MASK   Reference binary mask
  --ref_surface REF_SURFACE
                        Reference surface file
  --ref_image REF_IMAGE
                        Reference image file
  --ref_soft_mask REF_SOFT_MASK
                        Reference soft mask
  --mesh_autoalign_target MESH_AUTOALIGN_TARGET
                        Probabilistic atlas to globally initialize mesh
                        rotation
  --mesh_manually_oriented
                        Use this flag if you manually oriented the filled mesh
                        (please see manual)
  --photos_of_posterior_side
                        Use when photos are taken of posterior side of slabs
                        (default is anterior side)
  --order_posterior_to_anterior
                        Use when photos are ordered from posterior to anterior
                        (default is anterior to posterior)
  --allow_z_stretch     Use to adjust the slice thickness to best match the
                        reference. You should probably *never* use this with
                        soft references
  --slice_thickness SLICE_THICKNESS
                        Slice thickness in mm
  --photo_resolution PHOTO_RESOLUTION
                        Resolution of the photos in mm
  --n_cp_nonlin N_CP_NONLIN
                        number of control points for within slice nonlinear
                        deformation of the photos, along largest dimension of
                        image. You should probably *never* use this with soft
                        references
  --stiffness_nonlin STIFFNESS_NONLIN
                        stiffness of the nonlinear deformation
  --output_directory OUTPUT_DIRECTORY
                        Output directory with reconstructed photo volume and
                        reference
  --gpu GPU             Index of GPU to use
  ```

#### Running Example
```
PYTHONPATH=/path/to/photo-reconstruction/repo python scripts/3d_photo_reconstruction.py \
        --input_photo_dir <input_dir> \
        --input_segmentation_dir <segmentation_dir> \
        --ref_surface <reference_surface> \
        --mesh_autoalign_target <target> \
        [--photos_of_posterior_side] \
        [--allow_z_stretch] \
        --slice_thickness <thickness> \
        --photo_resolution <resolution>
```