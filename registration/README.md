# Rigid Registration of Paired Pre- and Post-Contrast MRI

This directory contains the script used to perform rigid registration of paired pre- and post-contrast T1-weighted MRI before multi-channel nnU-Net segmentation.

## Rationale

Paired pre- and post-contrast MRI must occupy the same spatial geometry when supplied to nnU-Net as separate input channels. Differences in patient positioning or image acquisition can result in anatomical misalignment between the two images even when they originate from the same examination.

The registration workflow therefore aligns the post-contrast MRI to the corresponding pre-contrast MRI before the images are used as paired model inputs.

## Method

The pre-contrast T1-weighted MRI is treated as the fixed reference image and the corresponding post-contrast MRI as the moving image.

Registration is performed using a three-dimensional Euler rigid-body transformation with six degrees of freedom:

- three rotations;
- three translations.

No scaling, shearing or nonlinear deformation is permitted.

The transformation is initialised from the physical image geometry. Mattes mutual information is used as the similarity metric because the intensity distributions of the pre- and post-contrast images differ following contrast administration.

Registration uses random sampling of 20% of voxels with a fixed random seed and a three-level multi-resolution scheme with shrink factors of `4`, `2` and `1`. Gaussian smoothing sigmas of `2`, `1` and `0` mm are applied at the corresponding levels.

Optimisation is performed using regular-step gradient descent.

After estimation of the rigid transformation, the original post-contrast image is resampled onto the pre-contrast image grid using linear interpolation. The pre-contrast image is not transformed.

The resulting paired images therefore have matching dimensions, voxel spacing, origin and direction. Matching image geometry does not itself demonstrate anatomical alignment, and registration outputs should therefore be visually reviewed before subsequent model development or evaluation.

## Requirements

- Python 3
- pandas
- SimpleITK

The required packages can be installed using:

```bash
pip install pandas SimpleITK
```

## Input structure

By default, the script expects a dataset directory containing separate `Pre` and `Post` directories:

```text
Dataset_Name/
├── Pre/
│   ├── Case001_T1_PRE.nii.gz
│   ├── Case002_T1_PRE.nii.gz
│   └── ...
└── Post/
    ├── Case001_T1_POST.nii.gz
    ├── Case002_T1_POST.nii.gz
    └── ...
```

Each pre-contrast image must have a corresponding post-contrast image with the same case identifier.

The default expected filename suffixes are:

```text
_T1_PRE.nii.gz
_T1_POST.nii.gz
```

## Usage

Run the registration pipeline by specifying the dataset directory:

```bash
python register_post_to_pre_rigid.py \
    --base_dir /path/to/Dataset_Name
```

The default `Pre` and `Post` directory names can be changed if required:

```bash
python register_post_to_pre_rigid.py \
    --base_dir /path/to/Dataset_Name \
    --pre_dir_name Pre \
    --post_dir_name Post
```

## nnU-Net output

For each successfully registered examination, the script writes paired images in nnU-Net-compatible channel format:

```text
Case001_0000.nii.gz    # original pre-contrast MRI
Case001_0001.nii.gz    # registered post-contrast MRI in pre-contrast space
```

The pre-contrast image therefore defines the reference geometry for both channels.

A left–right spatial coordinate channel can subsequently be generated as channel `0002` using the coordinate-channel script provided elsewhere in this repository:

```text
Case001_0000.nii.gz    # pre-contrast MRI
Case001_0001.nii.gz    # registered post-contrast MRI
Case001_0002.nii.gz    # left-right spatial coordinate channel
```

## Registration outputs

The default workflow creates:

```text
Dataset_Name/
├── Registered_nnUNet/
│   ├── Case001_0000.nii.gz
│   ├── Case001_0001.nii.gz
│   └── ...
├── Registration_Transforms/
│   ├── Case001_POST_to_PRE_rigid.tfm
│   └── ...
├── Pairing_Audit.csv
└── Registration_Mapping.csv
```

`Pairing_Audit.csv` records whether a complete pre/post pair was identified for each case.

`Registration_Mapping.csv` records processing status, transformation parameters, optimisation information and checks confirming whether the registered post-contrast image occupies the same geometry as the corresponding pre-contrast image.

The estimated rigid transformation for each examination is retained as a SimpleITK `.tfm` file for reproducibility.

## Visual quality control

Optional before- and after-registration checkerboard volumes can be generated using:

```bash
python register_post_to_pre_rigid.py \
    --base_dir /path/to/Dataset_Name \
    --save_checkerboards
```

These volumes are written to:

```text
Registration_QC/
```

They can be inspected using software such as ITK-SNAP or 3D Slicer to assess anatomical correspondence between the paired images.

The optimisation metric and estimated transformation parameters are retained as processing provenance and should not, by themselves, be interpreted as quantitative measures of registration accuracy.

## Error handling

The script checks that each pre-contrast image has a corresponding post-contrast image before registration begins. Incomplete pairs cause processing to stop rather than allowing cases to be silently omitted.

Following registration, the script verifies that the registered post-contrast image matches the pre-contrast image in:

- dimensions;
- voxel spacing;
- origin;
- direction.

Individual registration failures are recorded in `Registration_Mapping.csv`, and the script returns a non-zero exit status if any registration fails.

## Thesis methodology

This rigid-registration workflow was used to establish spatial correspondence between paired pre- and post-contrast MRI in the Nigerian pituitary adenoma datasets before paired-input nnU-Net model development and evaluation. The post-contrast MRI was registered to the corresponding pre-contrast MRI using a six-degree-of-freedom Euler transformation with Mattes mutual information, followed by resampling onto the pre-contrast image grid using linear interpolation. Registration outputs were visually reviewed before subsequent model evaluation.
