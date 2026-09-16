# Left–Right Spatial Coordinate Channel

This directory contains the script used to generate the left–right spatial coordinate channel incorporated into selected nnU-Net models in this project.

## Rationale

During development of multi-label segmentation models, confusion was observed between corresponding left- and right-sided anatomical structures, particularly the internal carotid arteries and cavernous sinuses. A spatial coordinate channel was therefore introduced to provide the network with explicit information about position along the patient left–right axis.

## Method

For each 3D MRI volume, the physical left–right coordinate of every voxel is calculated from the image origin, voxel spacing and direction matrix using the SimpleITK physical coordinate convention.

The coordinate volume is normalised to the range `[-1, 1]` by default and retains the size, spacing, origin and direction of the corresponding MRI volume.

The resulting volume is supplied to nnU-Net as an additional image channel.

## Requirements

- Python 3
- NumPy
- SimpleITK

The required packages can be installed using:

```bash
pip install numpy SimpleITK
```

## Input structure

The script expects an nnU-Net raw dataset directory containing `imagesTr` and, where applicable, `imagesTs`:

```text
DatasetXXX_Name/
├── imagesTr/
│   ├── Case001_0000.nii.gz
│   ├── Case002_0000.nii.gz
│   └── ...
├── imagesTs/
│   ├── Case101_0000.nii.gz
│   └── ...
└── dataset.json
```

Channel `0000` is used as the spatial reference for generating the coordinate volume.

## Usage

For a model in which the MRI occupies channel `0000` and the left–right coordinate volume occupies channel `0001`:

```bash
python generate_lr_coordinate_channel.py \
    --dataset_dir /path/to/nnUNet_raw/DatasetXXX_Name \
    --lr_channel_index 1
```

This generates files such as:

```text
Case001_0000.nii.gz    # MRI
Case001_0001.nii.gz    # left-right coordinate channel
```

For paired pre- and post-contrast models, where channels `0000` and `0001` are occupied by the two MRI inputs, the coordinate channel can instead be assigned to channel `0002`:

```bash
python generate_lr_coordinate_channel.py \
    --dataset_dir /path/to/nnUNet_raw/DatasetXXX_Name \
    --lr_channel_index 2
```

The resulting input structure is:

```text
Case001_0000.nii.gz    # pre-contrast MRI
Case001_0001.nii.gz    # post-contrast MRI
Case001_0002.nii.gz    # left-right coordinate channel
```

## Optional arguments

Use `--dry_run` to inspect which files would be generated without writing them:

```bash
python generate_lr_coordinate_channel.py \
    --dataset_dir /path/to/nnUNet_raw/DatasetXXX_Name \
    --lr_channel_index 1 \
    --dry_run
```

Use `--no_normalize` to retain the physical coordinate values instead of normalising them to `[-1, 1]`.

Existing coordinate-channel files are not overwritten.

## Thesis methodology

The coordinate channel was used to provide explicit left–right positional information for models segmenting bilateral anatomical structures. Full methodological details and experimental evaluation are described in the accompanying thesis.
