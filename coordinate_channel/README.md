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
