#!/usr/bin/env python3
"""
Generate a left-right spatial coordinate channel for nnU-Net datasets.

For each MRI volume stored as channel 0000, this script generates an
additional image channel containing the physical left-right coordinate
of each voxel. Coordinates are calculated from the image origin, voxel
spacing, and direction matrix using the SimpleITK/LPS coordinate system.

By default, the coordinate values are normalised independently for each
image to the range [-1, 1].

The generated image retains the size, spacing, origin, and direction of
the corresponding MRI volume.

Example
-------
python generate_lr_coordinate_channel.py \
    --dataset_dir /path/to/nnUNet_raw/DatasetXXX_Name \
    --lr_channel_index 1
"""

import argparse
import os

import numpy as np
import SimpleITK as sitk


def make_lr_channel_image(
    img: sitk.Image,
    normalize: bool = True,
) -> sitk.Image:
    """
    Create a left-right spatial coordinate channel from a 3D image.

    Parameters
    ----------
    img : sitk.Image
        Input 3D image defining the spatial geometry.
    normalize : bool, optional
        If True, normalise the physical left-right coordinates to
        [-1, 1]. Default is True.

    Returns
    -------
    sitk.Image
        Float32 image containing the left-right coordinate values and
        retaining the geometry of the input image.
    """
    if img.GetDimension() != 3:
        raise ValueError(
            f"Expected a 3D image, but received {img.GetDimension()}D."
        )

    size_x, size_y, size_z = img.GetSize()

    origin = np.asarray(img.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    direction = np.asarray(
        img.GetDirection(), dtype=np.float64
    ).reshape(3, 3)

    # Construct voxel indices in SimpleITK x, y, z order.
    x = np.arange(size_x, dtype=np.float64)
    y = np.arange(size_y, dtype=np.float64)
    z = np.arange(size_z, dtype=np.float64)

    grid_x, grid_y, grid_z = np.meshgrid(
        x,
        y,
        z,
        indexing="ij",
    )

    scaled_indices = np.stack(
        [
            grid_x * spacing[0],
            grid_y * spacing[1],
            grid_z * spacing[2],
        ],
        axis=0,
    )

    # Physical coordinates follow:
    # physical_point = origin + direction @ (index * spacing)
    #
    # In the SimpleITK LPS convention, the first physical coordinate
    # represents the patient left-right axis (+X towards patient left).
    physical_x = (
        origin[0]
        + direction[0, 0] * scaled_indices[0]
        + direction[0, 1] * scaled_indices[1]
        + direction[0, 2] * scaled_indices[2]
    )

    if normalize:
        minimum = float(physical_x.min())
        maximum = float(physical_x.max())

        if maximum - minimum > 1e-6:
            physical_x = (
                2.0 * (physical_x - minimum) / (maximum - minimum)
                - 1.0
            )
        else:
            physical_x[:] = 0.0

    # NumPy arrays passed to SimpleITK use z, y, x ordering.
    lr_array = np.transpose(
        physical_x.astype(np.float32),
        (2, 1, 0),
    )

    lr_image = sitk.GetImageFromArray(lr_array)
    lr_image.CopyInformation(img)

    return sitk.Cast(lr_image, sitk.sitkFloat32)


def process_folder(
    folder: str,
    lr_channel_index: int,
    normalize: bool,
    dry_run: bool,
) -> int:
    """
    Generate coordinate channels for all channel-0000 images in a folder.
    """
    if not os.path.isdir(folder):
        print(f"Skipping missing directory: {folder}")
        return 0

    input_files = sorted(
        filename
        for filename in os.listdir(folder)
        if filename.endswith("_0000.nii.gz")
    )

    if not input_files:
        print(f"No *_0000.nii.gz files found in: {folder}")
        return 0

    number_written = 0

    for filename in input_files:
        input_path = os.path.join(folder, filename)
        case_id = filename.removesuffix("_0000.nii.gz")

        output_name = (
            f"{case_id}_{lr_channel_index:04d}.nii.gz"
        )
        output_path = os.path.join(folder, output_name)

        if os.path.exists(output_path):
            print(f"Skipping existing file: {output_name}")
            continue

        if dry_run:
            print(f"Would write: {output_path}")
            number_written += 1
            continue

        image = sitk.ReadImage(input_path)

        lr_image = make_lr_channel_image(
            image,
            normalize=normalize,
        )

        sitk.WriteImage(lr_image, output_path)

        print(f"Wrote: {output_name}")
        number_written += 1

    return number_written


def main() -> None:
    """Parse command-line arguments and generate coordinate channels."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a left-right spatial coordinate channel for "
            "nnU-Net training and test images."
        )
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help=(
            "Path to the nnUNet_raw DatasetXXX_Name directory."
        ),
    )

    parser.add_argument(
        "--lr_channel_index",
        type=int,
        default=1,
        help=(
            "nnU-Net channel index assigned to the coordinate channel "
            "(default: 1, producing *_0001.nii.gz)."
        ),
    )

    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help=(
            "Retain physical coordinate values instead of normalising "
            "them to [-1, 1]."
        ),
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Display the files that would be generated without writing "
            "them."
        ),
    )

    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    images_tr = os.path.join(dataset_dir, "imagesTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")

    normalize = not args.no_normalize

    print(f"Dataset: {dataset_dir}")
    print(f"LR channel index: {args.lr_channel_index}")
    print(f"Normalize to [-1, 1]: {normalize}")
    print(f"Dry run: {args.dry_run}")
    print()

    training_count = process_folder(
        images_tr,
        args.lr_channel_index,
        normalize,
        args.dry_run,
    )

    test_count = process_folder(
        images_ts,
        args.lr_channel_index,
        normalize,
        args.dry_run,
    )

    total_count = training_count + test_count

    print()

    if args.dry_run:
        print(
            f"Dry run complete. Would write {total_count} files."
        )
    else:
        print(
            f"Complete. Wrote {total_count} coordinate-channel files."
        )


if __name__ == "__main__":
    main()
