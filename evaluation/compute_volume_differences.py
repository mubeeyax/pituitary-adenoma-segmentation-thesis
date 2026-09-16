#!/usr/bin/env python3
"""
Compute label-wise volume differences between manually delineated reference
labels and predicted segmentation labels stored as NIfTI volumes.

For each case and requested label, the script reports:
    - reference voxel count
    - predicted voxel count
    - reference volume (mm^3)
    - predicted volume (mm^3)
    - absolute volume difference (mm^3)
    - signed percentage volume difference relative to the reference volume

Positive percentage differences indicate overestimation of the reference
volume; negative values indicate underestimation.

Reference and prediction files are matched by filename.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def parse_labels(value: str):
    """Parse comma-separated integer labels, e.g. '1,2,3,4,5,6,7,8'."""
    try:
        labels = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Labels must be comma-separated integers."
        ) from exc

    if not labels:
        raise argparse.ArgumentTypeError("At least one label is required.")

    return labels


def read_label_map(path: Path):
    """Read a 3D NIfTI label map and return image, array and voxel volume."""
    image = sitk.ReadImage(str(path))

    if image.GetDimension() != 3:
        raise ValueError(
            f"{path.name}: expected a 3D label map, "
            f"but found {image.GetDimension()}D."
        )

    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()  # SimpleITK order: x, y, z
    voxel_volume_mm3 = float(np.prod(spacing))

    return image, array, voxel_volume_mm3


def label_volume_mm3(
    label_array: np.ndarray,
    label_value: int,
    voxel_volume_mm3: float,
):
    """Return voxel count and physical volume for one label."""
    voxel_count = int(np.count_nonzero(label_array == label_value))
    volume = float(voxel_count * voxel_volume_mm3)
    return voxel_count, volume


def geometry_matches(
    reference: sitk.Image,
    prediction: sitk.Image,
    tolerance: float = 1e-6,
) -> bool:
    """Check whether reference and prediction occupy the same voxel grid."""
    if reference.GetSize() != prediction.GetSize():
        return False

    for reference_values, prediction_values in (
        (reference.GetSpacing(), prediction.GetSpacing()),
        (reference.GetOrigin(), prediction.GetOrigin()),
        (reference.GetDirection(), prediction.GetDirection()),
    ):
        if not all(
            abs(a - b) <= tolerance
            for a, b in zip(reference_values, prediction_values)
        ):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute label-wise absolute and percentage volume differences "
            "between reference and predicted NIfTI segmentations."
        )
    )
    parser.add_argument(
        "--reference_dir",
        required=True,
        help="Directory containing manually delineated reference label maps.",
    )
    parser.add_argument(
        "--prediction_dir",
        required=True,
        help="Directory containing predicted label maps.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path for the output CSV file.",
    )
    parser.add_argument(
        "--labels",
        type=parse_labels,
        default=parse_labels("1,2,3,4,5,6,7,8"),
        help="Comma-separated label values (default: 1,2,3,4,5,6,7,8).",
    )
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir).expanduser().resolve()
    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    labels = args.labels

    if not reference_dir.is_dir():
        raise FileNotFoundError(
            f"Reference directory does not exist: {reference_dir}"
        )
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction directory does not exist: {prediction_dir}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    reference_files = sorted(reference_dir.glob("*.nii.gz"))
    if not reference_files:
        print(
            f"No .nii.gz files found in {reference_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = []
    missing_predictions = []

    for reference_path in reference_files:
        prediction_path = prediction_dir / reference_path.name

        if not prediction_path.exists():
            missing_predictions.append(reference_path.name)
            continue

        reference_img, reference_array, reference_voxel_volume = (
            read_label_map(reference_path)
        )
        prediction_img, prediction_array, prediction_voxel_volume = (
            read_label_map(prediction_path)
        )

        if not geometry_matches(reference_img, prediction_img):
            raise ValueError(
                f"{reference_path.name}: reference and prediction do not "
                "occupy the same image geometry. Verify size, spacing, "
                "origin and direction before calculating volume differences."
            )

        for label_value in labels:
            reference_voxels, reference_volume = label_volume_mm3(
                reference_array,
                label_value,
                reference_voxel_volume,
            )
            predicted_voxels, predicted_volume = label_volume_mm3(
                prediction_array,
                label_value,
                prediction_voxel_volume,
            )

            absolute_difference = abs(
                predicted_volume - reference_volume
            )

            if reference_volume == 0:
                percentage_difference = ""
            else:
                percentage_difference = (
                    (predicted_volume - reference_volume)
                    / reference_volume
                    * 100.0
                )

            rows.append(
                {
                    "case": reference_path.name.removesuffix(".nii.gz"),
                    "label_value": label_value,
                    "reference_voxels": reference_voxels,
                    "predicted_voxels": predicted_voxels,
                    "reference_volume_mm3": reference_volume,
                    "predicted_volume_mm3": predicted_volume,
                    "absolute_volume_difference_mm3": absolute_difference,
                    "percentage_volume_difference": percentage_difference,
                    "reference_voxel_volume_mm3": reference_voxel_volume,
                    "predicted_voxel_volume_mm3": prediction_voxel_volume,
                }
            )

    if not rows:
        print(
            "No paired cases were available for volume analysis.",
            file=sys.stderr,
        )
        sys.exit(2)

    fieldnames = list(rows[0].keys())

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(reference_files) - len(missing_predictions)} case(s).")

    if missing_predictions:
        print(
            f"Warning: {len(missing_predictions)} prediction(s) were missing:"
        )
        for filename in missing_predictions:
            print(f"  - {filename}")

    print(f"Volume comparison written to: {output_csv}")


if __name__ == "__main__":
    main()
