#!/usr/bin/env python3
"""
Rigidly register paired post-contrast MRI to pre-contrast MRI.

The pre-contrast image is used as the fixed/reference image and the
post-contrast image as the moving image. Registration uses a 3D Euler
rigid-body transform (three rotations and three translations), Mattes
mutual information, and a multi-resolution optimisation strategy.

Following registration, the post-contrast MRI is resampled onto the
pre-contrast voxel grid using linear interpolation. The pre-contrast
image is not transformed.

The script expects paired NIfTI files in separate Pre/ and Post/
directories and writes nnU-Net-compatible paired channels, the estimated
rigid transforms, a processing/QC CSV, and optional checkerboard images.
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


PRE_SUFFIX = "_T1_PRE.nii.gz"
POST_SUFFIX = "_T1_POST.nii.gz"

N_HISTOGRAM_BINS = 50
METRIC_SAMPLING_FRACTION = 0.20
RANDOM_SEED = 42
SHRINK_FACTORS = [4, 2, 1]
SMOOTHING_SIGMAS_MM = [2, 1, 0]
LEARNING_RATE = 2.0
MIN_STEP = 1e-4
N_ITERATIONS = 300
GRADIENT_MAGNITUDE_TOLERANCE = 1e-8


def case_id_from_filename(filename: str, suffix: str) -> str:
    """Return the case identifier after validating the expected suffix."""
    if not filename.endswith(suffix):
        raise ValueError(
            f"Filename does not end with expected suffix '{suffix}': {filename}"
        )
    return filename[:-len(suffix)]


def register_post_to_pre(
    fixed_pre: sitk.Image,
    moving_post: sitk.Image,
):
    """Rigidly register a post-contrast MRI to its pre-contrast MRI."""
    fixed_float = sitk.Cast(fixed_pre, sitk.sitkFloat32)
    moving_float = sitk.Cast(moving_post, sitk.sitkFloat32)

    rigid_transform = sitk.CenteredTransformInitializer(
        fixed_float,
        moving_float,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=N_HISTOGRAM_BINS
    )
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(
        METRIC_SAMPLING_FRACTION,
        seed=RANDOM_SEED,
    )
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=LEARNING_RATE,
        minStep=MIN_STEP,
        numberOfIterations=N_ITERATIONS,
        gradientMagnitudeTolerance=GRADIENT_MAGNITUDE_TOLERANCE,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=SHRINK_FACTORS)
    registration.SetSmoothingSigmasPerLevel(
        smoothingSigmas=SMOOTHING_SIGMAS_MM
    )
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(rigid_transform, inPlace=True)

    registration.Execute(fixed_float, moving_float)

    registered_post = sitk.Resample(
        moving_post,
        fixed_pre,
        rigid_transform,
        sitk.sitkLinear,
        0.0,
        moving_post.GetPixelID(),
    )

    registration_info = {
        "Final_Metric_Value": float(registration.GetMetricValue()),
        "Optimizer_Iterations": int(registration.GetOptimizerIteration()),
        "Optimizer_Stop_Condition":
            registration.GetOptimizerStopConditionDescription(),
    }

    return registered_post, rigid_transform, registration_info


def identity_resample_to_pre(
    moving_post: sitk.Image,
    fixed_pre: sitk.Image,
) -> sitk.Image:
    """Resample POST to the PRE grid without estimating registration."""
    return sitk.Resample(
        moving_post,
        fixed_pre,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        0.0,
        moving_post.GetPixelID(),
    )


def geometry_matches(
    reference: sitk.Image,
    candidate: sitk.Image,
    tolerance: float = 1e-6,
) -> dict:
    """Check whether two images occupy the same voxel grid."""
    size_match = reference.GetSize() == candidate.GetSize()
    spacing_match = all(
        abs(a - b) <= tolerance
        for a, b in zip(reference.GetSpacing(), candidate.GetSpacing())
    )
    origin_match = all(
        abs(a - b) <= tolerance
        for a, b in zip(reference.GetOrigin(), candidate.GetOrigin())
    )
    direction_match = all(
        abs(a - b) <= tolerance
        for a, b in zip(reference.GetDirection(), candidate.GetDirection())
    )

    return {
        "Size_Match_After": size_match,
        "Spacing_Match_After": spacing_match,
        "Origin_Match_After": origin_match,
        "Direction_Match_After": direction_match,
        "Geometry_Match_After": (
            size_match
            and spacing_match
            and origin_match
            and direction_match
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rigidly register paired post-contrast MRI to pre-contrast MRI "
            "and prepare nnU-Net-compatible paired channels."
        )
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Dataset directory containing Pre/ and Post/ subdirectories.",
    )
    parser.add_argument(
        "--pre_dir_name",
        default="Pre",
        help="Name of the pre-contrast directory (default: Pre).",
    )
    parser.add_argument(
        "--post_dir_name",
        default="Post",
        help="Name of the post-contrast directory (default: Post).",
    )
    parser.add_argument(
        "--output_dir_name",
        default="Registered_nnUNet",
        help="Output directory for nnU-Net-compatible images.",
    )
    parser.add_argument(
        "--save_checkerboards",
        action="store_true",
        help="Save before/after registration checkerboard volumes.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    pre_dir = base_dir / args.pre_dir_name
    post_dir = base_dir / args.post_dir_name
    output_dir = base_dir / args.output_dir_name
    transform_dir = base_dir / "Registration_Transforms"
    qc_dir = base_dir / "Registration_QC"
    pairing_csv = base_dir / "Pairing_Audit.csv"
    mapping_csv = base_dir / "Registration_Mapping.csv"

    if not pre_dir.is_dir():
        raise FileNotFoundError(f"PRE directory does not exist: {pre_dir}")
    if not post_dir.is_dir():
        raise FileNotFoundError(f"POST directory does not exist: {post_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    transform_dir.mkdir(parents=True, exist_ok=True)
    if args.save_checkerboards:
        qc_dir.mkdir(parents=True, exist_ok=True)

    pre_files = sorted(pre_dir.glob(f"*{PRE_SUFFIX}"))
    post_files = sorted(post_dir.glob(f"*{POST_SUFFIX}"))

    pre_map = {
        case_id_from_filename(path.name, PRE_SUFFIX): path
        for path in pre_files
    }
    post_map = {
        case_id_from_filename(path.name, POST_SUFFIX): path
        for path in post_files
    }

    if len(pre_map) != len(pre_files):
        raise RuntimeError("Duplicate PRE case identifiers were detected.")
    if len(post_map) != len(post_files):
        raise RuntimeError("Duplicate POST case identifiers were detected.")

    all_case_ids = sorted(set(pre_map) | set(post_map))
    if not all_case_ids:
        raise RuntimeError("No correctly named PRE or POST images were found.")

    pairing_rows = []
    for case_id in all_case_ids:
        pre_path = pre_map.get(case_id)
        post_path = post_map.get(case_id)
        pairing_rows.append(
            {
                "Case_ID": case_id,
                "PRE_present": pre_path is not None,
                "POST_present": post_path is not None,
                "PRE_file": pre_path.name if pre_path else "",
                "POST_file": post_path.name if post_path else "",
                "Pair_complete": (
                    pre_path is not None and post_path is not None
                ),
            }
        )

    pairing_df = pd.DataFrame(pairing_rows)
    pairing_df.to_csv(pairing_csv, index=False)

    incomplete = pairing_df.loc[~pairing_df["Pair_complete"]]
    if not incomplete.empty:
        print(incomplete.to_string(index=False))
        raise RuntimeError(
            "Incomplete PRE/POST pairs detected. "
            "Review Pairing_Audit.csv before continuing."
        )

    mapping_rows = []

    for case_id in pairing_df["Case_ID"].tolist():
        pre_file = pre_map[case_id]
        post_file = post_map[case_id]
        print(f"Processing: {case_id}")

        try:
            pre_img = sitk.ReadImage(str(pre_file))
            post_img = sitk.ReadImage(str(post_file))

            if pre_img.GetDimension() != 3 or post_img.GetDimension() != 3:
                raise ValueError(
                    "Both PRE and POST images must be three-dimensional."
                )

            registered_post, rigid_transform, reg_info = (
                register_post_to_pre(
                    fixed_pre=pre_img,
                    moving_post=post_img,
                )
            )

            geometry_info = geometry_matches(pre_img, registered_post)
            if not geometry_info["Geometry_Match_After"]:
                raise RuntimeError(
                    "Registered POST does not match PRE output geometry."
                )

            pre_output = output_dir / f"{case_id}_0000.nii.gz"
            post_output = output_dir / f"{case_id}_0001.nii.gz"
            transform_output = (
                transform_dir / f"{case_id}_POST_to_PRE_rigid.tfm"
            )

            # PRE remains the reference image and is not transformed.
            sitk.WriteImage(pre_img, str(pre_output))
            sitk.WriteImage(registered_post, str(post_output))
            sitk.WriteTransform(rigid_transform, str(transform_output))

            if args.save_checkerboards:
                post_before = identity_resample_to_pre(
                    moving_post=post_img,
                    fixed_pre=pre_img,
                )
                pre_qc = sitk.Cast(pre_img, sitk.sitkFloat32)
                before_qc = sitk.Cast(post_before, sitk.sitkFloat32)
                after_qc = sitk.Cast(registered_post, sitk.sitkFloat32)

                checker_before = sitk.CheckerBoard(
                    pre_qc, before_qc, [4, 4, 4]
                )
                checker_after = sitk.CheckerBoard(
                    pre_qc, after_qc, [4, 4, 4]
                )
                sitk.WriteImage(
                    checker_before,
                    str(qc_dir / f"{case_id}_checker_before.nii.gz"),
                )
                sitk.WriteImage(
                    checker_after,
                    str(qc_dir / f"{case_id}_checker_after.nii.gz"),
                )

            parameters = rigid_transform.GetParameters()
            rotations_rad = [float(value) for value in parameters[:3]]
            translations_mm = [float(value) for value in parameters[3:6]]
            rotations_deg = [
                math.degrees(value) for value in rotations_rad
            ]
            translation_magnitude_mm = math.sqrt(
                sum(value ** 2 for value in translations_mm)
            )

            mapping_rows.append(
                {
                    "Case_ID": case_id,
                    "Original_PRE_File": pre_file.name,
                    "Original_POST_File": post_file.name,
                    "Output_PRE_0000": pre_output.name,
                    "Output_POST_0001": post_output.name,
                    "Rigid_Transform_File": transform_output.name,
                    "Registration_Transform": "Euler3D rigid",
                    "Registration_Metric": "Mattes mutual information",
                    "MRI_Interpolation": "Linear",
                    "Rotation_X_deg": rotations_deg[0],
                    "Rotation_Y_deg": rotations_deg[1],
                    "Rotation_Z_deg": rotations_deg[2],
                    "Translation_X_mm": translations_mm[0],
                    "Translation_Y_mm": translations_mm[1],
                    "Translation_Z_mm": translations_mm[2],
                    "Translation_Magnitude_mm":
                        translation_magnitude_mm,
                    **reg_info,
                    **geometry_info,
                    "Processing_Status": "SUCCESS",
                    "Error_Message": "",
                }
            )

        except Exception as exc:
            print(f"[ERROR] {case_id}: {exc}")
            mapping_rows.append(
                {
                    "Case_ID": case_id,
                    "Original_PRE_File": pre_file.name,
                    "Original_POST_File": post_file.name,
                    "Processing_Status": "FAILED",
                    "Error_Message": str(exc),
                }
            )

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(mapping_csv, index=False)

    n_failed = int(
        (mapping_df["Processing_Status"] == "FAILED").sum()
    )
    n_success = int(
        (mapping_df["Processing_Status"] == "SUCCESS").sum()
    )

    print(f"Successful registrations: {n_success}")
    print(f"Failed registrations: {n_failed}")
    print(f"Registration log: {mapping_csv}")

    if n_failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
