#!/usr/bin/env python3
"""
Compute multi-label segmentation metrics for paired reference and predicted
NIfTI label maps.

Metrics are calculated with seg-metrics:
    Dice similarity coefficient
    Jaccard index
    Volumetric similarity (VS)
    Hausdorff distance (HD)
    95th-percentile Hausdorff distance (HD95)

Outputs:
    - a CSV containing case-level/label-level metrics
    - an Excel workbook containing:
        * Summary_by_label
        * one sheet per requested label

The script preserves the metric computation used in the original analysis
while replacing dataset-specific paths with command-line arguments.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import seg_metrics.seg_metrics as sg


METRICS = ["dice", "jaccard", "vs", "hd", "hd95"]


def parse_labels(value: str):
    """Parse comma-separated integer labels, e.g. '1,2,3,4,5,6,7,8'."""
    try:
        labels = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Labels must be comma-separated integers."
        ) from exc

    if not labels:
        raise argparse.ArgumentTypeError("At least one label is required.")

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Compute segmentation metrics from paired NIfTI label maps."
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
        "--output_dir",
        required=True,
        help="Directory in which results will be written.",
    )
    parser.add_argument(
        "--labels",
        type=parse_labels,
        default=parse_labels("1,2,3,4,5,6,7,8"),
        help="Comma-separated label values (default: 1,2,3,4,5,6,7,8).",
    )
    parser.add_argument(
        "--output_stem",
        default="segmentation_metrics",
        help="Base filename for output CSV/XLSX files.",
    )
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir).expanduser().resolve()
    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    labels = args.labels

    if not reference_dir.is_dir():
        raise FileNotFoundError(
            f"Reference directory does not exist: {reference_dir}"
        )
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction directory does not exist: {prediction_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{args.output_stem}.csv"
    excel_path = output_dir / f"{args.output_stem}.xlsx"

    # seg-metrics appends to the CSV; remove an existing file to avoid
    # accidental duplication across runs.
    if csv_path.exists():
        csv_path.unlink()

    reference_files = sorted(reference_dir.glob("*.nii.gz"))
    if not reference_files:
        print(
            f"No .nii.gz files found in {reference_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    missing_predictions = []
    processed = 0

    for reference_path in reference_files:
        prediction_path = prediction_dir / reference_path.name

        if not prediction_path.exists():
            missing_predictions.append(reference_path.name)
            continue

        reference_img = sitk.ReadImage(str(reference_path))
        prediction_img = sitk.ReadImage(str(prediction_path))

        if reference_img.GetDimension() != 3 or prediction_img.GetDimension() != 3:
            raise ValueError(
                f"{reference_path.name}: both label maps must be 3D."
            )

        reference_np = sitk.GetArrayFromImage(reference_img)
        prediction_np = sitk.GetArrayFromImage(prediction_img)

        if reference_np.shape != prediction_np.shape:
            raise ValueError(
                f"{reference_path.name}: reference and prediction arrays "
                f"have different shapes ({reference_np.shape} vs "
                f"{prediction_np.shape})."
            )

        # SimpleITK spacing is returned as (x, y, z), whereas
        # GetArrayFromImage returns an array indexed as (z, y, x).
        spacing_zyx = np.asarray(
            prediction_img.GetSpacing()[::-1],
            dtype=float,
        )

        present_labels = [
            label
            for label in labels
            if np.any(reference_np == label)
            or np.any(prediction_np == label)
        ]

        if not present_labels:
            print(
                f"Skipping {reference_path.name}: "
                "none of the requested labels is present."
            )
            continue

        sg.write_metrics(
            labels=present_labels,
            gdth_img=reference_np,
            pred_img=prediction_np,
            csv_file=str(csv_path),
            spacing=spacing_zyx,
            metrics=METRICS,
        )

        processed += 1

    print(f"Processed {processed} case(s).")

    if missing_predictions:
        print(
            f"Warning: {len(missing_predictions)} prediction(s) were missing:"
        )
        for filename in missing_predictions:
            print(f"  - {filename}")

    if not csv_path.exists():
        print(
            "No metrics CSV was produced; nothing to summarise.",
            file=sys.stderr,
        )
        sys.exit(2)

    df = pd.read_csv(csv_path)

    wanted_metrics = set(METRICS)
    id_columns = [
        column
        for column in df.columns
        if column.lower() in ("case", "name", "subject", "id")
    ]
    keep_columns = (
        ["label"]
        + id_columns
        + [column for column in df.columns if column in wanted_metrics]
    )
    df = df[keep_columns].copy()

    # Descriptive summary. Confidence intervals are retained because they
    # formed part of the original analysis workflow.
    try:
        from scipy.stats import t as tdist
        use_scipy = True
        print("Using Student's t critical values for 95% confidence intervals.")
    except ImportError:
        tdist = None
        use_scipy = False
        print(
            "SciPy not available: using z=1.96 for 95% confidence intervals."
        )

    summary_rows = []

    for label, group in df.groupby("label"):
        row = {"label": label}

        for metric in METRICS:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            n = int(values.size)
            mean = float(values.mean()) if n > 0 else np.nan
            std = float(values.std(ddof=1)) if n > 1 else np.nan
            se = std / np.sqrt(n) if n > 1 else np.nan

            if n > 1:
                if use_scipy:
                    critical = float(tdist.ppf(0.975, df=n - 1))
                else:
                    critical = 1.96
            else:
                critical = np.nan

            margin = critical * se if np.isfinite(se) else np.nan
            lower = mean - margin if np.isfinite(margin) else np.nan
            upper = mean + margin if np.isfinite(margin) else np.nan

            if metric in ("dice", "jaccard", "vs"):
                if np.isfinite(lower):
                    lower = max(0.0, lower)
                if np.isfinite(upper):
                    upper = min(1.0, upper)
            elif metric in ("hd", "hd95"):
                if np.isfinite(lower):
                    lower = max(0.0, lower)

            cv = (
                std / mean
                if np.isfinite(std) and np.isfinite(mean) and mean != 0
                else np.nan
            )

            row[f"{metric}_n"] = n
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_se"] = se
            row[f"{metric}_ci95_lower"] = lower
            row[f"{metric}_ci95_upper"] = upper
            row[f"{metric}_cv"] = cv

        summary_rows.append(row)

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("label")
        .reset_index(drop=True)
    )

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(
            writer,
            sheet_name="Summary_by_label",
            index=False,
        )

        for label in labels:
            label_df = df[df["label"] == label]
            label_df.to_excel(
                writer,
                sheet_name=f"Label_{label}",
                index=False,
            )

    print(f"Case-level metrics written to: {csv_path}")
    print(f"Summary workbook written to: {excel_path}")


if __name__ == "__main__":
    main()
