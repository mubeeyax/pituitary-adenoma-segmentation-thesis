# Segmentation Performance Evaluation

This directory contains the script used to calculate segmentation performance metrics for the automated segmentation experiments in this project.

## Metrics

The following metrics are calculated for each anatomical label:

- Dice similarity coefficient;
- Jaccard index;
- volumetric similarity (VS);
- Hausdorff distance (HD);
- 95th-percentile Hausdorff distance (HD95).

Overlap and volumetric similarity metrics are dimensionless. Hausdorff distance and HD95 are calculated using the physical voxel spacing supplied from the NIfTI images and are therefore reported in physical distance units.

## Requirements

- Python 3
- NumPy
- pandas
- SimpleITK
- seg-metrics
- XlsxWriter
- SciPy (optional, for Student's t critical values)

Install the required packages using:

```bash
pip install numpy pandas SimpleITK seg-metrics xlsxwriter scipy
```

## Input structure

The script expects separate directories containing manually delineated reference labels and corresponding model predictions.

```text
Evaluation/
├── Reference/
│   ├── Case001.nii.gz
│   ├── Case002.nii.gz
│   └── ...
└── Predictions/
    ├── Case001.nii.gz
    ├── Case002.nii.gz
    └── ...
```

Reference and prediction files are matched by filename.

## Usage

Run the evaluation using:

```bash
python compute_segmentation_metrics.py \
    --reference_dir /path/to/Reference \
    --prediction_dir /path/to/Predictions \
    --output_dir /path/to/Results
```

By default, labels `1` to `8` are evaluated.

Alternative labels can be specified explicitly:

```bash
python compute_segmentation_metrics.py \
    --reference_dir /path/to/Reference \
    --prediction_dir /path/to/Predictions \
    --output_dir /path/to/Results \
    --labels 1,2,3,4,5,6,7,8
```

## Outputs

The script produces:

```text
Results/
├── segmentation_metrics.csv
└── segmentation_metrics.xlsx
```

The CSV contains the case-level and label-level metric values produced by `seg-metrics`.

The Excel workbook contains a `Summary_by_label` worksheet together with a separate worksheet for each requested anatomical label.

Summary statistics include:

- number of observations;
- mean;
- sample standard deviation;
- standard error;
- 95% confidence interval;
- coefficient of variation.

Where SciPy is available, 95% confidence intervals are calculated using the Student's t distribution. If SciPy is unavailable, a normal approximation using `z = 1.96` is used.

## Physical voxel spacing

SimpleITK returns image spacing in `(x, y, z)` order, whereas arrays generated using `GetArrayFromImage` are indexed in `(z, y, x)` order. The script therefore reverses the image spacing before supplying it to `seg-metrics`.

This ensures that distance-based metrics are calculated using the corresponding physical voxel dimensions.

## Missing labels and predictions

A requested anatomical label is evaluated when it is present in either the reference delineation or prediction for that examination.

Labels absent from both are not evaluated for that examination.

If a corresponding prediction file is unavailable, the case is reported and excluded from metric calculation.

## Thesis methodology

Segmentation performance was evaluated separately for each anatomical structure using overlap-based, boundary-based and volumetric measures. Dice similarity coefficient and Jaccard index were used to assess spatial overlap, Hausdorff distance and HD95 to assess boundary agreement, and volumetric similarity to assess agreement in segmented volume.

Additional volume-difference analyses are implemented separately.
