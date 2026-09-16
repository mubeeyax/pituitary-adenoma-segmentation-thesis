# nnU-Net Training and Inference

This directory contains the custom nnU-Net v2 trainer used for the segmentation experiments in this project.

## Custom trainer

`nnUNetTrainerNoMirror200ep.py` inherits from the standard nnU-Net v2 `nnUNetTrainer` and introduces two modifications:

1. training is limited to 200 epochs;
2. mirroring augmentation is disabled.

The remaining training configuration is inherited from nnU-Net v2.

The trainer disables mirroring by setting the mirroring axes returned by `configure_rotation_dummyDA_mirroring_and_inital_patch_size()` to `None`. The allowed mirroring axes for inference are also set to `None`.

## Training duration

Models were trained for 200 epochs.

This duration was selected following preliminary experiments comparing 200 and 1000 epochs. The shorter training duration produced similar segmentation performance while substantially reducing computational requirements.

## Training

The custom trainer is specified using the nnU-Net `-tr` argument.

An individual cross-validation fold can be trained using:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerNoMirror200ep
```

For example:

```bash
nnUNetv2_train 016 3d_fullres 3 -tr nnUNetTrainerNoMirror200ep
```

Models in this project used the nnU-Net v2 3D full-resolution configuration (`3d_fullres`).

Five-fold cross-validation was used, with folds trained separately using the corresponding fold number.

## Inference

Predictions were generated using `nnUNetv2_predict` with the same custom trainer specified during model development.

Mirroring-based test-time augmentation was explicitly disabled during inference using `--disable_tta`.

A general prediction command is:

```bash
nnUNetv2_predict \
    -i /path/to/imagesTs \
    -o /path/to/predictions \
    -d DATASET_ID \
    -c 3d_fullres \
    -tr nnUNetTrainerNoMirror200ep \
    --save_probabilities \
    --disable_tta
```

Dataset-specific server paths are intentionally omitted from this repository.

## Input channels

The number of input channels depended on the experiment.

For normal pituitary segmentation and single-contrast pituitary adenoma segmentation, the model received:

```text
_0000 = T1-weighted MRI
_0001 = left-right spatial coordinate channel
```

For paired pre- and post-contrast pituitary adenoma segmentation, the model received:

```text
_0000 = pre-contrast T1-weighted MRI
_0001 = spatially matched post-contrast T1-weighted MRI
_0002 = left-right spatial coordinate channel
```

The left-right coordinate channel is generated using the script provided in the `coordinate_channel` directory.

For paired pre- and post-contrast models, image preparation and registration procedures are documented in the `registration` directory.

## Hardware

Model training was performed using NVIDIA RTX A6000 GPUs with 48 GB GPU memory.

Local workstation names and server-specific configuration are not included because they are not required to reproduce the model configuration.

## Thesis methodology

All experiments used the nnU-Net v2 3D full-resolution configuration. Models were trained for 200 epochs using five-fold cross-validation. Mirroring augmentation was disabled in the custom trainer, and mirroring-based test-time augmentation was disabled during inference.

Other preprocessing, architecture and training parameters followed the corresponding nnU-Net v2 configuration unless otherwise specified.
