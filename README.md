# MultiPathDx

This repository contains a minimal implementation of **MultiPathDx**, a robot-integrated
decision framework for multi-pathology risk screening. The code is designed to
work with the [Extended Distress Analysis Interview Corpus (E‑DAIC)] dataset.
It uses pre-computed audio and visual features to train a simple logistic
regression model for depression screening.

## Requirements

The implementation relies only on the Python standard library and does not
require external packages.

## Dataset Preparation

Download the E‑DAIC dataset and place the participant directories and the
`labels` folder in a local directory.  The directory structure should look
as follows:

```
E-DAIC/
  labels/
    train_split.csv
    dev_split.csv
    test_split.csv
  300_P/
    features/
      300_OpenSMILE2.3.0_egemaps.csv
      300_OpenFace2.1.0_Pose_gaze_AUs.csv
    ...
```

## Training

Run the training script by providing the dataset root and a split file
containing the labels:

```
python train.py /path/to/E-DAIC /path/to/E-DAIC/labels/train_split.csv --epochs 20
```

The script prints the training accuracy after fitting a logistic regression model.

This implementation is intentionally lightweight to serve as a starting point for
extending **MultiPathDx** with additional modules and more advanced models.

