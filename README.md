# Cloud Forecasting Models

This repository contains the code and utilities used for the cloud forecasting experiments.

## File Overview

| File | Purpose |
|---|---|
| `train_unified.py` | Main training and testing entry point. |
| `utils_unified.py` | Shared loss, training, evaluation, and visualization utilities. |
| `time_series_pt_dataset_v2.py` | Loads and prepares cloud-mask time-series data. |
| `model_temporal.zip` | Archive of the temporal model components. |
| `s2tnet_variants.py` | S²T-Net with optional SFNO and DLWP-HPX modules. |
| `independent_baselines.py` | Standalone SFNO and DLWP-HPX baseline models. |
| `experiment_utils.py` | Data loading, spherical mapping, HEALPix mapping, and evaluation helpers. |


The SFNO and DLWP-HPX implementations are provided as controlled additions to the S²T-Net backbone.
