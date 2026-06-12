# ML Models for Phase Prediction in Alloys

This repository contains code and models from the study:

**"Accelerating Phase Prediction via CALPHAD-Informed Machine Learning and Data Augmentation"**  
*Authors: Caroline Binde Stoco et al.*

## Contents

**Pipeline Scripts**
- `Filtering.py` – Applies FAB/FAT credibility filtering to the CALPHAD dataset
- `Train Test Split.py` – Stratified 80/20 train/test split by phase score
- `Data Augmentation.py` - Lever-rule-based interpolation to generate augmented compositions
- `Descriptors.py` – Computes 20 physics-based compositional descriptors
- `ML.py` – Full ML pipeline: Optuna hyperparameter tuning, training, and evaluation

## Reference

If using this work, please cite our paper: https://doi.org/10.1016/j.commatsci.2026.114834

## 📧 Contact

For additional models/data: caroline.stoco@estudante.ufscar.br
