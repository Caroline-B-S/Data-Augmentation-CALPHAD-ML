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

**Trained Models**
- `svm_ter0_dist0_model.pkl` – SVM trained on unfiltered dataset, no augmentation
- `svm_ter0_dist30_model.pkl` – SVM trained on unfiltered dataset, augmentation distance = 30%
- `svm_ter40_dist0_model.pkl` – SVM trained on FAT ≥ 40% dataset, no augmentation
- `svm_ter40_dist30_model.pkl` – SVM trained on FAT ≥ 40% dataset, augmentation distance = 30%

**Additional results** – including models for KNN, RF, GB, and SVM across all augmentation distances – are available upon request.

## Reference

If using this work, please cite our paper

## 📧 Contact

For additional models/data: caroline.stoco@estudante.ufscar.br
