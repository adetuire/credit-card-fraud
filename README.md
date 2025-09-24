# Credit Card Fraud Detection — Cost-Aligned Notebook

This notebook builds a practical fraud detector on `creditcard.csv`, reports **PR-AUC** and **low-FPR recall**, calibrates probabilities, and chooses the operating threshold by **minimizing business cost** on a validation set. It’s a single, self-contained workflow you can run top-to-bottom.

## Overview
- **Dataset**: https://www.kaggle.com/code/tarekmasryo/credit-card-fraud-detection-full-ml-pipeline/input?select=creditcard.csv.
- **Models**: Logistic Regression, Random Forest, XGBoost.
- **Calibration**: `CalibratedClassifierCV` (isotonic/sigmoid).
- **Metrics**: PR-AUC (primary), ROC-AUC, recall @ FPR {0.1%, 0.5%, 1%}.
- **Decision**: threshold τ chosen to minimize `C_FN·FN + C_FP·FP` on validation.
- **Sensitivity**: compare thresholds/results across multiple (C_FN, C_FP) pairs.

## Additional Details
- **Environment**: Python 3.9+, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imbalanced-learn` (optional: `shap`).
- **Run**: place `creditcard.csv` beside the notebook and execute all cells.
- **Costs**: defaults use `C_FN=200`, `C_FP=5` (illustrative). Edit in the “Cost Assumptions” cell and re-run.

## What the Notebook Does
1. **Load & Check** the data (columns, dtypes, basic integrity; sort by `Time` if needed).
2. **EDA**: univariate (distributions), bivariate by `Class` (box/KDE), correlation heatmap.
3. **Split** into train / validation / test; keep test strictly for final reporting.
4. **Train** Logistic, RF, XGBoost; generate validation/test probabilities.
5. **Calibrate** probabilities; plot a reliability curve.
6. **Report Metrics** (PR-AUC, ROC-AUC, recall at low FPR).
7. **Pick Threshold** τ on validation to minimize cost; lock τ and **evaluate on test**.
8. **Sensitivity** across several cost pairs; optional cost-curve and confusion matrices.

## Summary
- Use **PR-AUC** for quality on imbalanced data.
- Set decisions by **business cost**, not a fixed 0.5 threshold.
- Re-run with your own (C_FN, C_FP) to align model behavior with operational impact.
