# FFNN Breast Cancer Classification — Experimental Report

## Objective and dataset
Predict benign versus malignant diagnosis from 20 selected clinical measurements. Source: UCI Breast Cancer Wisconsin (Diagnostic), dataset 17; 569 samples, 30 original features. This experiment selects the **first 20 features in the dataset's published feature order** (10 mean and 10 standard-error features). scikit-learn ships a copy of the WDBC dataset; no synthetic observations are used. **Malignant = 1**, Benign = 0. Stratified 60%/20%/20% train/validation/test split; StandardScaler is fit to training data only.

## Architecture and mathematical mapping
20 inputs → 10 ReLU → 10 ReLU → 10 ReLU → 10 ReLU → 10 ReLU → 1 sigmoid output. PyTorch weights are stored (out_features, in_features): (10,20), (10,10) × 4, (1,10); biases (10,) × 5 and (1,). Baseline trainable parameters = (20×10+10) + 4×(10×10+10) + (10×1+1) = **661**. BatchNorm adds 20 learnable parameters per hidden layer (scale and shift), so BN variants have **761** learnable parameters (running statistics are buffers). Sigmoid outputs P(malignant | features). Binary cross-entropy is appropriate for probabilistic binary classification; MSE is a comparative alternative.

## Training protocol
Five hidden layers, 10 units each; BN is placed between linear and ReLU, dropout p=0.2 after ReLU. Adam lr=0.001, SGD lr=0.02 momentum=0.9; batch size 32; up to 180 epochs; early stopping monitors validation loss with patience=10, improvement tolerance 0.0001, and restores best weights. Fixed random seed 42. The test split is held out until model selection by validation performance is finished.

## Optimizer/loss comparison
| Configuration | Epochs run | Best epoch | Final training loss | Final validation loss | Test accuracy |
|---|---:|---:|---:|---:|---:|
| SGD + MSE | 33 | 23 | 0.0248 | 0.0361 | 92.11% |
| SGD + BCE | 28 | 18 | 0.1007 | 0.1004 | 92.11% |
| Adam + MSE | 72 | 62 | 0.0191 | 0.0262 | 94.74% |
| Adam + BCE | 72 | 62 | 0.0961 | 0.1128 | 94.74% |

**Selection:** Adam + MSE had the strongest validation accuracy (with validation malignant recall and validation loss as tie-breakers). The different raw BCE and MSE losses should not be compared numerically as if on the same scale. Epochs run means the stopping epoch, not necessarily the best epoch.

## Held-out test evaluation — Adam + MSE
- Accuracy: **94.74%**
- Precision (malignant): **95.00%** = TP/(TP+FP)
- Recall / sensitivity (malignant): **90.48%** = TP/(TP+FN)
- Specificity (benign): **97.22%** = TN/(TN+FP)
- F1: **0.9268** = harmonic mean of precision and recall
- ROC-AUC: **0.9878**
- Confusion matrix: **TN=70, FP=2, FN=4, TP=38**

## Interpretation
Batch normalization can smooth optimization by normalizing hidden activations, though results are dataset-dependent. Dropout introduces training-time noise to reduce dependence on individual neurons, so its training loss can be higher than validation loss. Early stopping limits training after validation loss stalls and restores the best checkpoint, reducing the risk of keeping overfitted final weights. Examine `training_dashboard.png` for observed effects; regularization is not guaranteed to improve every run. Adam often converges efficiently with adaptive learning rates; SGD with momentum may require different tuning. BCE is better aligned with binary probability estimation than MSE, though neither is guaranteed to win on every finite dataset.

False negatives are malignant patients predicted benign; these are especially concerning because they may delay follow-up. Sensitivity should be prioritized alongside specificity and precision. Accuracy alone can conceal failures on the less frequent malignant class. ROC-AUC summarizes ranking quality across thresholds, while a single accuracy score uses one cutoff. However, even high ROC-AUC does **not** guarantee clinical safety: threshold-specific sensitivity, calibration, external validation, and clinical evaluation are still required. This model is educational, **not** a diagnostic device.

## Figures and reproducibility
- `training_dashboard.png`: regularization, loss, and accuracy comparisons.
- `clinical_dashboard.png`: confusion matrix and ROC curve.
- `comparison_results.csv`: measured experiment results.
- `ffnn_breast_cancer.py`: executable source code.

## References
1. UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
2. scikit-learn, `load_breast_cancer`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
3. PyTorch documentation: https://pytorch.org/docs/stable/index.html
