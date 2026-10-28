# Breast Cancer Classification using a Feedforward Neural Network

**Author:** R. Akanksha 
Assignment One (Deep Learning)

A PyTorch implementation of a feedforward neural network (FFNN) for binary classification of breast tumors as benign or malignant using the Wisconsin Breast Cancer Diagnostic dataset.

> **Educational-use notice:** This project is for coursework and experimentation only. It is not a clinically validated diagnostic system and must not be used to make medical decisions.

## Model architecture

- Input: 20 selected numeric features
- Hidden layers: 5 fully connected layers, 10 neurons each
- Hidden activation: ReLU
- Output: 1 neuron with sigmoid probability
- Baseline trainable parameters: 661
- Regularization experiments: batch normalization, dropout (`p=0.2`), and early stopping (`patience=10`)

The source dataset contains 30 original features. To match the assignment requirement of 20 inputs, the script uses a documented, reproducible selection of 20 features. The target is encoded as benign = 0 and malignant = 1.

## Experiments

The training script compares:

1. SGD with momentum + MSE
2. SGD with momentum + binary cross-entropy
3. Adam + MSE
4. Adam + binary cross-entropy

It also produces training/validation loss and accuracy curves, a confusion matrix, classification metrics, and an ROC curve with AUC.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python ffnn_breast_cancer.py
```

The script uses scikit-learn's bundled breast-cancer dataset, so a separate manual download is not required. Generated charts and CSV results are saved by the script.

## Repository structure

```text
FFNN-Breast-Cancer-Classification/
├── ffnn_breast_cancer.py
├── requirements.txt
├── README.md
└── results/
    ├── analysis_report.md
    ├── architecture.png
    ├── training_dashboard.png
    ├── clinical_dashboard.png
    └── comparison_results.csv
```

## Dataset

UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic), Dataset 17:  
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Evaluation notes

Accuracy alone can conceal false negatives. For this reason, the project also reports sensitivity/recall, specificity, precision, F1-score, confusion-matrix counts, and ROC-AUC. Results depend on the fixed random seed, data split, feature selection, and hyperparameters; they should not be interpreted as universal performance guarantees.
