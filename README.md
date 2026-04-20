# Credit Card Fraud Detection

A research-oriented machine learning analysis of credit card transactions to detect fraudulent activity.

## Overview

This project builds and compares six machine learning models on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset contains 284,807 transactions made by European cardholders, of which only 492 are fraud (0.17%). The core challenge is finding those rare fraud cases without generating excessive false alarms on legitimate transactions.

## Notebook

The analysis is contained in a single end-to-end notebook:

**`Definitive_Fraud_Analysis_Merge.ipynb`**

### What the Notebook Covers

| Section | Description |
|---|---|
| 1. Setup and Data Ingestion | Load the dataset, remove duplicates, audit for missing values |
| 2. Visual Data Analysis | Histograms, distribution charts, correlation heatmaps, scatter plots, 3D interactive plot |
| 3. Redundant Column Check | Variance Inflation Factor (VIF) analysis to confirm data quality |
| 4. Data Preprocessing | RobustScaler for Amount, StandardScaler for Time, IQR-based outlier removal, interaction feature creation |
| 5. Balancing the Data | Comparison of SMOTE (synthetic oversampling) vs. Random Under-sampling |
| 6. Model Training | Six models trained and compared |
| 7. Evaluation | Full metrics table (Accuracy, Precision, Recall, F1-Score, ROC-AUC, AUPRC), Precision-Recall curves, Confusion Matrices, per-model classification reports |
| 8. Advanced Analysis | Probability calibration curves, neural network learning curves, optimal threshold analysis |

### Models Compared

- Logistic Regression
- Random Forest
- XGBoost (with hyperparameter tuning via RandomizedSearchCV)
- LightGBM
- Neural Network (Keras/TensorFlow, with MLPClassifier fallback for Python 3.14+)
- Stacking Ensemble (RF + XGBoost + LightGBM with Logistic Regression as meta-learner)

### Evaluation Metrics

Since fraud is extremely rare, standard accuracy is not a reliable metric. This project uses:

- **Precision** — Of all transactions flagged as fraud, how many actually were?
- **Recall** — Of all actual fraud cases, how many did the model catch?
- **F1-Score** — Harmonic mean of Precision and Recall.
- **ROC-AUC** — Area under the Receiver Operating Characteristic curve.
- **AUPRC** — Area Under the Precision-Recall Curve. The primary metric for imbalanced datasets.

## Dataset

The dataset is not included in this repository due to its size.

Download it from Kaggle:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file at:
```
data/creditcard.csv
```

The notebook will automatically look for the file at that path when it runs.

## Requirements

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels imbalanced-learn xgboost lightgbm plotly
```

**Note on TensorFlow:** TensorFlow does not currently support Python 3.14. The notebook detects this automatically and falls back to Scikit-Learn's `MLPClassifier` for the Neural Network section, so it runs without errors on any Python version.

## How to Run

1. Clone this repository
2. Download and place `creditcard.csv` in the `data/` folder
3. Install requirements
4. Open `Definitive_Fraud_Analysis_Merge.ipynb` in Jupyter or VS Code
5. Run all cells from top to bottom

## Key Findings

- Combining multiple models into a Stacking Ensemble consistently outperforms any single model.
- AUPRC is the most reliable measure of performance for highly imbalanced fraud data.
- Adjusting the fraud decision threshold beyond the default 0.5 can significantly increase fraud recall without a proportional rise in false positives.
