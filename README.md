# Classic ML Models – Breast Cancer Classification

This project demonstrates the application of classical machine learning algorithms to a binary classification problem: detecting whether a tumor is malignant or benign using the Breast Cancer Wisconsin dataset.

## Dataset

- Source: `sklearn.datasets.load_breast_cancer()`
- Features: 30 numerical features related to tumor characteristics
- Target: Binary (0 = malignant, 1 = benign)

## Models Implemented

The following scikit-learn models were trained and compared:
- Support Vector Machine (`SVC`)
- Logistic Regression
- k-Nearest Neighbors (`KNeighborsClassifier`)
- Decision Tree (`DecisionTreeClassifier`)
- Random Forest (`RandomForestClassifier`)

## Pipeline Overview

1. Load dataset and create a DataFrame
2. Stratified train-test split (80% train, 20% test)
3. Feature standardization with `StandardScaler`
4. Training and evaluation of all models
5. Metrics:
   - Accuracy
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1)
6. Final results saved to `results.csv`
7. Accuracy comparison visualized in a bar chart

## Files

- `train.py` – Loads data, trains models, evaluates them
- `models.py` – Dictionary of ML models
- `utils.py` – Evaluation and visualization functions
- `requirements.txt` – Python dependencies
- `.gitignore` – Common files and folders to ignore

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
