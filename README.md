# Detecting Credit Card Fraud 
Machine learning project for detecting fraudulent credit card transactions. Includes EDA, class imbalance handling (SMOTE), model comparison (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM), hyperparameter tuning, evaluation with ROC/PR curves, and SHAP interpretability.

# Medium article walkthrough 
https://medium.com/@nana.achiaaoforikuragu/detecting-credit-card-fraud-with-machine-learning-balancing-recall-precision-8d37cd5fc933

# Dataset
Source: Kaggle Credit Card Fraud Detection Dataset
Transactions from European cardholders (Sept 2013)
284,807 transactions, of which only 492 are fraud (~0.17%)
Features: anonymized PCA components V1…V28, plus Amount and Time
Target: Class (0 = Non-Fraud, 1 = Fraud)

## 📈 Results

| Model                | Precision | Recall | F1 Score | ROC-AUC |
|-----------------------|-----------|--------|----------|---------|
| Logistic Regression   | 0.062     | **0.918** | 0.116    | **0.980** |
| Random Forest         | **0.973** | 0.735  | 0.837    | 0.948   |
| Gradient Boosting     | 0.738     | 0.602  | 0.663    | 0.786   |
| **XGBoost (Chosen)**  | 0.929     | 0.796  | **0.857** | 0.945   |
| LightGBM              | 0.375     | 0.612  | 0.465    | 0.731   |


# Chosen Model: Tuned XGBoost – best trade-off between recall and precision.
