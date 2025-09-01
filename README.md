# Detecting Credit Card Fraud 
This project builds and evaluates machine learning models for detecting fraudulent credit card transactions.  We explore the dataset, handle class imbalance, train multiple models, and choose the best based on precision-recall balance.

Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions. Includes EDA, class imbalance handling (SMOTE), model comparison (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM), hyperparameter tuning, evaluation with ROC/PR curves, and SHAP interpretability.

📖 Medium article walkthrough: (https://medium.com/@nana.achiaaoforikuragu/detecting-credit-card-fraud-with-machine-learning-balancing-recall-precision-8d37cd5fc933)

📂 Repository Structure
credit-card-fraud-detection/
├── data/                  # dataset (not included, see instructions)
├── notebooks/             # Jupyter notebooks (EDA, modeling, tuning)
├── results/               # exported plots, metrics, shap explanations
├── src/                   # source code for preprocessing, training, API
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── app_fastapi.py     # FastAPI inference server
├── requirements.txt       # dependencies
├── .gitignore
├── LICENSE                # MIT
└── README.md

📊 Dataset

Source: Kaggle Credit Card Fraud Detection Dataset

Transactions from European cardholders (Sept 2013)

284,807 transactions, of which only 492 are fraud (~0.17%)

Features: anonymized PCA components V1…V28, plus Amount and Time

Target: Class (0 = Non-Fraud, 1 = Fraud)

⚠️ Download the dataset from Kaggle and place creditcard.csv in the data/ folder. This file is git-ignored.

⚙️ Installation
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

🚀 Usage
1. Run EDA & Modeling (in notebooks)

notebooks/01_EDA.ipynb – class imbalance, distributions, correlations

notebooks/02_Modeling.ipynb – baseline models comparison

notebooks/03_Tuning_and_PR_Curves.ipynb – hyperparameter tuning + PR/ROC curves

2. Train & Evaluate Models (CLI)
python src/train_models.py --data data/creditcard.csv --out results
python src/evaluate_models.py --preds results/metrics.json --out results

3. Serve Model via FastAPI
uvicorn src.app_fastapi:app --host 0.0.0.0 --port 8080


Send a POST request with transaction features:

curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, -1.3, 2.4, ...]}'

📈 Results
Model	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	0.062	0.918	0.116	0.980
Random Forest	0.973	0.735	0.837	0.948
Gradient Boosting	0.738	0.602	0.663	0.786
XGBoost (chosen)	0.929	0.796	0.857	0.945
LightGBM	0.375	0.612	0.465	0.731

Chosen Model: Tuned XGBoost – best trade-off between recall and precision.
