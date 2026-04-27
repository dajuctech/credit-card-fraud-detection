# Credit Card Fraud Detection


## Project Overview

This project builds and compares six machine learning models to detect fraudulent credit card transactions. The core challenge is class imbalance — fraudulent transactions represent less than 0.2% of all records — so three data balancing techniques (oversampling, undersampling, and SMOTE) are applied and compared alongside raw and scaled versions of the dataset.

Read the full write-up on Medium: [How I Built a Machine Learning System to Catch Credit Card Fraud](https://medium.com/@danieljude1992)

---

## Dataset

- **Source:** Kaggle — Credit Card Fraud Detection dataset
- **Transactions:** 284,807 records (European cardholders, September 2013)
- **Features:** 30 input columns (V1–V28 from PCA, Time, Amount) + 1 target column (Class)
- **Class distribution:** Class 0 = legitimate, Class 1 = fraudulent (~0.17% fraud)
- **Duplicates found and removed:** 1,081

The dataset is downloaded automatically inside the notebook via `gdown`. You do not need to download it manually.

---

## Project Structure

```
credit-card-fraud-detection/
├── Credit Card Fraud Detection.ipynb   # Main notebook — all code
├── README.md                           # This file
└── creditcard.csv                      # Dataset (downloaded automatically)
```

---

## Pipeline

The project follows a structured machine learning workflow:

```
1. Define the Problem
2. Collect Data (Kaggle via gdown)
3. Data Preprocessing (DataPreprocessing class)
   ├── Check missing values
   ├── Handle missing values (mean/mode fill)
   ├── Check data types
   ├── Remove duplicates (1,081 removed)
   ├── Summary statistics
   ├── Class distribution check
   ├── Skewness check
   ├── Visualise distributions
   ├── Correlation matrix
   └── Visualise outliers
4. Exploratory Data Analysis (EDA class)
   ├── Class distribution visualisation
   ├── Transaction amount by class
   ├── Time distribution by class
   ├── Correlation heatmap
   ├── Outlier box plots
   ├── Feature distributions
   ├── Time vs fraud plot
   ├── Amount vs fraud
   └── Pairplots
5. Feature Selection & Engineering
   ├── Drop Time column
   ├── Define X and y (normal dataset)
   ├── Normalise Amount (StandardScaler → scaled dataset)
   └── Handle class imbalance:
       ├── Oversampling (resample minority → 284,315 samples)
       ├── Undersampling (RandomUnderSampler)
       └── SMOTE (synthetic minority samples)
6. Model Development (5 datasets × 6 models)
   ├── Logistic Regression
   ├── Random Forest
   ├── K-Nearest Neighbours (K=3)
   ├── Naive Bayes (Gaussian)
   ├── XGBoost
   └── SVM (RBF kernel)
7. Model Comparison (heatmap across all combinations)
8. Conclusion
```

---

## Models Trained

Each model was implemented as a Python class with a `fit_and_evaluate()` method. Every model was trained and tested on five dataset variants:

| Dataset | Description |
|---|---|
| Normal | Raw imbalanced dataset |
| Scaled | Normal + Amount column standardised with StandardScaler |
| Undersampled | Majority class reduced to match minority class size |
| Oversampled | Minority class duplicated to match majority class size |
| SMOTE | Synthetic fraud samples generated via interpolation |

---

## Results Summary

Accuracy across all models and dataset types:

| Model | Normal | Scaled | Undersampled | Oversampled | SMOTE |
|---|---|---|---|---|---|
| Logistic Regression | 99.92% | 99.92% | 95.43% | 94.90% | 95.98% |
| Random Forest | 99.97% | 99.97% | 98.98% | 99.26% | 98.67% |
| K-Nearest Neighbours | 99.95% | 99.97% | 94.72% | 99.99% | 99.95% |
| Naive Bayes | 97.78% | 97.78% | 91.77% | 91.71% | 91.88% |
| XGBoost | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| SVM | 99.88% | 99.96% | 83.43% | — (timeout) | — (timeout) |

**Best overall models:** XGBoost and Random Forest
**Note:** XGBoost's perfect scores require further validation to rule out overfitting.
**SVM limitation:** Could not complete training on oversampled and SMOTE datasets due to excessive training time.

---

## Key Findings

- High accuracy on imbalanced data is misleading — a model predicting "not fraud" 100% of the time achieves 99.8% accuracy while catching zero fraud.
- Balanced datasets (SMOTE and oversampled) consistently improved fraud recall across all models.
- XGBoost and Random Forest delivered the most reliable results across all conditions.
- Naive Bayes was the weakest performer, particularly on imbalanced datasets.
- SVM is not practical for large balanced datasets due to its computational cost.

---

## How to Run

**Install dependencies:**
```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost gdown termcolor
```

**Run the notebook:**
```bash
jupyter notebook "Credit Card Fraud Detection.ipynb"
```

Or open directly in Google Colab — the dataset downloads automatically.

---

## Technologies Used

| Library | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib / seaborn | Data visualisation |
| scikit-learn | Preprocessing, models, and evaluation metrics |
| imbalanced-learn | RandomUnderSampler and SMOTE |
| xgboost | XGBoost classifier |
| gdown | Downloading dataset from Google Drive |
| termcolor | Coloured terminal output |

---

## Evaluation Metrics Used

- **Accuracy** — overall correctness
- **Precision** — of all transactions flagged as fraud, how many were actually fraud
- **Recall** — of all actual fraud transactions, how many were correctly caught
- **F1 Score** — harmonic mean of precision and recall (most important for imbalanced problems)
- **Confusion Matrix** — breakdown of true positives, false positives, true negatives, false negatives

---

## License

This project is for academic purposes. Dataset sourced from Kaggle under their terms of use.
