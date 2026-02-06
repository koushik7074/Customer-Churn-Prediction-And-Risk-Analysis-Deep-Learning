# Customer Churn Prediction & Risk Analysis

> **Deep Learning approach to predicting customer churn on imbalanced data — combining Artificial Neural Networks and advanced resampling strategies**

---

## 📌 Project Overview

This project tackles the critical business problem of customer churn using a deep learning pipeline built end-to-end with TensorFlow/Keras. The core challenge — a heavily **imbalanced dataset** where churned customers are the rare minority class — is addressed through a rigorous comparison of five resampling and weighting techniques, ultimately achieving a significant improvement in churn detection performance.

---

## 🎯 Key Results

| Metric | Value |
|---|---|
| Best Minority-Class F1-Score | **0.80** |
| Churn Detection Improvement | **+26%** over baseline |
| Best Resampling Method | SMOTE |

---

## 🔧 Methodology

### 2. Feature Engineering
- Encoding categorical variables and scaling continuous features
- Train/test splitting with stratification to preserve class ratios

### 3. Class Imbalance Handling
Five techniques were benchmarked head-to-head on minority-class F1-score:

| # | Technique | Approach |
|---|---|---|
| 1 | **Undersampling** | Reduces majority class to match minority |
| 2 | **Oversampling** | Duplicates minority samples |
| 3 | **SMOTE** ✅ | Generates synthetic minority samples |
| 4 | **Class Weights** | Penalizes majority-class misclassification in loss |
| 5 | **Ensemble Methods** | Combines multiple models for robust prediction |

> **SMOTE delivered the best results**, achieving an F1-score of **0.80** and improving churn detection by **26%** compared to the unbalanced baseline.

### 4. Model Architecture
- **Framework:** TensorFlow / Keras
- **Model:** Fully-connected Artificial Neural Network (ANN)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow, Keras |
| Data Manipulation | Pandas, NumPy |
| Imbalance Handling | imbalanced-learn (imblearn) |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |

---

## 🚀 Future Work

- Integrate **XGBoost / LightGBM** as ensemble baselines for comparison
- Explore **temporal features** if longitudinal customer data is available
- Deploy the model as a **FastAPI or Flask microservice** for production scoring
- Add **SHAP-based explainability** to surface per-customer churn risk drivers

---

