# Customer Churn Prediction & Risk Analysis

> **Deep Learning approach to predicting customer churn on imbalanced data — combining Artificial Neural Networks, advanced resampling strategies, and Bayesian hyperparameter optimization.**

---

## 📌 Project Overview

This project tackles the critical business problem of customer churn using a deep learning pipeline built end-to-end with TensorFlow/Keras. The core challenge — a heavily **imbalanced dataset** where churned customers are the rare minority class — is addressed through a rigorous comparison of five resampling and weighting techniques, ultimately achieving a significant improvement in churn detection performance.

---

## 🎯 Key Results

| Metric | Value |
|---|---|
| Best Minority-Class F1-Score | **0.78** |
| Churn Detection Improvement | **+35%** over baseline |
| Best Resampling Method | SMOTE |
| Hyperparameter Tuner | Optuna (Bayesian) |

---

## 🔧 Methodology

### 1. Exploratory Data Analysis (EDA)
- Statistical profiling and distribution analysis across all features
- Correlation heatmaps and class-imbalance quantification
- Identification of key churn-driving signals

### 2. Feature Engineering
- Encoding categorical variables and scaling continuous features
- Derived interaction features to surface latent churn patterns
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

> **SMOTE delivered the best results**, achieving an F1-score of **0.78** and improving churn detection by **35%** compared to the unbalanced baseline.

### 4. Model Architecture
- **Framework:** TensorFlow / Keras
- **Model:** Fully-connected Artificial Neural Network (ANN)
- **Regularization:** Dropout layers to prevent overfitting + Early Stopping on validation loss

### 5. Hyperparameter Optimization
- **Tool:** [Optuna](https://optuna.org/) — a Bayesian optimization framework
- Tuned learning rate, layer sizes, dropout rates, and batch size across multiple trials
- Early stopping callback used during each trial to ensure efficient training

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow, Keras |
| Data Manipulation | Pandas, NumPy |
| Imbalance Handling | imbalanced-learn (imblearn) |
| Hyperparameter Tuning | Optuna |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |

---

## 📈 Results & Insights

- **SMOTE** significantly outperformed all other resampling strategies by generating meaningful synthetic minority samples that helped the ANN learn robust churn patterns without introducing noise.
- **Optuna-guided tuning** converged on an optimal architecture faster than manual grid search, reducing experimentation overhead.
- **Dropout + Early Stopping** proved essential — without regularization, the model overfitted heavily given the dataset size.
- The final model provides a strong foundation for a **real-time churn risk scoring** system deployable via REST API.

---

## 🚀 Future Work

- Integrate **XGBoost / LightGBM** as ensemble baselines for comparison
- Explore **temporal features** if longitudinal customer data is available
- Deploy the model as a **FastAPI or Flask microservice** for production scoring
- Add **SHAP-based explainability** to surface per-customer churn risk drivers

---

