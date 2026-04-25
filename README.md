# Customer Behavioral Analysis for Retail E-commerce

##  Project Overview
This project focuses on predicting customer **churn** (attrition) for a gift e-commerce business. By leveraging machine learning, the goal is to identify high-risk customers and enable personalized marketing strategies to improve retention.

The analysis is based on a dataset of **4,372 clients** with an initial churn rate of **33.3%**.

##  Data Science Pipeline
The project implements a robust, leak-free pipeline designed for reproducibility:

1.  **Exploration (EDA):** Structural analysis and anomaly detection.
2.  **Feature Engineering:** * Extracted temporal data from registration dates.
    * Parsed IP addresses for location-based insights.
    * Created 10 new business-driven features (e.g., `TenureRecencyRatio`).
3.  **Encoding:** Applied Ordinal, One-Hot, and Target Encoding to categorical variables.
4.  **Feature Selection:** Reduced feature space from 86 to 58 relevant features using:
    * Variance Thresholds.
    * Missing Value Thresholds (>50%).
    * Pearson Correlation ($|r| > 0.85$).
    * Random Forest Feature Importance.
5.  **Advanced Imputation:**
    * **Age:** KNN Imputer ($k=5$).
    * **Satisfaction Score:** Iterative Imputer (MICE).
    * **Support/Purchases:** Median imputation.
6.  **Normalization:** Continuous features scaled using `StandardScaler`.
7.  **PCA (Dimensionality Reduction):** * 2D for visualization.
    * 10 components for fast modeling.
    * 25 components for 90% variance retention.
8.  **Class Balancing:** Applied **SMOTE** to the training set to address class imbalance.
9.  **Export:** Saved processed datasets and artifacts (`joblib`) for model deployment.

##  Key Predictors
The most significant indicators of churn identified are:
* **ChurnRiskCategory** ($|r|=0.879$)
* **Recency** ($|r|=0.859$)
* **TenureRecencyRatio** ($|r|=0.607$)

##  Processed Datasets
The following datasets are prepared for the modeling phase:

| Dataset | Dimensions | Use Case |
| :--- | :--- | :--- |
| **X_selected** | 58 Features | Interpretable models (XGBoost, Random Forest) |
| **X_pca10** | 10 Components | Fast modeling and visualization |
| **X_pca25** | 25 Components | Performance-oriented (90% variance) |

## 🛠 Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`

---
**Author:** Ala Belguith  
**Academic Year:** 2025-2026  
**Project:** Machine Learning Workshop — GI