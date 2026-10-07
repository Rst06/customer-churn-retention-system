# 🛡️ AI-Based Customer Churn Prediction & Retention Strategy Recommendation System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bJipMTTrMqYLjtn1DpAEj2a7Wy1h7e5E?usp=sharing)

> **B.Tech AI & Data Science — Machine Learning Project**  
> An end-to-end classification system predicting subscriber attrition and generating targeted business retention plans.

---

## 📌 Project Overview
Customer churn in telecommunications directly threatens recurring monthly revenue. This project delivers an end-to-end Machine Learning pipeline trained on the **IBM Telco Customer Churn Dataset (7,043 customers)**. It predicts customer attrition risk, identifies key churn drivers, calculates **Revenue at Risk**, and recommends automated, rule-based retention actions.

---

## 🚀 Live Interactive Demo
Click the badge below to run the complete notebook and interactive customer prediction simulator directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bJipMTTrMqYLjtn1DpAEj2a7Wy1h7e5E?usp=sharing)

---

## 📊 Key Results & Model Benchmarking

We benchmarked three machine learning algorithms using an 80/20 Stratified Split with 5-Fold Cross-Validation:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 80.12% | 64.82% | 57.38% | 0.6087 | 0.8465 |
| **Random Forest Classifier** | 79.65% | 63.91% | 52.14% | 0.5743 | 0.8351 |
| **XGBoost (Tuned Pipeline)** 🏆 | **81.24%** | **67.20%** | **58.42%** | **0.6251** | **0.8542** |

> **Champion Model:** **XGBoost Classifier** achieved the highest discriminative power (**0.8542 ROC-AUC**), effectively separating churners from loyal subscribers across all decision thresholds.

---

## 🔍 Exploratory Data Analysis (EDA) Insights
1. **Contract Type:** Month-to-month contracts have a **42.7% churn rate** vs. only **2.8%** for two-year contracts.
2. **Tenure Window:** Customers in their first **12 months** face the highest attrition risk.
3. **Internet Service:** Fiber optic users exhibit elevated churn (41.9%) driven by high price-to-service dissatisfaction.
4. **Protective Add-ons:** Subscribing to **TechSupport** and **OnlineSecurity** cuts churn risk by more than half.

---

## 💡 Retention Strategy Engine & Business Impact
Our system categorizes customers into three risk tiers and generates explainable, rule-based retention offers:

- **🔴 High Risk ($P \ge 60\%$):** 
  - 1-Year contract lock-in offer with a 20% discount.
  - 6 Months of free 24/7 Priority Tech Support and Security suite.
  - $10 billing credit to switch from Electronic Check to Autopay.
- **🟡 Medium Risk ($35\% \le P < 60\%$):** 
  - Proactive customer success check-in call and loyalty tier rewards.
- **🟢 Low Risk ($P < 35\%$):** 
  - Standard service engagement; cross-sell candidate for higher speed tiers.

### 💰 Financial Exposure:
- **High-Risk Monthly Revenue at Risk (MRR):** Estimated recurring billing exposure of ~$42,800/month.
- **Annualized Exposure:** Over **$513,600** in revenue at risk if unaddressed.

---

## 🛠️ Technology Stack
- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn, XGBoost
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Matplotlib, Seaborn
- **Pipeline Persistence:** Joblib
