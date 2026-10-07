# 🛡️ AI-Based Customer Churn Prediction and Retention Strategy Recommendation System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bJipMTTrMqYLjtn1DpAEj2a7Wy1h7e5E?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20XGBoost-orange)
![Currency](https://img.shields.io/badge/Financials-Indian%20Rupees%20(%E2%82%B9)-green)

> **B.Tech Artificial Intelligence & Data Science — Machine Learning Capstone Project**  
> An End-to-End Decision-Support Pipeline for Telecom Subscriber Churn Prediction, Risk Stratification, Revenue-at-Risk Quantification, and Automated Retention Strategy Advisory.

---

## 1. Project Title
**AI-Based Customer Churn Prediction and Retention Strategy Recommendation System**

---

## 2. Team Members & Affiliation
- **Student Name:** [Your Name]
- **Roll Number / PRN:** [Your Roll Number]
- **Department:** Department of Artificial Intelligence & Data Science
- **Degree:** Bachelor of Technology (B.Tech)
- **Academic Year:** 2025–2026

---

## 3. Problem Statement
Customer attrition (churn) directly impacts the recurring revenue of subscription-based telecommunications providers (e.g., Airtel, Jio, Vi). Acquiring a new telecom subscriber costs **5 to 7 times more** than retaining an existing one. Conventional telecom operations only react to cancellations after a customer files a port-out (MNP) request. The objective of this project is to build an end-to-end Machine Learning decision-support system that predicts customer churn probability ahead of time, quantifies monetary **Revenue at Risk (in ₹)**, and automatically generates explainable retention action plans.

---

## 4. Business Motivation
- **Protect Monthly Recurring Revenue (MRR):** Identify accounts likely to cancel before their billing cycle ends.
- **Capital Efficiency:** Prevent wasteful spending of retention discounts on customers who are already loyal.
- **Actionable Decision Support:** Bridge the gap between an abstract probability score and tangible marketing/customer success interventions in Indian Rupees (₹).

---

## 5. Dataset & Data Source
- **Dataset:** IBM / Kaggle Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- **Scale:** 7,043 subscriber records.
- **Target Variable:** `Churn` (`Yes` / `No`) — Binary classification with ~26.5% minority class representation.
- **Currency Context:** Converted to Indian Rupees (₹) at standard conversion rate (1 USD ≈ ₹83) to represent realistic Indian broadband and postpaid plan billing (₹1,500 – ₹9,800/month).

---

## 6. Dataset Features & Descriptions

| Category | Attributes | Business Description |
| :--- | :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Subscriber demographic profile and household status |
| **Account Tenancy** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` | Subscription duration, billing mechanism, and contract type |
| **Subscribed Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Catalog of active connectivity and add-on subscriptions |
| **Financials (₹)** | `MonthlyCharges` (₹), `TotalCharges` (₹) | Current monthly plan rate and cumulative lifetime spend |
| **Target** | `Churn` | Whether the customer terminated service within the last month |

---

## 7. Methodology

---

## 8. Exploratory Data Analysis (EDA) Insights
1. **Contract Type Dominance:** Subscribers on **Month-to-month contracts** exhibit an alarming **42.7% churn rate**, compared to just **11.3%** for 1-year and **2.8%** for 2-year contracts.
2. **Tenure Critical Window:** Highest churn occurs within the first **0 to 12 months** of customer onboarding. Customers surviving past 48 months show loyal retention.
3. **Fiber Optic Disparity:** Fiber optic users churn at **41.9%**, significantly higher than DSL (19.0%), driven by high monthly costs without adequate bundling.
4. **Protective Add-ons:** Subscribing to **TechSupport** and **OnlineSecurity** cuts churn probability by more than half.
5. **Payment Friction:** Electronic check users exhibit 45.3% churn, compared to < 16% for automated bank autopay and credit card billing.

---

## 9. Data Preprocessing & Leakage Prevention
- **Type Conversion:** Converted `TotalCharges` from whitespace strings to float. Imputed ₹0.0 for new customers (`tenure = 0`).
- **Target Mapping:** Standardized `Yes` $\rightarrow$ 1, `No` $\rightarrow$ 0.
- **Zero Data Leakage:** Imputers and scalers were enclosed inside Scikit-Learn's `ColumnTransformer` and fit **strictly on the 80% training split**.
- **Stratified Partition:** Used `stratify=y` on the 80/20 train-test split to ensure identical class distributions across both partitions.

---

## 10. Feature Engineering
We engineered five domain-motivated features:
1. `service_count`: Total active services subscribed (measures customer switching barrier/stickiness).
2. `tenure_group`: Lifecycle cohorts (`0-12m`, `13-24m`, `25-48m`, `49-72m`).
3. `avg_monthly_charges`: Cumulative lifetime spend divided by tenure (`TotalCharges / (tenure + 1)`).
4. `monthly_to_total_ratio`: Identifies new high-spend accounts vs mature long-term accounts.
5. `has_security_techsupport`: Interaction flag indicating complete technical support adoption.

---

## 11. Machine Learning Models Evaluated
1. **Logistic Regression:** Linear baseline with balanced class weights for direct odds-ratio interpretability.
2. **Random Forest Classifier:** Bagging ensemble of 150 decision trees capturing non-linear relationships.
3. **XGBoost Classifier:** Extreme Gradient Boosted regularized decision trees.

---

## 12. Model Comparison & Benchmarking Results

*Evaluated on the 20% Stratified Unseen Test Set (1,409 customers):*

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 80.12% | 64.82% | 57.38% | 0.6087 | 0.8465 |
| **Random Forest Classifier** | 79.65% | 63.91% | 52.14% | 0.5743 | 0.8351 |
| **XGBoost (Base)** | 80.48% | 65.52% | 56.42% | 0.6063 | 0.8491 |
| **XGBoost (Tuned Pipeline)** 🏆 | **81.24%** | **67.20%** | **58.42%** | **0.6251** | **0.8542** |

---

## 13. Hyperparameter Tuning
- **Method:** 5-Fold Stratified `GridSearchCV` on the full end-to-end pipeline.
- **Search Grid:** `n_estimators: [100, 150]`, `max_depth: [3, 4]`, `learning_rate: [0.05, 0.1]`.
- **Optimal Hyperparameters:** `max_depth=3`, `learning_rate=0.1`, `n_estimators=100`.
- **Result:** Tuning boosted test ROC-AUC to **0.8542** and test F1-score to **0.6251**, effectively regularizing tree depth against overfitting.

---

## 14. Final Model Selection
**XGBoost (Tuned Pipeline)** was chosen as the champion model.  
- **Business Rationale:** In telecom customer churn, **Recall and ROC-AUC** are much more critical than raw Accuracy. A false negative (missing a churning customer) results in losing the subscriber's entire Customer Lifetime Value (₹50,000+ LTV), whereas a false positive only sends a low-cost promotional email.
- XGBoost demonstrated the highest discriminative ranking power (0.8542 AUC) across all decision thresholds.

---

## 15. Model Persistence
- The complete pipeline (Feature Engineering + Imputation + Scaling + One-Hot Encoding + Tuned XGBoost) was serialized to `models/customer_churn_retention_pipeline.joblib`.
- Loading with `joblib.load()` requires **zero retraining** and accepts raw JSON/dictionary inputs for instantaneous real-time prediction.

---

## 16. New Customer Prediction Demonstration

```python
from src.predict import predict_customer

sample_customer = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
    'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': 74.50, 'TotalCharges': 149.00
}
result = predict_customer(sample_customer)
```
## 17. Retention Strategy Recommendation Engine
Our system enforces strict architectural separation between statistical ML inference and business logic:
- **🔴 High Risk ($P \ge 60\%$):** 
  - 1-Year contract lock-in offer with 20% discount.
  - 6 Months of free 24/7 Priority Tech Support and Router Security pack.
  - Instant ₹500 bill credit on enrolling in UPI / Bank Autopay.
- **🟡 Medium Risk ($35\% \le P < 60\%$):** 
  - Proactive customer relationship check-in call.
  - Complimentary OTT bundle / speed boost loyalty milestones.
- **🟢 Low Risk ($P < 35\%$):** 
  - Standard service engagement; cross-sell candidate for Smart Home & IoT add-ons.

---

## 18. Business Revenue at Risk Analysis (INR / ₹)
- **Calculation Formula:**
  $$\text{Monthly Revenue at Risk (MRR)} = \sum_{i \in \text{High Risk}} \text{MonthlyCharges}_i \times 83$$
- **Portfolio Exposure (Test Cohort):**
  - **Total Monthly Billing Evaluated:** ~₹77,40,000
  - **High-Risk Monthly Revenue at Risk:** **~₹35,50,000 / month** (45.8% of portfolio revenue)
  - **Annualized Exposure:** Over **₹4.26 Crore** in potential recurring billing loss.
- **Business Meaning:** Gives leadership a clear financial justification to fund retention discount campaigns.

---

## 19. How to Run the Project

### Option A: Run Live in Google Colab (Recommended)
Click the badge below to run the complete notebook with interactive sliders in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bJipMTTrMqYLjtn1DpAEj2a7Wy1h7e5E?usp=sharing)

### Option B: Run Locally on Your Machine
```bash
# 1. Clone repository
git clone https://github.com/Rst06/customer-churn-retention-system.git
cd customer-churn-retention-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models and export pipeline
python src/train.py

# 4. Run test prediction
python src/predict.py

# 5. Launch interactive web dashboard
streamlit run app.py
# 5. Launch interactive web dashboard
streamlit run app.py
