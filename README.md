# 🛡️ AI-Based Customer Churn Prediction and Retention Strategy Recommendation System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange)
![Dataset](https://img.shields.io/badge/Dataset-Telco%20Customer%20Churn-green)

> **B.Tech Artificial Intelligence & Data Science — Machine Learning Project**  
> An end-to-end Machine Learning system for predicting telecom customer churn, identifying customer risk levels, estimating revenue at risk, and recommending suitable customer retention strategies.

---

## 📌 Project Title

**AI-Based Customer Churn Prediction and Retention Strategy Recommendation System**

---

## 👥 Team Members

| Name | Roll Number |
|---|---|
| **Roop Sai Teja.A** | **24EU01006** |
| **Harsha Nandan.K** | **24EU01025** |
| **Pradeep Matla** | **24EU01041** |

**Department:** Artificial Intelligence & Data Science  
**Degree:** Bachelor of Technology (B.Tech)

---

## 🎯 Problem Statement

Customer churn is a major challenge for subscription-based businesses such as telecommunications companies.

When customers discontinue their services, companies lose recurring revenue and may need to spend additional resources to acquire new customers.

The objective of this project is to develop a Machine Learning-based system that can:

- Predict whether a customer is likely to churn.
- Calculate the probability of customer churn.
- Classify customers into Low, Medium, and High-risk categories.
- Identify important factors associated with customer churn.
- Estimate potential monthly revenue at risk.
- Recommend suitable customer retention strategies.
- Save and reload the trained ML model.
- Make predictions on new and unseen customer data.

The project goes beyond simple churn prediction by connecting ML predictions with actionable business retention strategies.

---

# 💼 Business Motivation

Customer churn prediction can help telecom companies make proactive business decisions.

### Our system aims to:

- 📉 Reduce customer churn.
- 💰 Protect recurring monthly revenue.
- 🎯 Identify high-risk customers.
- 📊 Support data-driven customer retention.
- 🤝 Provide personalized retention recommendations.
- 💡 Help businesses prioritize retention campaigns.

Instead of waiting until a customer leaves, the company can identify customers who are likely to churn and take preventive action.

---

# 📊 Dataset

### Dataset Name

**IBM / Kaggle Telco Customer Churn Dataset**

### Dataset File

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Dataset Characteristics

- Approximately **7,043 customer records**
- Customer demographic information
- Account information
- Subscription information
- Service information
- Billing information
- Churn status

### Target Variable

```text
Churn
```

Values:

```text
Yes
No
```

This is a **binary classification problem**.

---

# 🧾 Dataset Features

| Category | Features |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod |
| Services | PhoneService, MultipleLines, InternetService |
| Security & Support | OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport |
| Entertainment | StreamingTV, StreamingMovies |
| Financial | MonthlyCharges, TotalCharges |
| Target | Churn |

---

# 🔄 Project Workflow

```text
                 Telco Customer Dataset
                          │
                          ▼
                 Data Understanding
                          │
                          ▼
                    EDA & Analysis
                          │
                          ▼
                 Data Preprocessing
                          │
                          ▼
                 Feature Engineering
                          │
                          ▼
                 Train/Test Split
                          │
                          ▼
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        Logistic       Random       XGBoost
       Regression      Forest
              │           │           │
              └───────────┼───────────┘
                          ▼
                  Model Comparison
                          │
                          ▼
               Hyperparameter Tuning
                          │
                          ▼
                  Best ML Model
                          │
                          ▼
                Churn Probability
                          │
                          ▼
                   Risk Classification
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
             Low       Medium        High
              │           │           │
              └───────────┼───────────┘
                          ▼
              Retention Recommendation
                          │
                          ▼
                  Business Insights
                          │
                          ▼
               Revenue-at-Risk Analysis
                          │
                          ▼
                 Save Model using Joblib
                          │
                          ▼
                   Reload Model
                          │
                          ▼
               New Customer Prediction
```

---

# 🔍 Exploratory Data Analysis

The project performs exploratory analysis to understand customer behavior and identify patterns associated with churn.

### EDA includes:

- Dataset shape and structure
- Data types
- Missing-value analysis
- Duplicate-value analysis
- Churn distribution
- Numerical feature distributions
- Categorical feature distributions
- Churn vs Contract
- Churn vs Tenure
- Churn vs Monthly Charges
- Churn vs Internet Service
- Churn vs Payment Method
- Churn vs Technical Support
- Churn vs Online Security
- Correlation analysis

### Example business questions

- Are month-to-month customers more likely to churn?
- Does short tenure increase churn probability?
- Does higher monthly billing affect churn?
- Does technical support reduce churn?
- Which payment methods are associated with higher churn?

---

# 🧹 Data Preprocessing

The following preprocessing steps are performed:

### 1. Missing Value Handling

`TotalCharges` is converted from string to numeric format.

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Missing values are handled appropriately before model training.

### 2. Target Encoding

```text
Yes → 1
No  → 0
```

### 3. Categorical Encoding

Categorical variables are transformed using:

```text
OneHotEncoder
```

### 4. Numerical Scaling

Numerical variables are scaled where required using:

```text
StandardScaler
```

### 5. Train/Test Split

The dataset is divided into:

```text
80% → Training
20% → Testing
```

Stratified splitting is used to maintain the class distribution.

---

# 🧠 Feature Engineering

Additional meaningful features are created to improve the ML model and provide better business insights.

Examples include:

### Service Count

Counts the number of subscribed additional services.

```text
service_count
```

### Tenure Group

Customers are grouped according to their subscription duration.

```text
New
Growing
Loyal
Long-Term
```

### Average Monthly Charges

A derived financial feature based on customer spending.

### Monthly-to-Total Charge Ratio

Helps identify differences between newer and longer-term customers.

### Support & Security Indicator

Identifies customers who have adopted services such as:

- Online Security
- Technical Support

Feature engineering is performed only when the feature has a meaningful relationship with the business problem.

---

# 🤖 Machine Learning Models

At least three Machine Learning models are evaluated.

## 1. Logistic Regression

Used as an interpretable classification baseline.

Advantages:

- Simple
- Fast
- Easy to interpret
- Suitable for binary classification

---

## 2. Random Forest

An ensemble learning algorithm that can capture non-linear relationships between customer characteristics and churn.

Advantages:

- Handles non-linear relationships
- Robust to many feature types
- Provides feature importance

---

## 3. XGBoost

A gradient boosting algorithm that is effective for structured/tabular datasets.

Advantages:

- Strong predictive performance
- Handles complex relationships
- Supports regularization
- Suitable for classification problems

---

# 📈 Model Evaluation

The models are compared using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Confusion Matrix

A confusion matrix is used to understand:

- True Positives
- True Negatives
- False Positives
- False Negatives

### ROC Curve

ROC-AUC is used to evaluate how effectively the model separates customers who churn from customers who do not churn.

---

# 🏆 Model Comparison

The final model comparison table is generated directly from the actual Colab execution.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | Generated from experiment | Generated | Generated | Generated | Generated |
| Random Forest | Generated from experiment | Generated | Generated | Generated | Generated |
| XGBoost | Generated from experiment | Generated | Generated | Generated | Generated |
| Tuned Best Model | Generated from experiment | Generated | Generated | Generated | Generated |

> **Note:** Actual values should be updated here after running the final Colab notebook. No performance values are manually assumed or copied from another implementation.

---

# ⚙️ Hyperparameter Tuning

Hyperparameter optimization is performed using:

```text
GridSearchCV
```

or

```text
RandomizedSearchCV
```

The tuning process searches for the best combination of model parameters.

Example XGBoost parameters:

```text
n_estimators
max_depth
learning_rate
subsample
```

The optimized model is evaluated again on the unseen test set.

---

# 🏅 Final Model Selection

The final model is selected based on the actual experimental results.

For customer churn prediction, **Recall and ROC-AUC are particularly important** because failing to identify a customer who is likely to churn can result in lost business.

However, model selection is based on the overall evaluation rather than automatically assuming that one algorithm will be the best.

---

# 💾 Model Persistence

The complete preprocessing and ML pipeline is saved using **Joblib**.

Example:

```python
import joblib

joblib.dump(
    best_model,
    "customer_churn_retention_pipeline.joblib"
)
```

The model can later be loaded without retraining:

```python
loaded_model = joblib.load(
    "customer_churn_retention_pipeline.joblib"
)
```

This demonstrates that the trained model can be reused for real-world predictions.

---

# 🔮 New Customer Prediction

The system accepts new customer information and generates:

```text
Churn Prediction
Churn Probability
Risk Level
Retention Recommendation
```

Example:

```text
Customer Churn Prediction
-------------------------
Churn Probability : 78.5%
Prediction        : Likely to Churn
Risk Level        : HIGH
```

The prediction is generated using the saved model without retraining.

---

# 🚦 Customer Risk Classification

Customers are divided into three risk categories.

### 🔴 High Risk

```text
Churn Probability ≥ 70%
```

These customers require immediate retention attention.

### 🟡 Medium Risk

```text
40% ≤ Churn Probability < 70%
```

These customers should be monitored and provided with targeted offers.

### 🟢 Low Risk

```text
Churn Probability < 40%
```

These customers generally require no immediate retention intervention.

> The exact thresholds can be adjusted based on business requirements.

---

# 💡 Retention Strategy Recommendation

This is the main business-focused feature of the project.

The Machine Learning model predicts **churn probability**, while a separate rule-based business layer converts the prediction into an actionable recommendation.

### 🔴 High-Risk Customer

Possible recommendations:

- Offer a long-term contract discount.
- Provide technical support benefits.
- Offer security-service bundles.
- Provide personalized promotional offers.
- Encourage automated payment methods.

### 🟡 Medium-Risk Customer

Possible recommendations:

- Provide loyalty benefits.
- Offer targeted promotional campaigns.
- Conduct proactive customer-support follow-ups.
- Monitor future churn probability.

### 🟢 Low-Risk Customer

Possible recommendations:

- Continue normal customer engagement.
- Offer optional cross-selling opportunities.
- Provide loyalty benefits.

The recommendation engine is intentionally separated from the ML model so that the business rules can be changed without retraining the model.

---

# 💰 Revenue-at-Risk Analysis

The project estimates the potential monthly revenue associated with high-risk customers.

A simplified calculation is:

```text
Monthly Revenue at Risk
=
Sum of Monthly Charges
of High-Risk Customers
```

If currency conversion is required for presentation, the conversion rate should be clearly documented and applied consistently.

### Important

Revenue at risk represents **potential financial exposure**, not guaranteed future loss.

The calculation helps businesses understand the financial importance of customer retention.

---

# 🖥️ Application / Dashboard

A Streamlit dashboard can be used to demonstrate the project.

The dashboard can display:

### Business Overview

- Total customers
- Churn rate
- High-risk customers
- Medium-risk customers
- Low-risk customers
- Estimated revenue at risk

### Model Performance

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### Customer Prediction

Users can enter customer information and receive:

```text
Churn Probability
Risk Level
Retention Recommendation
```

---

# 🛠️ Technologies Used

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Model Persistence

- Joblib

### Dashboard

- Streamlit

### Development Environment

- Google Colab
- Jupyter Notebook
- GitHub

---

# 📁 Project Structure

```text
customer-churn-retention-system/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── customer_churn_project.ipynb
│
├── models/
│   └── customer_churn_retention_pipeline.joblib
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── retention_strategy.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🚀 How to Run

## Option 1 — Google Colab

Open the project notebook in Google Colab and run the cells sequentially.

The notebook performs:

```text
Dataset Loading
↓
EDA
↓
Preprocessing
↓
Feature Engineering
↓
Model Training
↓
Model Comparison
↓
Hyperparameter Tuning
↓
Model Saving
↓
Prediction
```

---

## Option 2 — Run Locally

Clone the repository:

```bash
git clone https://github.com/Rst06/customer-churn-retention-system.git
```

Enter the project directory:

```bash
cd customer-churn-retention-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Run prediction:

```bash
python src/predict.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

# 📚 Project Deliverables

The project provides:

- ✅ Real-world dataset
- ✅ Exploratory Data Analysis
- ✅ Data preprocessing
- ✅ Feature engineering
- ✅ Feature selection/analysis
- ✅ Three or more ML models
- ✅ Model comparison
- ✅ Hyperparameter tuning
- ✅ Best-model selection
- ✅ Integrated ML pipeline
- ✅ Joblib model persistence
- ✅ Model reload
- ✅ New customer prediction
- ✅ Customer risk classification
- ✅ Retention strategy recommendation
- ✅ Revenue-at-risk analysis
- ✅ Business dashboard

---

# 🌍 Real-World Application

The proposed system can help subscription-based businesses identify customers who are likely to leave.

A telecom company can use the system to:

1. Identify high-risk customers.
2. Understand the factors contributing to churn.
3. Estimate potential revenue exposure.
4. Prioritize retention campaigns.
5. Provide personalized offers.
6. Monitor customer risk over time.

The same concept can be adapted to:

- Telecom
- Internet Service Providers
- Streaming platforms
- SaaS companies
- Subscription businesses
- Banking
- Insurance
- E-commerce

---

# ⚠️ Limitations

- The dataset is based on a publicly available Telco customer dataset.
- Predictions depend on the quality and representativeness of the training data.
- Churn probability is not a guarantee that a customer will leave.
- Retention recommendations are business rules and should be validated using real company data.
- Revenue-at-risk is an estimate rather than guaranteed financial loss.
- Model performance may change when applied to a different telecom company or market.

---

# 🔮 Future Enhancements

Future versions could include:

- Real-time customer data integration.
- Advanced explainable AI using SHAP.
- Customer lifetime value prediction.
- Automated campaign generation.
- A/B testing of retention offers.
- Real-time churn monitoring.
- Cloud deployment.
- Database integration.
- Automated email/SMS retention campaigns.
- Deep Learning-based churn prediction.

---

# 👨‍💻 Team

### Roopsai.A
**24EU01006**

### Harsha Nandan.K
**24EU01025**

### Pradeep Matla
**24EU01041**

**B.Tech — Artificial Intelligence & Data Science**

---

# 📌 Conclusion

This project demonstrates a complete Machine Learning workflow for a real-world business problem.

The system moves beyond simple customer churn prediction by connecting:

```text
Machine Learning
      +
Customer Risk Analysis
      +
Business Insights
      +
Revenue-at-Risk
      +
Retention Strategy
```

The final objective is to help businesses make proactive, data-driven customer retention decisions.

---

## ⭐ Project Highlights

```text
✔ Real-world Telecom Dataset
✔ Complete EDA
✔ Data Preprocessing
✔ Feature Engineering
✔ Multiple ML Models
✔ Hyperparameter Tuning
✔ Model Comparison
✔ Model Persistence
✔ New Customer Prediction
✔ Risk Classification
✔ Retention Recommendation
✔ Revenue-at-Risk Analysis
✔ Business Decision Support
✔ Google Colab
✔ GitHub
✔ Streamlit Dashboard
```

---

### 📜 Disclaimer

This project is developed for academic and educational purposes. The churn predictions, revenue-at-risk calculations, and retention recommendations should not be treated as guaranteed business outcomes without validation using real organizational data.
