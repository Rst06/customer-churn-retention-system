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
