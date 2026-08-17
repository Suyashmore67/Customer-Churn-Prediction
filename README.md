# 📊 Telecom Customer Churn Prediction
### **End-to-End Machine Learning Pipeline, Batch Scoring Engine, & Interactive Streamlit Web App**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview & Business Value
Customer churn—the rate at which customers stop doing business with an entity—is one of the most critical metrics for contract-based business models (such as Telecom and SaaS). Acquiring new customers is up to **5x more expensive** than retaining existing ones. 

This project delivers a **production-ready Machine Learning system** to predict customer churn using the Telco Customer Churn dataset. It translates raw behavioral data into actionable retention strategies by:
1. **Identifying High-Risk Customers**: Spotting churn indicators early so customer success teams can intervene.
2. **Handling Data Imbalance**: Utilizing **SMOTE** to ensure the model doesn't ignore the minority churn class.
3. **Tuning for Business Objectives**: Optimizing the decision threshold to prioritize **Recall** (capturing risk) while balancing **Precision** (avoiding false alarms).
4. **Dual Operations**: Providing both an **interactive web application** (for real-time manual lookups) and a **batch-scoring utility** (for enterprise database integrations).

---

## 🛠️ Tech Stack & Key Libraries
* **Data Processing & EDA**: `Pandas`, `NumPy`, `Seaborn`, `Matplotlib`
* **Machine Learning Pipeline**: `Scikit-Learn` (Logistic Regression, StandardScaler, Metrics)
* **Imbalance Resolution**: `Imbalanced-Learn` (SMOTE)
* **Model Serialization**: `Joblib`
* **Real-time App Interface**: `Streamlit`

---

## 📂 Project Architecture
```bash
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv       # Original dataset from Kaggle
│   └── scored_customer_churn.csv          # Output from batch scoring script
├── model/
│   ├── churn_model.pkl                    # Trained Logistic Regression model (SMOTE-balanced)
│   ├── feature_names.pkl                  # List of feature names matching encoded columns
│   ├── scaler.pkl                         # Fitted StandardScaler for numerical columns
│   └── threshold.txt                      # Tuned decision threshold (0.4)
├── .gitignore                             # Ignored files (virtual environments, binary weights, checkouts)
├── app.py                                 # Streamlit web application code
├── score_data.py                          # Batch scoring command-line engine
├── customer-churned.ipynb                 # Jupyter Notebook containing EDA, preprocessing, & training
└── requirements.txt                       # Project dependencies
```

---

## ⚙️ Installation & Setup

Follow these steps to run the pipeline or Streamlit application locally.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/telecom-customer-churn.git
cd telecom-customer-churn
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Operational Workflow

This project offers two operation modes matching standard corporate workflows:

### A. Real-Time Customer Profiling (Streamlit Web App)
Ideal for customer support agents or account managers looking to assess individual customers during high-touch interactions.
```bash
streamlit run app.py
```
* **Inputs**: Numerical and categorical options (Tenure, Contract type, Payment method, Internet service type, Monthly/Total charges).
* **Output**: Real-time Churn Probability percentage and visual risk classification based on the optimized threshold.

### B. Enterprise Batch Scoring (`score_data.py`)
Designed for regular cron jobs or automated data pipelines to score millions of customers at once.
```bash
python score_data.py
```
* **Inputs**: Raw customer table (`data/raw/Telco-Customer-Churn.csv`).
* **Outputs**: Saves scored dataset to `data/scored_customer_churn.csv` with predictions and risk tiers:
  * 🔴 **High Risk**: Churn Probability $\ge$ 70%
  * 🟡 **Medium Risk**: Churn Probability between 30% and 70%
  * 🟢 **Low Risk**: Churn Probability $<$ 30%

---

## 🧠 Machine Learning Pipeline

### 1. Data Cleaning & Feature Engineering
* **Parsing Space Values**: Cleaned the `TotalCharges` column, converting empty strings (corresponding to new customers with `tenure = 0`) to NaNs and imputing them with the median.
* **Feature Encoding**: Used one-hot encoding on categorical variables, removing the target and primary key column (`customerID`).
* **Standard Scaling**: Fitted a `StandardScaler` to numerical inputs (`tenure`, `MonthlyCharges`, `TotalCharges`) to prevent scale bias in optimization.

### 2. Handling Class Imbalance (SMOTE)
The target column (`Churn`) displays a significant skew (~73% non-churn vs. ~27% churn). Unbalanced training causes classifiers to default to the majority class.
* **SMOTE** (Synthetic Minority Over-sampling Technique) was applied on the training data to synthetically generate balanced samples, allowing the classifier to learn representative decision boundaries for churn indicators.

### 3. Logistic Regression & Custom Threshold Tuning
A Logistic Regression classifier (`max_iter=1000`) was trained on the oversampled dataset. Rather than using the default classification threshold ($0.5$):
* Precision/Recall trade-offs were evaluated.
* The threshold was customized to **`0.4`**.
* *Business Rationale*: It is more expensive to miss a churning customer (False Negative) than to target a loyal customer with retention incentives (False Positive). Thus, lowering the threshold to `0.4` raises the recall metric substantially.

---

## 📈 Model Performance & Evaluation

The final model evaluated on the unseen **Test Set** (using the tuned threshold of `0.4`) yielded:

### 1. Evaluation Summary
| Metric | Value |
| :--- | :---: |
| **ROC-AUC Score** | **84.03%** |
| **Overall Accuracy** | **78.00%** |
| **Churn Recall (Class 1)** | **67.00%** |
| **Churn Precision (Class 1)** | **57.00%** |
| **Churn F1-Score** | **61.00%** |

### 2. Confusion Matrix (Test Set)
```text
               Predicted Retained     Predicted Churn
Actual Retained       844 (TN)               191 (FP)
Actual Churned        124 (FN)               250 (TP)
```

---

## 🔮 Strategic Future Improvements
To scale this project to enterprise grade:
1. **Compare Tree Models**: Implement ensemble algorithms like **XGBoost**, **LightGBM**, or **Random Forest** to capture complex feature interactions.
2. **Hyperparameter Tuning**: Optimize hyperparameter grids via cross-validated searches (e.g., `GridSearchCV`).
3. **Deploy as a Microservice**: Wrap the pipeline in a **FastAPI** container to enable low-latency API integration with other SaaS systems.

---

## 👤 Developer Profile & Contact
**Developed by [Suyash MORE]**  
*Looking for Data Scientist / Machine Learning Engineer opportunities.*

* **LinkedIn**: [https://www.linkedin.com/in/suyash-more67]
* **Email**: [suyashmore6704@gmail.com]
