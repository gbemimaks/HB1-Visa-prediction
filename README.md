# 🧾 OFLC Visa Case Status Prediction

This project is focused on predicting **H1-B visa certification outcomes** using machine learning. It is developed as part of a business case from OFLC, where the goal is to build a deployable system that can recommend whether a visa application should be **Certified** or **Denied** based on historical data patterns.

---

## 📌 Project Objectives

- Analyze and extract insights from visa application data
- Predict the **case status** using ML models
- Deploy a live prediction API using **FastAPI** on Render
- Monitor models using **MLflow + DagsHub**

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, scikit-learn, XGBoost, TensorFlow**
- **FastAPI** for REST API
- **MLflow + DagsHub** for experiment tracking
- **Render.com** for deployment
- **Google Colab + GitHub** for development

---

## 🔁 Project Workflow

### 🔹 Phase 1: Data Preparation
- raw data of the oflc in CSV format
- Converted to Pandas DataFrame
- Cleaned & standardized (handled missing values, duplicates)
- Exploratory Data Analysis (EDA)
- Feature Engineering + Label/Target Encoding
- Scaled using MinMaxScaler
- Split into training/test sets

### 🔹 Phase 2: Model Development
- Trained Logistic Regression, Decision Tree, XGBoost, and LSTM
- Hyperparameter tuning with GridSearchCV
- Evaluated using accuracy, precision, recall, F1-score
- Final model: **XGBoost**

### 🔹 Phase 3: Deployment
- Serialized best model using `joblib`
- Built and deployed REST API via FastAPI
- Hosted on **Render.com**
- `/predict` endpoint handles real-time JSON input

### 🔹 Phase 4: Monitoring
- Used **MLflow** to track experiments and log model artifacts
- Linked to **DagsHub** for centralized dashboard

---

## 🚀 Deployment

🔗 **Live API:**  
Coming soon via Render...  
Swagger Docs at `/docs`

---

## 📦 Project Structure

📁 src/ │ ├── feature_engineering/ │ ├── utils/ │ ├── logging/ │ └── exception/ 📁 data/ 📁 models/ 📁 deployment/ 📁 notebooks/ 📁 logs/ 📄 requirements.txt 📄 render.yaml 📄 README.md
