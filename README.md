# 📊 Telecom Customer Churn Prediction & Retention Platform

An end-to-end Machine Learning web application for predicting customer churn risk and generating actionable customer retention insights. Built with **Scikit-Learn**, **FastAPI**, and a modern **Vanilla HTML/CSS/JavaScript** frontend.

---

## 🚀 Key Features

- **Machine Learning Inference**: Real-time customer churn probability predictions powered by the trained `LogisticRegression` model.
- **Preprocessing Pipeline**: Integrates the existing `preprocessor.pkl` (`ColumnTransformer` with `OneHotEncoder`) to process 19 raw customer inputs into 30 encoded feature dimensions.
- **Explainability & Risk Drivers**: Calculates feature-level contribution scores based on model coefficients to highlight why a customer is likely to churn or stay.
- **Actionable Retention Strategies**: Context-aware recommendations tailored to the customer's risk profile (contract upgrades, payment incentives, service bundles).
- **Interactive Single Assessment**: Form with live probability gauge, risk badges (High, Medium, Low), and 1-click persona presets.
- **Batch CSV Analysis**: Upload customer datasets to generate bulk churn predictions and export results to CSV.

---

## 📁 Repository Structure

```
Customer-Churn-Prediction/
├── backend/
│   ├── __init__.py
│   ├── main.py               # FastAPI application & route controllers
│   ├── model_service.py      # ML model loading, preprocessing & inference
│   └── schemas.py            # Pydantic data schemas & request validation
├── frontend/
│   ├── css/
│   │   └── styles.css        # Modern glassmorphic dark-theme stylesheet
│   ├── js/
│   │   └── app.js            # Frontend application controller & API client
│   └── index.html            # Single-page web application interface
├── data/
│   ├── raw/
│   │   └── telco_churn_raw.csv       # Raw Telco customer churn dataset
│   └── processed/
│       ├── cleaned_data.csv          # Cleaned dataset
│       ├── preprocessor.pkl          # Saved ColumnTransformer preprocessor
│       ├── X_train_processed.csv     # 30-feature processed training matrix
│       ├── X_test_processed.csv      # 30-feature processed testing matrix
│       ├── y_train.csv               # Training target vector
│       └── y_test.csv                # Testing target vector
├── models/
│   └── logistic_regression_model.pkl # Trained Logistic Regression model
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Exploratory Data Analysis & Preprocessing
│   └── 02_ml_models.ipynb            # ML Model training, tuning & evaluation
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone or Open the Repository
```bash
git clone https://github.com/Bareeqqayoom/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install Dependencies
Make sure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```

---

## ⚡ Running the Application

### Start the Backend & Frontend Server
Run the FastAPI application via Uvicorn:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```

- **Web Application UI**: Open [http://127.0.0.1:8001/](http://127.0.0.1:8001/) in your browser.
- **Interactive API Docs (Swagger UI)**: Open [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs).
- **API Health Check**: Open [http://127.0.0.1:8001/api/health](http://127.0.0.1:8001/api/health).

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page frontend application |
| `GET` | `/api/health` | Service health status and model metadata |
| `GET` | `/api/metadata` | Allowed feature choices, defaults, and presets |
| `POST` | `/api/predict` | Single customer churn inference with risk factor explainability |
| `POST` | `/api/predict-batch` | Batch inference via CSV file upload |

---

## 🧪 Machine Learning Details

- **Input Features (19 raw fields)**:
  - **Categorical (15)**: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.
  - **Numerical (4)**: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`.
- **Preprocessing Pipeline**: `OneHotEncoder(drop='first', handle_unknown='ignore')` for categorical features + passthrough for numerical features $\rightarrow$ **30 model features**.
- **Model**: `LogisticRegression(max_iter=3000)` with probability calibration for binary classification (`Churn = 1`, `No Churn = 0`).
