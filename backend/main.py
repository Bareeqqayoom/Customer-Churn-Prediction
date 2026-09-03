import io
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionResponse
)
from backend.model_service import model_service, RAW_FEATURE_ORDER

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready REST API for Telecom Customer Churn Prediction and Risk Factor Explainability",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "model_loaded": True,
        "preprocessor_loaded": True,
        "features_expected": 30,
        "raw_inputs_expected": 19
    }


@app.get("/api/metadata", tags=["Metadata"])
def get_metadata():
    """Returns dropdown choices, default values, and sample customer personas."""
    return {
        "raw_features": RAW_FEATURE_ORDER,
        "options": {
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["No", "Yes"],
            "Dependents": ["No", "Yes"],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["No", "Yes", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "Yes", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        },
        "presets": {
            "high_risk": {
                "name": "High Risk Customer",
                "description": "Month-to-month Fiber Optic subscriber with Electronic check payment and short tenure.",
                "data": {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 2,
                    "PhoneService": "Yes",
                    "MultipleLines": "Yes",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 98.50,
                    "TotalCharges": 197.00
                }
            },
            "low_risk": {
                "name": "Loyal Low Risk Customer",
                "description": "Two-year contract subscriber with multiple services, long tenure, and automated payments.",
                "data": {
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "Yes",
                    "tenure": 60,
                    "PhoneService": "Yes",
                    "MultipleLines": "Yes",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "Yes",
                    "TechSupport": "Yes",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Two year",
                    "PaperlessBilling": "No",
                    "PaymentMethod": "Credit card (automatic)",
                    "MonthlyCharges": 85.00,
                    "TotalCharges": 5100.00
                }
            },
            "medium_risk": {
                "name": "Moderate Risk Customer",
                "description": "1-year contract subscriber with Fiber optic and partial add-ons.",
                "data": {
                    "gender": "Female",
                    "SeniorCitizen": 1,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 18,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "Yes",
                    "Contract": "One year",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Bank transfer (automatic)",
                    "MonthlyCharges": 79.20,
                    "TotalCharges": 1425.60
                }
            }
        }
    }


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerInput):
    """Predict churn probability and return explainability factors for a single customer."""
    try:
        data_dict = customer.model_dump()
        result = model_service.predict_single(data_dict)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/api/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_csv(file: UploadFile = File(...)):
    """Upload a CSV file of customers and receive batch churn predictions."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a .csv format."
        )

    try:
        contents = await file.read()
        df_uploaded = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if df_uploaded.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV file is empty."
            )

        result = model_service.predict_batch(df_uploaded)
        return result
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file must be UTF-8 encoded."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )


# Serve index.html at root
@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(str(index_file), media_type="text/html")


# Mount static frontend assets
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    if (FRONTEND_DIR / "css").is_dir():
        app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    if (FRONTEND_DIR / "js").is_dir():
        app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

