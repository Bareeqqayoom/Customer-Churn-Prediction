import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PREPROCESSOR_PATH = BASE_DIR / "data" / "processed" / "preprocessor.pkl"
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.pkl"

# Human-readable labels for the 30 transformed features
FEATURE_METADATA = {
    0: {"name": "cat__gender_Male", "label": "Gender: Male", "desc": "Customer is Male"},
    1: {"name": "cat__Partner_Yes", "label": "Has Partner", "desc": "Customer has a partner/spouse"},
    2: {"name": "cat__Dependents_Yes", "label": "Has Dependents", "desc": "Customer has dependents/children"},
    3: {"name": "cat__PhoneService_Yes", "label": "Phone Service", "desc": "Subscribed to phone service"},
    4: {"name": "cat__MultipleLines_No phone service", "label": "No Phone Service", "desc": "No telephone line"},
    5: {"name": "cat__MultipleLines_Yes", "label": "Multiple Phone Lines", "desc": "Multiple telephone lines"},
    6: {"name": "cat__InternetService_Fiber optic", "label": "Fiber Optic Internet", "desc": "High-speed Fiber Optic connection"},
    7: {"name": "cat__InternetService_No", "label": "No Internet Service", "desc": "No internet connection"},
    8: {"name": "cat__OnlineSecurity_No internet service", "label": "No Internet (Security)", "desc": "No internet service"},
    9: {"name": "cat__OnlineSecurity_Yes", "label": "Online Security", "desc": "Active cyber/online security add-on"},
    10: {"name": "cat__OnlineBackup_No internet service", "label": "No Internet (Backup)", "desc": "No internet service"},
    11: {"name": "cat__OnlineBackup_Yes", "label": "Cloud Backup", "desc": "Subscribed to cloud backup storage"},
    12: {"name": "cat__DeviceProtection_No internet service", "label": "No Internet (Device Prot)", "desc": "No internet service"},
    13: {"name": "cat__DeviceProtection_Yes", "label": "Device Protection Plan", "desc": "Hardware device warranty/protection"},
    14: {"name": "cat__TechSupport_No internet service", "label": "No Internet (Tech Support)", "desc": "No internet service"},
    15: {"name": "cat__TechSupport_Yes", "label": "Dedicated Tech Support", "desc": "Subscribed to premium tech support"},
    16: {"name": "cat__StreamingTV_No internet service", "label": "No Internet (TV)", "desc": "No internet service"},
    17: {"name": "cat__StreamingTV_Yes", "label": "Streaming TV", "desc": "Streaming TV entertainment add-on"},
    18: {"name": "cat__StreamingMovies_No internet service", "label": "No Internet (Movies)", "desc": "No internet service"},
    19: {"name": "cat__StreamingMovies_Yes", "label": "Streaming Movies", "desc": "Streaming movies entertainment add-on"},
    20: {"name": "cat__Contract_One year", "label": "1-Year Contract", "desc": "Committed to 1-year agreement"},
    21: {"name": "cat__Contract_Two year", "label": "2-Year Contract", "desc": "Committed to 2-year agreement"},
    22: {"name": "cat__PaperlessBilling_Yes", "label": "Paperless Billing", "desc": "Opted in for electronic billing"},
    23: {"name": "cat__PaymentMethod_Credit card (automatic)", "label": "Credit Card Auto-pay", "desc": "Automatic credit card payment"},
    24: {"name": "cat__PaymentMethod_Electronic check", "label": "Electronic Check Payment", "desc": "Manual electronic check payment"},
    25: {"name": "cat__PaymentMethod_Mailed check", "label": "Mailed Check Payment", "desc": "Paper mailed check payment"},
    26: {"name": "num__SeniorCitizen", "label": "Senior Citizen", "desc": "Customer is aged 65 or older"},
    27: {"name": "num__tenure", "label": "Tenure (Months)", "desc": "Duration with the telecom provider"},
    28: {"name": "num__MonthlyCharges", "label": "Monthly Charges ($)", "desc": "Current monthly subscription rate"},
    29: {"name": "num__TotalCharges", "label": "Total Charges ($)", "desc": "Cumulative amount billed to date"},
}

RAW_FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


class ChurnModelService:
    def __init__(self):
        self._load_artifacts()

    def _load_artifacts(self):
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found at: {PREPROCESSOR_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        self.model = joblib.load(MODEL_PATH)
        self.coefficients = self.model.coef_[0]
        self.intercept = float(self.model.intercept_[0])

    def prepare_dataframe(self, input_data: Dict[str, Any] | List[Dict[str, Any]]) -> pd.DataFrame:
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = pd.DataFrame(input_data)

        # Ensure all required columns exist with defaults if missing
        for col in RAW_FEATURE_ORDER:
            if col not in df.columns:
                if col == "SeniorCitizen":
                    df[col] = 0
                elif col in ["tenure", "MonthlyCharges", "TotalCharges"]:
                    df[col] = 0.0
                elif col == "gender":
                    df[col] = "Female"
                elif col == "InternetService":
                    df[col] = "Fiber optic"
                elif col == "Contract":
                    df[col] = "Month-to-month"
                elif col == "PaymentMethod":
                    df[col] = "Electronic check"
                else:
                    df[col] = "No"

        # Coerce numerical types and handle missing TotalCharges
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(1).astype(int)
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(50.0).astype(float)
        
        # TotalCharges: if None, blank or <= 0 with tenure > 0, estimate as tenure * MonthlyCharges
        def clean_total_charges(row):
            val = row.get("TotalCharges")
            if pd.isna(val) or val is None or str(val).strip() == "":
                return float(row["tenure"] * row["MonthlyCharges"])
            try:
                f_val = float(val)
                return f_val if f_val > 0 else float(row["tenure"] * row["MonthlyCharges"])
            except (ValueError, TypeError):
                return float(row["tenure"] * row["MonthlyCharges"])

        df["TotalCharges"] = df.apply(clean_total_charges, axis=1)

        # Reorder to exact raw feature order
        return df[RAW_FEATURE_ORDER]

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        df = self.prepare_dataframe(input_data)
        
        # Transform through preprocessor
        processed_arr = self.preprocessor.transform(df)
        if hasattr(processed_arr, "toarray"):
            processed_arr = processed_arr.toarray()
            
        # Match model's feature names ['0', '1', ..., '29']
        feature_cols = [str(i) for i in range(30)]
        processed_df = pd.DataFrame(processed_arr, columns=feature_cols)

        # Model inference
        prob_churn = float(self.model.predict_proba(processed_df)[0][1])
        prediction_class = int(self.model.predict(processed_df)[0])
        percentage = round(prob_churn * 100, 2)

        # Determine risk level
        if prob_churn >= 0.60:
            risk_level = "High"
        elif prob_churn >= 0.35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Calculate feature impact scores (x_i * w_i)
        features_row = processed_arr[0]
        risk_drivers = []
        protective_factors = []

        for idx, (val, coef) in enumerate(zip(features_row, self.coefficients)):
            impact = float(val * coef)
            meta = FEATURE_METADATA.get(idx, {"name": f"feature_{idx}", "label": f"Feature {idx}", "desc": ""})
            
            # Only consider active/relevant features
            if abs(impact) > 0.001:
                item = {
                    "feature": meta["name"],
                    "label": meta["label"],
                    "impact_score": round(impact, 4),
                    "direction": "increases_churn" if impact > 0 else "decreases_churn",
                    "description": meta["desc"]
                }
                if impact > 0:
                    risk_drivers.append(item)
                else:
                    protective_factors.append(item)

        # Sort top drivers and protective factors
        risk_drivers.sort(key=lambda x: x["impact_score"], reverse=True)
        protective_factors.sort(key=lambda x: x["impact_score"])  # most negative first

        # Generate retention recommendations
        recommendations = self._generate_recommendations(df.iloc[0].to_dict(), risk_level, risk_drivers)

        # Summary text
        if risk_level == "High":
            summary = f"Customer has a {percentage}% probability of churning (High Risk). Immediate retention intervention recommended."
        elif risk_level == "Medium":
            summary = f"Customer has a {percentage}% probability of churning (Medium Risk). Moderate risk detected with potential for account stabilization."
        else:
            summary = f"Customer has a {percentage}% probability of churning (Low Risk). Customer exhibits strong loyalty indicators."

        return {
            "churn_probability": round(prob_churn, 4),
            "churn_percentage": percentage,
            "prediction": prediction_class,
            "prediction_label": "Churn" if prediction_class == 1 else "No Churn",
            "risk_level": risk_level,
            "confidence": round(max(prob_churn, 1 - prob_churn) * 100, 2),
            "summary": summary,
            "retention_recommendations": recommendations,
            "top_risk_drivers": risk_drivers[:5],
            "top_protective_factors": protective_factors[:5],
            "customer_input": df.iloc[0].to_dict()
        }

    def _generate_recommendations(self, data: Dict[str, Any], risk_level: str, drivers: List[Dict[str, Any]]) -> List[str]:
        recs = []
        contract = data.get("Contract", "")
        internet = data.get("InternetService", "")
        payment = data.get("PaymentMethod", "")
        tech_support = data.get("TechSupport", "")
        security = data.get("OnlineSecurity", "")
        tenure = data.get("tenure", 0)

        if contract == "Month-to-month":
            recs.append("Offer a discounted 1-year or 2-year annual contract extension with price lock.")
        if internet == "Fiber optic" and (tech_support != "Yes" or security != "Yes"):
            recs.append("Bundle complimentary Tech Support or Online Security to improve service reliability.")
        if payment == "Electronic check":
            recs.append("Incentivize switching to automated Credit Card or Bank Transfer payments with a $5 monthly bill credit.")
        if tenure <= 6:
            recs.append("Enroll in high-touch onboarding and 30-day proactive customer satisfaction check-ins.")
        if data.get("MonthlyCharges", 0) > 80:
            recs.append("Review plan value: offer targeted value-added service bundle or loyalty loyalty credit.")
        
        if not recs:
            recs.append("Maintain high-touch customer support and continue monitoring account health.")

        return recs[:4]

    def predict_batch(self, df_raw: pd.DataFrame) -> Dict[str, Any]:
        customer_ids = df_raw["customerID"].tolist() if "customerID" in df_raw.columns else [f"Cust-{i+1}" for i in range(len(df_raw))]
        df_prepared = self.prepare_dataframe(df_raw)
        
        processed_arr = self.preprocessor.transform(df_prepared)
        if hasattr(processed_arr, "toarray"):
            processed_arr = processed_arr.toarray()
            
        feature_cols = [str(i) for i in range(30)]
        processed_df = pd.DataFrame(processed_arr, columns=feature_cols)

        probs = self.model.predict_proba(processed_df)[:, 1]
        preds = self.model.predict(processed_df)

        results = []
        churn_count = 0
        high_risk_count = 0
        med_risk_count = 0
        low_risk_count = 0

        for i, (p, prob) in enumerate(zip(preds, probs)):
            prob_f = float(prob)
            pred_i = int(p)
            pct = round(prob_f * 100, 2)
            
            if prob_f >= 0.60:
                risk = "High"
                high_risk_count += 1
            elif prob_f >= 0.35:
                risk = "Medium"
                med_risk_count += 1
            else:
                risk = "Low"
                low_risk_count += 1

            if pred_i == 1:
                churn_count += 1

            results.append({
                "row_index": i + 1,
                "customer_id": str(customer_ids[i]),
                "churn_probability": round(prob_f, 4),
                "churn_percentage": pct,
                "prediction": pred_i,
                "prediction_label": "Churn" if pred_i == 1 else "No Churn",
                "risk_level": risk
            })

        total = len(results)
        churn_rate = round((churn_count / total) * 100, 2) if total > 0 else 0.0

        return {
            "total_customers": total,
            "churn_count": churn_count,
            "non_churn_count": total - churn_count,
            "churn_rate_percentage": churn_rate,
            "high_risk_count": high_risk_count,
            "medium_risk_count": med_risk_count,
            "low_risk_count": low_risk_count,
            "results": results
        }


# Singleton instance
model_service = ChurnModelService()
