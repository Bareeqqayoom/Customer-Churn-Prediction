from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any


class CustomerInput(BaseModel):
    # Demographics
    gender: Literal["Female", "Male"] = Field(
        default="Female",
        description="Customer gender"
    )
    SeniorCitizen: Literal[0, 1] = Field(
        default=0,
        description="Whether customer is senior citizen (0=No, 1=Yes)"
    )
    Partner: Literal["No", "Yes"] = Field(
        default="No",
        description="Whether customer has a partner"
    )
    Dependents: Literal["No", "Yes"] = Field(
        default="No",
        description="Whether customer has dependents"
    )
    
    # Phone Services
    PhoneService: Literal["No", "Yes"] = Field(
        default="Yes",
        description="Whether customer has phone service"
    )
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(
        default="No",
        description="Whether customer has multiple lines"
    )
    
    # Internet Services
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        default="Fiber optic",
        description="Customer internet service provider"
    )
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has online security add-on"
    )
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has online backup"
    )
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has device protection"
    )
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has premium tech support"
    )
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has streaming TV"
    )
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(
        default="No",
        description="Whether customer has streaming movies"
    )
    
    # Account & Contract Information
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        default="Month-to-month",
        description="The contract term of the customer"
    )
    PaperlessBilling: Literal["No", "Yes"] = Field(
        default="Yes",
        description="Whether customer has paperless billing"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = Field(
        default="Electronic check",
        description="Payment method used by the customer"
    )
    
    # Numerical Charges
    tenure: int = Field(
        default=1,
        ge=0,
        le=120,
        description="Number of months the customer has stayed with the company"
    )
    MonthlyCharges: float = Field(
        default=70.35,
        ge=0.0,
        description="The amount charged to the customer monthly"
    )
    TotalCharges: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="The total amount charged (optional; auto-estimated as tenure * MonthlyCharges if None)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }
    }


class FeatureImpact(BaseModel):
    feature: str
    label: str
    impact_score: float
    direction: Literal["increases_churn", "decreases_churn"]
    description: str


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_percentage: float
    prediction: int  # 0 or 1
    prediction_label: str  # "Churn" or "No Churn"
    risk_level: Literal["Low", "Medium", "High"]
    confidence: float
    summary: str
    retention_recommendations: List[str]
    top_risk_drivers: List[FeatureImpact]
    top_protective_factors: List[FeatureImpact]
    customer_input: Dict[str, Any]


class BatchPredictionItem(BaseModel):
    row_index: int
    customer_id: Optional[str] = None
    churn_probability: float
    churn_percentage: float
    prediction: int
    prediction_label: str
    risk_level: Literal["Low", "Medium", "High"]


class BatchPredictionResponse(BaseModel):
    total_customers: int
    churn_count: int
    non_churn_count: int
    churn_rate_percentage: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    results: List[BatchPredictionItem]
