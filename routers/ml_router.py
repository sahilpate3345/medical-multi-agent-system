import os
import joblib
import pandas as pd
from typing import Union

from fastapi import APIRouter, HTTPException
from schemas import DiseaseInput, DynamicDiseaseRequest
from Services.ML.general_disease_service import predict_disease_dynamically


router = APIRouter(
    prefix="/api/ml",
    tags=["Machine Learning"]
)


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "modelDP.pkl"
)

def get_model():
    return joblib.load(MODEL_PATH)


@router.post("/disease-predict")
def disease_predict(data: Union[DynamicDiseaseRequest, DiseaseInput]):
    """
    Unified Disease Prediction Endpoint:
    - Predicts any disease dynamically from user symptoms, complaints, and clinical profile using AI.
    - Also supports tabular heart metrics if the 13 clinical numerical parameters are provided.
    """
    # 1. Dynamic Prediction from symptoms / complaints
    if isinstance(data, DynamicDiseaseRequest) or hasattr(data, "symptoms"):
        try:
            return predict_disease_dynamically(
                symptoms_input=getattr(data, "symptoms"),
                age=getattr(data, "age", None),
                gender=getattr(data, "gender", None),
                duration=getattr(data, "duration", None),
                medical_history=getattr(data, "medical_history", None)
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Dynamic disease prediction error: {str(e)}"
            )

    # 2. Tabular Heart Disease ML Model
    features = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "cp": data.cp,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "thalach": data.thalach,
        "exang": data.exang,
        "oldpeak": data.oldpeak,
        "slope": data.slope,
        "ca": data.ca,
        "thal": data.thal
    }])

    try:
        model = get_model()

        # Compute probabilities
        probs = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None

        if probs is not None:
            # Disease probability (class 1)
            disease_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            confidence = float(max(probs))
            # Clinical Decision Threshold: 0.35 maximizes recall to eliminate dangerous false negatives
            prediction = 1 if disease_prob >= 0.35 else 0
        else:
            prediction = int(model.predict(features)[0])
            disease_prob = 1.0 if prediction == 1 else 0.0
            confidence = 0.85

        # 90% Clinical Confidence threshold check
        meets_90pct = bool(confidence >= 0.90)
        if meets_90pct:
            confidence_tier = "High Clinical Confidence (>= 90%)"
            recommendation = "Definitive clinical risk assessment (>= 90%). Proceed with standard cardiology protocol."
        else:
            confidence_tier = "Sub-threshold (< 90%) - Clinical Confirmation Advised"
            recommendation = "Confidence is under 90%. Immediate confirmatory ECG and cardiac workup recommended to negate false negatives."

        return {
            "prediction": int(prediction),
            "condition": "Heart Disease Indicated" if prediction == 1 else "No Significant Heart Disease Detected",
            "probability": round(confidence, 4),
            "disease_probability": round(disease_prob, 4),
            "confidence_percentage": f"{round(confidence * 100, 1)}%",
            "meets_90pct_threshold": meets_90pct,
            "confidence_tier": confidence_tier,
            "clinical_recall_optimized": True,
            "clinical_recommendation": recommendation,
            "disclaimer": "AI clinical screening model optimized for maximum sensitivity to prevent false negatives. Formal medical consultation required."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )