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
        prediction = model.predict(features)[0]

        result = {
            "prediction": int(prediction)
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            result["probability"] = float(max(probabilities))

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )