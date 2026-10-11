"""
Test Single Route /api/ml/disease-predict
"""
import sys
from fastapi.testclient import TestClient
from main import app
from Services.ML.general_disease_service import predict_disease_dynamically

client = TestClient(app)

def test_service_directly():
    print("=" * 60)
    print("1. DIRECT SERVICE DYNAMIC PREDICTION TEST")
    print("=" * 60)
    result = predict_disease_dynamically(
        symptoms_input="Sudden severe throbbing headache on one side of head, nausea, sensitivity to light",
        age=32,
        gender="Female"
    )
    print("Predicted Disease:", result.get("predicted_disease"))
    print("Confidence:", result.get("confidence_percentage"))
    print("Severity:", result.get("severity"))
    print("Detected Symptoms:", result.get("detected_symptoms"))
    print("Specialist:", result.get("specialist"))
    assert result.get("predicted_disease") is not None
    assert result.get("confidence") > 0.0
    print("[PASS] Direct service dynamic prediction succeeded!")

def test_single_route_dynamic_mode():
    print("\n" + "=" * 60)
    print("2. SINGLE ROUTE /api/ml/disease-predict (DYNAMIC SYMPTOMS MODE)")
    print("=" * 60)
    payload = {
        "symptoms": "Fever, headache, and mild cough",
        "age": 23,
        "gender": "male",
        "duration": "2 days",
        "medical_history": "No known medical conditions"
    }
    res = client.post("/api/ml/disease-predict", json=payload)
    print("Status Code:", res.status_code)
    data = res.json()
    print("Predicted Disease:", data.get("predicted_disease"))
    print("Confidence:", data.get("confidence_percentage"))
    print("Specialist:", data.get("specialist"))
    assert res.status_code == 200
    assert data.get("predicted_disease") is not None
    print("[PASS] Single route dynamic mode succeeded!")

def test_single_route_tabular_mode():
    print("\n" + "=" * 60)
    print("3. SINGLE ROUTE /api/ml/disease-predict (TABULAR HEART MODEL MODE)")
    print("=" * 60)
    res_tab = client.post("/api/ml/disease-predict", json={
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    })
    print("Status Code:", res_tab.status_code)
    data_tab = res_tab.json()
    print("Prediction:", data_tab.get("prediction"), "Probability:", data_tab.get("probability"))
    assert res_tab.status_code == 200
    assert "prediction" in data_tab
    print("[PASS] Single route tabular mode succeeded!")

if __name__ == "__main__":
    test_service_directly()
    test_single_route_dynamic_mode()
    test_single_route_tabular_mode()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY ON SINGLE ROUTE /api/ml/disease-predict!")
    print("=" * 60)
