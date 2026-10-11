"""
10 Dynamic Clinical Test Cases Evaluation
"""
import sys
import time

# Ensure utf-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from Services.ML.general_disease_service import predict_disease_dynamically

CASES = [
    {
        "id": 1,
        "specialty": "Cardiology (Emergency)",
        "patient": "Male, 62 yrs",
        "symptoms": "Severe crushing retrosternal chest pain radiating to left shoulder and jaw, profuse cold sweats, shortness of breath for 45 minutes",
        "duration": "45 minutes",
        "expected": "Acute Myocardial Infarction",
        "age": 62, "gender": "male"
    },
    {
        "id": 2,
        "specialty": "Pulmonology",
        "patient": "Female, 28 yrs",
        "symptoms": "Sudden severe wheezing, chest tightness, acute breathlessness triggered by cold air, unable to speak full sentences",
        "duration": "2 hours",
        "expected": "Acute Asthma Exacerbation",
        "age": 28, "gender": "female"
    },
    {
        "id": 3,
        "specialty": "Neurology (Emergency)",
        "patient": "Male, 67 yrs",
        "symptoms": "Sudden onset right-sided facial droop, right arm weakness, slurred speech (dysarthria), confusion",
        "duration": "30 minutes",
        "expected": "Acute Ischemic Stroke",
        "age": 67, "gender": "male"
    },
    {
        "id": 4,
        "specialty": "Endocrinology",
        "patient": "Female, 22 yrs",
        "symptoms": "Unquenchable excessive thirst (polydipsia), frequent urination (polyuria), fruity-smelling breath, nausea, rapid deep breathing, extreme fatigue",
        "duration": "2 days",
        "expected": "Diabetic Ketoacidosis (DKA)",
        "age": 22, "gender": "female"
    },
    {
        "id": 5,
        "specialty": "General Surgery",
        "patient": "Male, 21 yrs",
        "symptoms": "Periumbilical abdominal pain migrating to right lower abdomen (McBurney's point), low-grade fever, nausea, vomiting, rebound tenderness",
        "duration": "14 hours",
        "expected": "Acute Appendicitis",
        "age": 21, "gender": "male"
    },
    {
        "id": 6,
        "specialty": "Infectious Disease",
        "patient": "Female, 34 yrs",
        "symptoms": "Sudden high fever (103°F), severe retro-orbital eye pain, intense bone and joint aching ('breakbone'), petechial rash",
        "duration": "3 days",
        "expected": "Dengue Fever",
        "age": 34, "gender": "female"
    },
    {
        "id": 7,
        "specialty": "Dermatology",
        "patient": "Male, 40 yrs",
        "symptoms": "Well-demarcated raised red plaques covered with silvery-white scales on bilateral elbows and knees, pitted nails",
        "duration": "3 months",
        "expected": "Psoriasis Vulgaris",
        "age": 40, "gender": "male"
    },
    {
        "id": 8,
        "specialty": "Nephrology / Urology",
        "patient": "Female, 48 yrs",
        "symptoms": "High spiking fever with shaking chills, severe right flank pain, burning urination, nausea, costovertebral angle tenderness",
        "duration": "24 hours",
        "expected": "Acute Pyelonephritis",
        "age": 48, "gender": "female"
    },
    {
        "id": 9,
        "specialty": "Rheumatology",
        "patient": "Female, 52 yrs",
        "symptoms": "Bilateral symmetrical swelling, warmth, and pain in MCP and PIP hand joints, severe morning stiffness lasting over 2 hours",
        "duration": "6 weeks",
        "expected": "Rheumatoid Arthritis",
        "age": 52, "gender": "female"
    },
    {
        "id": 10,
        "specialty": "Pediatrics / ENT",
        "patient": "Male, 6 yrs",
        "symptoms": "Sudden severe right ear pain (otalgia), pulling at ear, fever of 102°F, irritability, crying, temporary decreased hearing following cold",
        "duration": "1 day",
        "expected": "Acute Otitis Media",
        "age": 6, "gender": "male"
    }
]

def main():
    print("=" * 105, flush=True)
    print("EVALUATING 10 DIVERSE CLINICAL TEST CASES ON DYNAMIC AI ENGINE", flush=True)
    print("=" * 105, flush=True)

    results = []

    for c in CASES:
        print(f"\n--- [Case {c['id']}/10] {c['specialty']} ---", flush=True)
        print(f"Patient: {c['patient']} | Duration: {c['duration']}", flush=True)
        print(f"Symptoms: {c['symptoms']}", flush=True)
        print(f"Target Diagnostic Focus: {c['expected']}", flush=True)

        t0 = time.time()
        try:
            res = predict_disease_dynamically(
                symptoms_input=c["symptoms"],
                age=c["age"],
                gender=c["gender"],
                duration=c["duration"]
            )
            elapsed = round(time.time() - t0, 2)

            pred = res.get("predicted_disease", "N/A")
            conf = res.get("confidence_percentage", "N/A")
            sev = res.get("severity", "N/A")
            spec = res.get("specialist", "N/A")
            meets_90 = res.get("meets_90pct_threshold", False)
            diffs = [d.get("disease", "") for d in res.get("differential_diagnoses", [])]
            tests = res.get("recommended_tests", [])

            print(f"  -> Predicted: {pred} ({conf})", flush=True)
            print(f"  -> Severity: {sev} | Specialist: {spec}", flush=True)
            print(f"  -> >=90% Threshold Met: {'YES' if meets_90 else 'NO (Clinical confirmation flag)'}", flush=True)
            print(f"  -> Top Differentials: {', '.join(diffs[:2])}", flush=True)
            print(f"  -> Recommended Tests: {', '.join(tests[:2])}", flush=True)
            print(f"  -> Latency: {elapsed}s", flush=True)

            results.append({
                "id": c["id"],
                "specialty": c["specialty"],
                "expected": c["expected"],
                "predicted": pred,
                "confidence": conf,
                "severity": sev,
                "meets_90": meets_90,
                "specialist": spec,
                "latency": elapsed,
                "status": "PASS"
            })
        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            print(f"  -> ERROR: {str(e)}", flush=True)
            results.append({
                "id": c["id"],
                "specialty": c["specialty"],
                "expected": c["expected"],
                "predicted": "ERROR",
                "confidence": "0%",
                "severity": "N/A",
                "meets_90": False,
                "specialist": "N/A",
                "latency": elapsed,
                "status": "FAIL"
            })

    print("\n" + "=" * 105, flush=True)
    print("FINAL CLINICAL EVALUATION SUMMARY: 10 DIVERSE DYNAMIC CASES", flush=True)
    print("=" * 105, flush=True)
    print(f"{'#':<3} | {'Specialty':<22} | {'Predicted Disease':<30} | {'Conf':<6} | {'Severity':<10} | {'Specialist':<18} | {'Status'}", flush=True)
    print("-" * 115, flush=True)
    for r in results:
        print(f"{r['id']:<3} | {r['specialty'][:22]:<22} | {r['predicted'][:30]:<30} | {r['confidence']:<6} | {r['severity']:<10} | {r['specialist'][:18]:<18} | {r['status']}", flush=True)
    print("=" * 105, flush=True)

if __name__ == "__main__":
    main()
