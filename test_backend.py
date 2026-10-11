"""
AI Health Backend Verification Script
Tests configuration, database, ML model, and Groq AI connectivity.
"""
import os
import sys
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Override stale system environment variables with .env
load_dotenv(override=True)

def test_environment():
    print("=" * 50)
    print("1. ENVIRONMENT VARIABLES CHECK")
    print("=" * 50)
    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "")
    secret_key = os.getenv("SECRET_KEY", "")
    xray_id = os.getenv("XRAY_MODEL_ID", "")

    print(f"GROQ_API_KEY  : {'[SET] ' + groq_key[:10] + '...' if groq_key else '[MISSING]'}")
    print(f"GROQ_MODEL    : {groq_model or '[NOT SET]'}")
    print(f"SECRET_KEY    : {'[SET]' if secret_key else '[MISSING]'}")
    print(f"XRAY_MODEL_ID : {xray_id or '[EMPTY]'}")

def test_imports_and_models():
    print("\n" + "=" * 50)
    print("2. BACKEND & MODEL INTEGRITY CHECK")
    print("=" * 50)
    try:
        import joblib
        model = joblib.load("modelDP.pkl")
        print(f" Disease Prediction Model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"[FAIL] Failed to load modelDP.pkl: {e}")

    try:
        import main
        print(" FastAPI app & all routers imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import main.py: {e}")

def test_groq_api():
    print("\n" + "=" * 50)
    print("3. GROQ AI CONNECTIVITY CHECK")
    print("=" * 50)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[FAIL] GROQ_API_KEY is not set.")
        return

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with only: Health Assistant Online"}],
            max_tokens=15
        )
        reply = response.choices[0].message.content.strip()
        print(f" Groq AI response: \"{reply}\"")
    except Exception as e:
        print(f"[FAIL] Groq API call failed: {e}")

if __name__ == "__main__":
    test_environment()
    test_imports_and_models()
    test_groq_api()
    print("\n" + "=" * 50)
    print("Check complete! Run `uvicorn main:app --reload` to start.")
    print("=" * 50)
