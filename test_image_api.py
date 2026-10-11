import os
import io
import requests
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"

# 1. Create sample JPEG image
img_jpg = Image.new("RGB", (200, 200), color=(100, 150, 200))
img_jpg.save("sample_xray.jpg", format="JPEG")

# 2. Create sample PNG image (with RGBA / transparency channel)
img_png = Image.new("RGBA", (200, 200), color=(120, 180, 120, 200))
img_png.save("sample_xray.png", format="PNG")

print("Generated sample_xray.jpg (JPEG) and sample_xray.png (PNG)")

# Detect whether live uvicorn server is running or fallback to direct TestClient
is_live = False
try:
    res = requests.get(f"{BASE_URL}/", timeout=1)
    if res.status_code == 200:
        is_live = True
except Exception:
    is_live = False

if is_live:
    print(f"[*] Testing against live server at {BASE_URL}...")
    def post_req(path, files=None, data=None):
        return requests.post(f"{BASE_URL}{path}", files=files, data=data)
else:
    print(f"[*] Live server not detected. Testing via FastAPI TestClient directly...")
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    def post_req(path, files=None, data=None):
        return client.post(path, files=files, data=data)

# Test 1: Upload JPEG
with open("sample_xray.jpg", "rb") as f:
    files = {"image": ("sample_xray.jpg", f, "image/jpeg")}
    data = {"image_type": "xray"}
    resp_jpeg = post_req("/api/image/analyze", files=files, data=data)
    print("\n--- JPEG (.jpg/.jpeg) Test Result ---")
    print("Status Code:", resp_jpeg.status_code)
    print("Response JSON:", resp_jpeg.json())
    assert resp_jpeg.status_code == 200
    assert "prediction" in resp_jpeg.json()
    assert "class_name" in resp_jpeg.json()["prediction"]
    assert "class_probabilities" in resp_jpeg.json()
    assert "diagnosis_summary" in resp_jpeg.json()
    assert "is_reliable" in resp_jpeg.json()

# Test 2: Upload PNG (with alpha channel)
with open("sample_xray.png", "rb") as f:
    files = {"image": ("sample_xray.png", f, "image/png")}
    data = {"image_type": "xray"}
    resp_png = post_req("/api/image/analyze", files=files, data=data)
    print("\n--- PNG (.png) Test Result ---")
    print("Status Code:", resp_png.status_code)
    print("Response JSON:", resp_png.json())
    assert resp_png.status_code == 200
    assert "prediction" in resp_png.json()
    assert "class_name" in resp_png.json()["prediction"]

# Test 3: Upload unsupported file format (e.g. text file pretending to be image)
files_bad = {"image": ("invalid.txt", io.BytesIO(b"Hello medical text"), "text/plain")}
resp_bad = post_req("/api/image/analyze", files=files_bad, data={"image_type": "xray"})
print("\n--- Unsupported Format Test Result ---")
print("Status Code:", resp_bad.status_code)
print("Response JSON:", resp_bad.json())
assert resp_bad.status_code == 400

print("\n" + "=" * 50)
print("All PNG and JPEG image format tests passed successfully!")
print("=" * 50)
