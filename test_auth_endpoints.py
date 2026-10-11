"""
Test Authentication Flow: Login, Profile, and Admin Endpoints
"""
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_auth_flow():
    print("=" * 60)
    print("TESTING AUTH FLOW (LOGIN, PROFILE, ADMIN)")
    print("=" * 60)

    # 1. Admin Login (JSON format)
    login_payload = {
        "email": "admin@example.com",
        "password": "admin123"
    }
    res_login = requests.post(f"{BASE_URL}/api/auth/login", json=login_payload)
    print("1. Admin Login Status:", res_login.status_code)
    assert res_login.status_code == 200, f"Login failed: {res_login.text}"
    login_data = res_login.json()
    token = login_data["access_token"]
    role = login_data["role"]
    print("   Token received! Role:", role)
    assert role == "admin"

    headers = {"Authorization": f"Bearer {token}"}

    # 2. Access Profile
    res_prof = requests.get(f"{BASE_URL}/api/auth/profile", headers=headers)
    print("\n2. Profile Endpoint Status:", res_prof.status_code)
    print("   Response:", res_prof.json())
    assert res_prof.status_code == 200
    assert res_prof.json()["user"]["email"] == "admin@example.com"

    # 3. Access Admin Panel with Admin Token
    res_admin = requests.get(f"{BASE_URL}/api/auth/admin", headers=headers)
    print("\n3. Admin Panel Status (with admin token):", res_admin.status_code)
    print("   Response:", res_admin.json())
    assert res_admin.status_code == 200
    assert "Welcome Admin" in res_admin.json()["message"]

    # 4. OAuth2 Form login (Swagger Authorize button simulation)
    form_data = {
        "username": "admin@example.com",
        "password": "admin123"
    }
    res_form = requests.post(f"{BASE_URL}/api/auth/token", data=form_data)
    print("\n4. Form Login (Swagger Authorize modal simulation):", res_form.status_code)
    assert res_form.status_code == 200
    print("   Form login succeeded with token!")

    print("\n" + "=" * 60)
    print("ALL AUTH TESTS (LOGIN, PROFILE, ADMIN) PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_auth_flow()
