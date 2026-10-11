from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    admin_only
)

from database import get_db
from models import User
from schemas import (
    RegisterRequest,
    RegisterResponse,
    LoginRequest,
    LoginResponse,
    ProfileResponse,
    AdminResponse
)

router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"]
)


# =========================
# USER REGISTRATION
# =========================

@router.post(
    "/register",
    response_model=RegisterResponse,
    summary="Register",
    description="Register a new user account with name, email, password, and optional role ('user' or 'admin')."
)
def register(
    data: RegisterRequest,
    db: Session = Depends(get_db)
):
    existing_user = (
        db.query(User)
        .filter(User.email == data.email.strip())
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    assigned_role = (data.role or "user").strip().lower()
    if assigned_role not in ["admin", "user"]:
        assigned_role = "user"

    user = User(
        name=data.name.strip(),
        email=data.email.strip(),
        password_hash=hash_password(data.password),
        role=assigned_role
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "message": "Registration successful",
        "user_id": user.id,
        "role": user.role
    }


# =========================
# USER & ADMIN LOGIN (JSON)
# =========================

@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login",
    description="Login with email and password to receive a JWT access token."
)
def login(
    data: LoginRequest,
    db: Session = Depends(get_db)
):
    email = data.email.strip()
    password = data.password

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_access_token(
        user.id,
        user.role
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "user_id": user.id
    }


# ===============================================
# OAUTH2 TOKEN ENDPOINT (Swagger Authorize modal)
# ===============================================

@router.post(
    "/token",
    response_model=LoginResponse,
    include_in_schema=False
)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    email = form_data.username.strip()
    password = form_data.password

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_access_token(
        user.id,
        user.role
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role,
        "user_id": user.id
    }


# =========================
# USER PROFILE
# =========================

@router.get(
    "/profile",
    response_model=ProfileResponse,
    summary="Profile",
    description="Get the authenticated user's profile details. Requires Bearer Token."
)
def profile(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_id = current_user.get("user_id")
    user = db.query(User).filter(User.id == user_id).first() if user_id else None

    return {
        "message": "You are authenticated",
        "user": {
            "id": user.id if user else user_id,
            "name": user.name if user else "User",
            "email": user.email if user else "",
            "role": user.role if user else current_user.get("role", "user")
        }
    }


# =========================
# ADMIN PANEL
# =========================

@router.get(
    "/admin",
    response_model=AdminResponse,
    summary="Admin Panel",
    description="Access the admin dashboard. Requires an account with role 'admin' and Bearer Token."
)
def admin_panel(
    current_user: dict = Depends(admin_only),
    db: Session = Depends(get_db)
):
    user_id = current_user.get("user_id")
    user = db.query(User).filter(User.id == user_id).first() if user_id else None

    return {
        "message": "Welcome Admin",
        "admin": {
            "id": user.id if user else user_id,
            "name": user.name if user else "Administrator",
            "email": user.email if user else "",
            "role": "admin"
        }
    }