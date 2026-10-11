from pydantic import BaseModel
from typing import Optional, Union, List

# =========================
# AUTHENTICATION SCHEMAS
# =========================

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: Optional[str] = "user"

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Sahil",
                "email": "sahil@example.com",
                "password": "mypassword123",
                "role": "user"
            }
        }

class RegisterResponse(BaseModel):
    message: str
    user_id: int
    role: str

class LoginRequest(BaseModel):
    email: str
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "email": "admin@example.com",
                "password": "admin123"
            }
        }

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    role: str
    user_id: Optional[int] = None

class UserProfile(BaseModel):
    id: int
    name: str
    email: str
    role: str

class ProfileResponse(BaseModel):
    message: str
    user: UserProfile

class AdminResponse(BaseModel):
    message: str
    admin: UserProfile


# =========================
# RAG & CHAT SCHEMAS
# =========================

class RAGQuestionRequest(BaseModel):
    document_id: str
    question: str

class ChatRequest(BaseModel):
    message: str


# =========================
# MACHINE LEARNING SCHEMAS
# =========================

class DiseaseInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

class DynamicDiseaseRequest(BaseModel):
    symptoms: Union[str, List[str]]
    age: Optional[float] = None
    gender: Optional[str] = None
    duration: Optional[str] = None
    medical_history: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "symptoms": "Fever, headache, and mild cough",
                "age": 23,
                "gender": "male",
                "duration": "2 days",
                "medical_history": "No known medical conditions"
            }
        }