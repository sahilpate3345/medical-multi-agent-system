# 🏥 Medical Multi-Agent System (AI Health Intelligence API)

An intelligent, multi-agent AI healthcare platform powered by FastAPI, modern machine learning models, and LLM reasoning. Provides dynamic disease prediction from clinical symptoms, deep learning chest X-ray diagnosis, intelligent medical document analysis (RAG), and role-based access control.

---

## 🌟 Key Features

### 1. 🧠 Dynamic AI Disease Prediction Engine
- **100% Dynamic Inference**: Analyzes patient symptoms, clinical complaints, duration, age, and medical history without static disease dictionaries.
- **Comprehensive Clinical Reports**: Dynamically outputs predicted condition, confidence score, severity rating (`Mild`, `Moderate`, `High`, `Emergency`), recommended specialists, differential diagnoses, recommended diagnostic tests, and precautions.
- **Dual-Mode ML Support**: Accepts free-form symptom descriptions or traditional 13-feature tabular cardiac parameters.

### 2. 🩻 Deep Learning Medical X-Ray Analysis
- **Convolutional Neural Network**: Classifies chest X-ray images into **COVID-19**, **Normal**, and **Pneumonia**.
- **Uncertainty & Reliability Detection**: Flags low-confidence or borderline predictions to prompt professional radiological re-evaluation.
- **Format Agnostic**: Accepts standard medical image uploads (`.jpg`, `.jpeg`, `.png`, `.webp`).

### 3. 🤖 Medical AI Assistant & RAG
- **Clinical Chat Assistant**: Conversational medical agent powered by Groq and high-performance LLMs.
- **Document Intelligence (RAG)**: Extracts, vectorizes, and answers queries from uploaded medical PDFs and patient lab reports.

### 4. 🔐 Security & Access Control
- **JWT Authentication**: Secure token-based user authentication.
- **Role-Based Access**: Granular user (`user`) and administrative (`admin`) permission controls.
- **Multi-Scheme Hashing**: Compatible with PBKDF2 and Bcrypt.

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/sahilpate3345/medical-multi-agent-system.git
cd medical-multi-agent-system
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.8-27b
SECRET_KEY=your_jwt_secret_key
XRAY_MODEL_ID=
```

### 3. Run the Application
Start the FastAPI server:
```bash
uvicorn main:app --reload
```
Interactive Swagger API documentation will be available at:
👉 **`http://127.0.0.1:8000/docs`**

---

## 📚 API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/auth/register` | Register a new user or administrator |
| `POST` | `/api/auth/login` | Authenticate and obtain JWT access token |
| `GET` | `/api/auth/profile` | Retrieve authenticated user profile |
| `GET` | `/api/auth/admin` | Admin dashboard access (Admin only) |
| `POST` | `/api/ml/disease-predict` | Dynamic symptom analysis & disease prediction |
| `POST` | `/api/image/analyze` | Deep learning chest X-ray image classification |
| `POST` | `/api/chat` | AI Healthcare chat consultation |
| `POST` | `/api/rag/upload` | Upload medical document (PDF) for analysis |
| `POST` | `/api/rag/ask` | Query uploaded medical documents |

---

## 👨‍💻 Author
- **Sahil Pate** - [GitHub Profile](https://github.com/sahilpate3345)
