import os
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException
)

from Services.model_loader import load_xray_model, load_xray_vision_pipeline


router = APIRouter(
    prefix="/api/image",
    tags=["Image Analysis"]
)


# =========================
# LOAD X-RAY MODEL (FALLBACK)
# =========================

xray_model = None


def get_xray_model():
    global xray_model
    if xray_model is None:
        xray_model = load_xray_model()
    return xray_model


# =========================
# CLASS NAMES & SUMMARIES
# =========================

XRAY_CLASSES = [
    "covid19",
    "normal",
    "pneumonia"
]

CLASS_DESCRIPTIONS = {
    "covid19": "Findings indicative of viral pulmonary involvement characteristic of COVID-19 pneumonitis. Clinical correlation, pulse oximetry, and RT-PCR confirmation advised.",
    "normal": "Clear lung fields with no significant focal consolidation, pneumothorax, or acute airspace infiltration detected. Unremarkable thoracic study.",
    "pneumonia": "Airspace consolidation or pulmonary infiltrates detected consistent with pneumonia. Follow-up clinical evaluation, CBC, and sputum cultures recommended."
}

RECOMMENDED_TESTS = {
    "covid19": [
        "SARS-CoV-2 RT-PCR / Rapid Molecular Assay",
        "High-Resolution CT (HRCT) Chest without contrast",
        "Complete Blood Count (CBC) with Differential and Ferritin / D-Dimer",
        "Continuous Pulse Oximetry (SpO2) Monitoring"
    ],
    "pneumonia": [
        "Sputum Gram Stain and Bacterial/Fungal Culture",
        "Complete Blood Count (CBC) with Differential",
        "Serum C-Reactive Protein (CRP) / Procalcitonin",
        "Repeat Chest Radiograph at 4-6 weeks post-treatment"
    ],
    "normal": [
        "Clinical evaluation by an attending physician",
        "12-Lead Electrocardiogram (ECG) if presenting with atypical chest pain"
    ]
}

# Supported image file extensions
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif"}


# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_image(image: Image.Image):
    """
    Preprocess image for CNN model fallback:
    - Handles PNG transparency (RGBA) by compositing over white background
    - Converts grayscale or other formats to RGB
    - Resizes to (200, 200)
    - Normalizes pixel values to [0, 1]
    """
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        background = Image.new("RGB", image.size, (255, 255, 255))
        converted = image.convert("RGBA")
        background.paste(converted, mask=converted.split()[3])
        image = background
    else:
        image = image.convert("RGB")

    image = image.resize((200, 200))
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image


# =========================
# ANALYZE MEDICAL IMAGE
# =========================

@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(..., description="Uploaded medical image file (.jpg, .jpeg, .png, etc.)"),
    image_type: Optional[str] = Form("xray", description="Type of medical image analysis (e.g. 'xray'). Note: Describes medical modality, not file extension.")
):
    """
    Analyze medical image (JPG, JPEG, PNG supported).
    Utilizes a pretrained Medical Vision Transformer specialized in Chest Radiographs
    to classify into: covid19, normal, or pneumonia with genuine clinical confidence.
    """
    # 1. Flexible image_type handling: defaults to 'xray'
    normalized_type = "xray"
    if image_type:
        clean_type = image_type.lower().strip()
        if clean_type in ["xray", "x-ray", "chest", "chest xray", "chest x-ray", "chest-xray"]:
            normalized_type = "xray"
        elif clean_type in [".jpg", "jpg", ".jpeg", "jpeg", ".png", "png", ".webp", "webp"]:
            normalized_type = "xray"
        else:
            normalized_type = "xray"

    # 2. Check filename extension if provided
    filename = image.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    try:
        # Read file contents
        contents = await image.read()

        if not contents or len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image file is empty."
            )

        # Open and validate image format with Pillow
        try:
            pil_image = Image.open(BytesIO(contents))
            # Verify it is a valid decodable image
            pil_image.verify()
            # Re-open after verify() because verify() clears the image stream
            pil_image = Image.open(BytesIO(contents))
        except Exception as img_err:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode image. Please ensure it is a valid JPG or PNG image file: {str(img_err)}"
            )

        # 3. Primary Medical Vision Model (Pretrained Vision Transformer)
        pipe = load_xray_vision_pipeline()

        if pipe is not None:
            rgb_image = pil_image.convert("RGB")
            hf_results = pipe(rgb_image)

            norm_scores = {"covid19": 0.0, "normal": 0.0, "pneumonia": 0.0}
            for item in hf_results:
                lbl = item.get("label", "").lower().strip()
                score = float(item.get("score", 0.0))
                if "covid" in lbl:
                    norm_scores["covid19"] = score
                elif "normal" in lbl:
                    norm_scores["normal"] = score
                elif "pneumonia" in lbl:
                    norm_scores["pneumonia"] = score

            total_score = sum(norm_scores.values()) or 1.0
            class_probabilities = {k: round(v / total_score, 4) for k, v in norm_scores.items()}
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            confidence = class_probabilities[predicted_class]

        else:
            # Fallback to Keras model if Vision Transformer is unavailable
            processed_image = preprocess_image(pil_image)
            model = get_xray_model()
            predictions = model.predict(processed_image, verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index])
            predicted_class = XRAY_CLASSES[predicted_index]
            class_probabilities = {
                cls_name: round(float(prob), 4)
                for cls_name, prob in zip(XRAY_CLASSES, predictions)
            }

        # 4. Clinical Confidence Thresholding (>= 90% target)
        meets_90pct = bool(confidence >= 0.90)
        is_reliable = bool(confidence >= 0.70)
        confidence_pct_int = int(round(confidence * 100))

        if not is_reliable:
            diagnosis_summary = "Uncertain prediction. Model confidence is below reliable diagnostic threshold. Clinical radiological evaluation is recommended."
            confidence_tier = "Low Diagnostic Confidence (< 70%)"
        elif meets_90pct:
            confidence_tier = "High Clinical Confidence (>= 90%)"
            diagnosis_summary = CLASS_DESCRIPTIONS.get(
                predicted_class,
                f"The model predicts {predicted_class}. Clinical evaluation is required."
            )
        else:
            confidence_tier = "Moderate Clinical Confidence (70% - 89%)"
            diagnosis_summary = CLASS_DESCRIPTIONS.get(
                predicted_class,
                f"The model predicts {predicted_class}. Confirmatory tests are recommended."
            )

        recommended_tests = RECOMMENDED_TESTS.get(predicted_class, [])

        return {
            "status": "success",
            "image_type": normalized_type,
            "prediction": {
                "class_name": predicted_class,
                "confidence": round(confidence, 4),
                "confidence_percentage": f"{confidence_pct_int}%"
            },
            "class_probabilities": class_probabilities,
            "meets_90pct_threshold": meets_90pct,
            "confidence_tier": confidence_tier,
            "recommended_clinical_tests": recommended_tests,
            "diagnosis_summary": diagnosis_summary,
            "is_reliable": is_reliable,
            "disclaimer": "AI prediction only; not a confirmed medical diagnosis."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis error: {str(e)}"
        )
