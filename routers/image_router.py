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

from Services.model_loader import load_xray_model


router = APIRouter(
    prefix="/api/image",
    tags=["Image Analysis"]
)


# =========================
# LOAD X-RAY MODEL
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
    "covid19": "Findings indicative of viral pulmonary involvement characteristic of COVID-19 pneumonitis. Clinical correlation and RT-PCR confirmation advised.",
    "normal": "Clear lung fields with no significant focal consolidation, pneumothorax, or acute airspace infiltration detected.",
    "pneumonia": "Airspace consolidation or pulmonary infiltrates detected consistent with pneumonia. Follow-up clinical evaluation recommended."
}

# Supported image file extensions
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif"}


# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_image(image: Image.Image):
    """
    Preprocess image for CNN model:
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
    Classifies X-Ray images into: covid19, normal, or pneumonia.
    Default image_type is 'xray' (medical modality).
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

        # Preprocess
        processed_image = preprocess_image(pil_image)

        # Run Prediction
        model = get_xray_model()
        predictions = model.predict(processed_image, verbose=0)[0]

        # Predicted index and confidence
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])
        predicted_class = XRAY_CLASSES[predicted_index]

        # Probabilities dictionary for all classes
        class_probabilities = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(XRAY_CLASSES, predictions)
        }

        # Check for prediction certainty (reliable if top class >= 0.50)
        is_reliable = bool(confidence >= 0.50)

        # Set diagnosis summary reflecting prediction certainty
        if not is_reliable:
            diagnosis_summary = "Uncertain prediction. Clinical evaluation is recommended."
        else:
            diagnosis_summary = f"The model predicts {predicted_class}. Clinical evaluation is required."

        return {
            "status": "success",
            "image_type": normalized_type,
            "prediction": {
                "class_name": predicted_class,
                "confidence": round(confidence, 4),
                "confidence_percentage": f"{int(round(confidence * 100))}%"
            },
            "class_probabilities": class_probabilities,
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
