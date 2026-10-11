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

from Services.model_loader import load_xray_model, load_xray_vision_pipeline, load_clip_zero_shot_pipeline
from Services.ML.dynamic_vision_service import analyze_medical_image_dynamically


router = APIRouter(
    prefix="/api/image",
    tags=["Image Analysis"]
)


# Supported image file extensions
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif"}


# =========================
# ANALYZE MEDICAL IMAGE (UNIVERSAL & DYNAMIC)
# =========================

@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(..., description="Uploaded medical image file (.jpg, .jpeg, .png, etc.)"),
    image_type: Optional[str] = Form(None, description="Optional medical modality or anatomical region (e.g. 'hand', 'leg', 'eye', 'chest', 'xray'). If omitted, AI auto-detects dynamically.")
):
    """
    Analyze ANY medical image dynamically (JPG, JPEG, PNG supported).
    Auto-detects anatomical region (Hand, Leg, Eye/Orbit, Skull, Spine, Pelvis, Chest, Abdomen)
    and predicts pathology, clinical severity, specialist, tests, and precautions dynamically.
    Enforces >=90% clinical confidence threshold.
    """
    # Check filename extension if provided
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

        # Dynamic Multi-Anatomy AI Evaluation (Hand, Leg, Eye, Chest, Spine, Pelvis, Abdomen)
        result = analyze_medical_image_dynamically(pil_image, user_modality=image_type)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis error: {str(e)}"
        )
