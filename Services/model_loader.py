import os
import gdown

from dotenv import load_dotenv

load_dotenv(override=True)


# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)


# =========================
# MODEL DIRECTORY
# =========================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml_models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================
# X-RAY MODEL
# =========================

XRAY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Xraymodel.h5"
)

XRAY_MODEL_ID = os.getenv(
    "XRAY_MODEL_ID"
)


# =========================
# DOWNLOAD MODEL
# =========================

def download_model(file_id, output_path):

    if os.path.exists(output_path):

        print(
            f"Model already exists: {output_path}"
        )

        return

    if not file_id:

        raise RuntimeError(
            f"Google Drive ID missing for {output_path}"
        )

    print(
        f"Downloading model: {output_path}"
    )

    gdown.download(
        id=file_id,
        output=output_path,
        quiet=False
    )

    if not os.path.exists(output_path):

        raise RuntimeError(
            f"Failed to download model: {output_path}"
        )


# =========================
# LOAD X-RAY & VISION MODELS
# =========================

PRETRAINED_XRAY_MODEL_ID = "yuighj123/image_classification_covid19"
_xray_vision_pipeline = None

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
_clip_zero_shot_pipeline = None


def load_clip_zero_shot_pipeline():
    """
    Loads and caches the open-vocabulary Zero-Shot Vision Classifier (CLIP)
    for dynamic anatomical recognition (Hand, Leg, Eye/Skull, Spine, Pelvis, Chest, Abdomen).
    """
    global _clip_zero_shot_pipeline
    if _clip_zero_shot_pipeline is not None:
        return _clip_zero_shot_pipeline

    print(f"Loading Zero-Shot Multi-Anatomy Vision Pipeline: {CLIP_MODEL_ID}...")
    try:
        from transformers import pipeline
        _clip_zero_shot_pipeline = pipeline(
            "zero-shot-image-classification",
            model=CLIP_MODEL_ID,
            device=-1
        )
        print("Zero-Shot Multi-Anatomy Vision Pipeline loaded successfully!")
        return _clip_zero_shot_pipeline
    except Exception as e:
        print(f"Warning: Failed to load CLIP zero-shot pipeline ({e})")
        return None


def load_xray_vision_pipeline():
    """
    Loads and caches the fine-tuned Medical Vision Transformer pipeline
    specialized in COVID-19, Normal, and Viral Pneumonia Chest X-Rays.
    """
    global _xray_vision_pipeline
    if _xray_vision_pipeline is not None:
        return _xray_vision_pipeline

    print(f"Loading pretrained medical Vision Transformer: {PRETRAINED_XRAY_MODEL_ID}...")
    try:
        from transformers import pipeline
        _xray_vision_pipeline = pipeline(
            "image-classification",
            model=PRETRAINED_XRAY_MODEL_ID,
            device=-1
        )
        print("Pretrained medical Vision Transformer loaded successfully!")
        return _xray_vision_pipeline
    except Exception as e:
        print(f"Warning: Failed to load Vision Transformer ({e}), falling back to Keras model.")
        return None


def load_xray_model():

    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError(
            "TensorFlow is not installed. Please install it using 'pip install tensorflow'"
        )

    print("Checking X-Ray model...")

    download_model(
        XRAY_MODEL_ID,
        XRAY_MODEL_PATH
    )

    print("Loading X-Ray model...")

    model = tf.keras.models.load_model(
        XRAY_MODEL_PATH
    )

    print("X-Ray model loaded successfully!")

    return model
