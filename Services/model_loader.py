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
# LOAD X-RAY MODEL
# =========================

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
