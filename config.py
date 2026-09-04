import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

IMG_SIZE = 224
IMG_CHANNELS = 3

MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "corn_disease_model.keras"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "20"))

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
