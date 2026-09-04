import logging
import time

import numpy as np

from app.config import CLASS_NAMES, MODEL_PATH
from app.inference.preprocess import load_image_from_bytes, preprocess_image
from app.schemas import ClassProbability, PredictionResponse

logger = logging.getLogger(__name__)


class CornDiseaseModel:
    def __init__(self) -> None:
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            logger.warning(
                "Nenhum modelo encontrado em %s — rode training/train.py antes de servir previsões.",
                MODEL_PATH,
            )
            return

        import tensorflow as tf

        self._model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Modelo carregado de %s", MODEL_PATH)

    def predict(self, filename: str, image_bytes: bytes) -> PredictionResponse:
        if self._model is None:
            raise RuntimeError("Modelo não carregado. Verifique MODEL_PATH.")

        start = time.perf_counter()
        image = load_image_from_bytes(image_bytes)
        batch = preprocess_image(image)
        raw_probs = self._model.predict(batch, verbose=0)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        probs = [float(p) for p in raw_probs]
        best_idx = int(np.argmax(probs))

        return PredictionResponse(
            filename=filename,
            predicted_class=CLASS_NAMES[best_idx],
            confidence=probs[best_idx],
            probabilities=[
                ClassProbability(class_name=name, probability=prob)
                for name, prob in zip(CLASS_NAMES, probs)
            ],
            inference_ms=elapsed_ms,
        )


corn_disease_model = CornDiseaseModel()
