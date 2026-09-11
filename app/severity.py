"""Severidade: quanto do quadro é lesão.

Um modelo de segmentação (``models/leaf_severity.keras``, treinado por
:mod:`training.train_severity`) marca, pixel a pixel, o que é lesão. A severidade
é a fração do quadro marcada, em ``[0, 1]``.

É **área lesionada no quadro**, não da folha: aproxima a fração da folha quando
a folha preenche o quadro, que é a premissa dos modelos foliares ao lado dos
quais ela roda. Numa foto com solo ou céu, o número sai menor do que a
severidade real.

Opcional como a novidade: sem os pesos, a API sobe do mesmo jeito e
``severity`` vem nulo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings
from app.inference.preprocess import preprocess_batch

logger = logging.getLogger(__name__)

WEIGHTS_NAME = "leaf_severity.keras"
LESION_THRESHOLD = 0.5
"""Probabilidade a partir da qual um pixel conta como lesão."""


def lesion_fraction(probabilities: Any, threshold: float = LESION_THRESHOLD) -> list[float]:
    """Fração de pixels acima do limiar, por imagem de um lote ``(N, H, W[, 1])``."""
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim == 4:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Esperado lote (N, H, W[, 1]); recebido {array.shape}.")
    return [float(value) for value in (array > threshold).mean(axis=(1, 2))]


class SeverityModel:
    """O modelo de segmentação, carregado uma vez por processo."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.models_dir / WEIGHTS_NAME
        self._model: Any = None
        self._img_size = 224

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Carrega os pesos, se existirem. Falhar aqui nunca derruba a API."""
        if not self.path.exists():
            logger.info(
                "Sem modelo de severidade em %s — 'severity' virá nulo. Treine com "
                "'python -m training.train_severity'.",
                self.path,
            )
            return
        from app.keras_backend import configure

        configure()
        import keras

        try:
            model = keras.models.load_model(self.path, compile=False)
        except Exception as exc:  # pesos ruins desligam só a severidade, não a API
            logger.error("Falha ao carregar o modelo de severidade %s: %s", self.path, exc)
            return
        self.attach(model)
        logger.info("Modelo de severidade carregado de %s", self.path)

    def attach(self, model: Any) -> None:
        """Anexa um modelo já carregado — é também por aqui que os testes entram."""
        shape = getattr(model, "input_shape", None)
        self._img_size = int(shape[1]) if shape and shape[1] else 224
        self._model = model

    def unload(self) -> None:
        self._model = None

    def estimate(self, images: list[Image.Image]) -> list[float | None]:
        """Severidade de cada imagem, numa passada só pelo modelo."""
        if self._model is None or not images:
            return [None] * len(images)
        from app.keras_backend import to_numpy

        batch = preprocess_batch(images, self._img_size)
        try:
            return lesion_fraction(to_numpy(self._model(batch, training=False)))
        except (ValueError, RuntimeError) as exc:
            # Severidade é um extra: se falhar, a classificação continua valendo.
            logger.error("Falha ao estimar severidade: %s", exc)
            return [None] * len(images)


severity_model = SeverityModel()
"""Instância única, carregada no ``lifespan`` da aplicação."""
