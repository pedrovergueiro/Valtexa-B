"""Fila de revisão: acumula o que o modelo não soube explicar.

É a engrenagem do ciclo de dados próprio. Toda imagem que a detecção de novidade
marca cai aqui com o palpite do modelo ao lado. O agrônomo revisa, move o arquivo
para a pasta da classe correta, e o ``prepare_dataset`` consome esse diretório
como qualquer outro dataset — fechando o ciclo sem código extra:

.. code-block:: text

    /predict  →  review/pending/<data>/<id>.jpg + <id>.json
                                    ↓  agrônomo diagnostica e move
                 review/labeled/<Classe>/<id>.jpg
                                    ↓  python -m training.prepare_dataset
                 data/<modelo>/<Classe>/
                                    ↓  python -m training.train
                 modelo novo, agora com dado da lavoura de verdade

Fica **desligada por padrão**: só grava se ``REVIEW_DIR`` estiver definido, já que
guardar imagem de cliente em disco é decisão de quem opera, não default nosso.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from app.schemas import PredictionResponse

logger = logging.getLogger(__name__)

PENDING = "pending"
LABELED = "labeled"


class ReviewQueue:
    """Grava em disco as imagens que precisam de olho humano."""

    def __init__(self, directory: Path | None) -> None:
        self._directory = directory

    @property
    def enabled(self) -> bool:
        return self._directory is not None

    @property
    def pending_dir(self) -> Path | None:
        return self._directory / PENDING if self._directory else None

    def enqueue(
        self, image_bytes: bytes, prediction: PredictionResponse, content_type: str
    ) -> str | None:
        """Arquiva uma imagem para revisão e devolve o id gerado.

        Falhas de escrita são registradas em log mas **não** derrubam a previsão:
        perder um item da fila é ruim, devolver 500 para o cliente por causa de
        disco cheio é pior.
        """
        if self._directory is None:
            return None

        item_id = uuid.uuid4().hex
        folder = self._directory / PENDING / date.today().isoformat()

        try:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{item_id}{_suffix_for(content_type)}").write_bytes(image_bytes)
            (folder / f"{item_id}.json").write_text(
                json.dumps(_metadata(item_id, prediction), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Não foi possível arquivar %s para revisão: %s", prediction.filename, exc)
            return None

        logger.info(
            "Enfileirado para revisão: %s (novidade %.2f, palpite %s) -> %s",
            prediction.filename,
            prediction.novelty_score or 0.0,
            prediction.predicted_class,
            item_id,
        )
        return item_id


def _metadata(item_id: str, prediction: PredictionResponse) -> dict:
    """O que o agrônomo precisa ver para decidir, e o treino para rastrear."""
    return {
        "id": item_id,
        "original_filename": prediction.filename,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model_key": prediction.model_key,
        "model_guess": prediction.predicted_class,
        "confidence": prediction.confidence,
        "novelty_score": prediction.novelty_score,
        "probabilities": {item.class_name: item.probability for item in prediction.probabilities},
        "reviewed_label": None,
        "reviewed_by": None,
    }


def _suffix_for(content_type: str) -> str:
    return {
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(content_type, ".jpg")
