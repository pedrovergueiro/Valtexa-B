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
import re
import shutil
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from app.schemas import PredictionResponse

logger = logging.getLogger(__name__)

PENDING = "pending"
LABELED = "labeled"

IMAGE_SUFFIXES = (".jpg", ".png", ".webp")

ITEM_ID = re.compile(r"^[0-9a-f]{32}$")
"""Formato do id gerado por :meth:`ReviewQueue.enqueue` (``uuid4().hex``).

Validado com rigor porque o id vem da URL e vira caminho de arquivo: sem isto,
um ``../`` no lugar do id leria qualquer arquivo do servidor.
"""

CLASS_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
"""Nome de classe aceito do agrônomo.

Deliberadamente **não** restrito ao catálogo: descobrir uma doença que ainda não
existe nele é justamente o objetivo da fila. Mas o nome vira diretório, então o
formato é fechado — sem barra, sem ponto, sem espaço.
"""


class UnknownReviewItemError(LookupError):
    """O id pedido não está na fila de pendentes."""


class InvalidClassNameError(ValueError):
    """O rótulo informado não serve como nome de pasta."""


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

    def pending(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Os itens à espera de diagnóstico, do mais recente para o mais antigo.

        É o que a interface de revisão lista. Cada item traz o palpite do modelo
        e a pontuação de novidade — o agrônomo decide melhor sabendo o que a
        máquina achou e o quanto ela estranhou.
        """
        if self._directory is None:
            return []

        itens = sorted(self._all_metadata(), key=lambda m: m.get("captured_at", ""), reverse=True)
        return itens[offset : offset + limit]

    def stats(self) -> dict:
        """Quantos itens esperam revisão, e desde quando."""
        if self._directory is None:
            return {"enabled": False, "pending": 0, "labeled": 0, "oldest_pending": None}

        metadados = self._all_metadata()
        capturas = sorted(m.get("captured_at") for m in metadados if m.get("captured_at"))
        rotuladas = 0
        labeled_dir = self._directory / LABELED
        if labeled_dir.is_dir():
            rotuladas = sum(
                1
                for classe in labeled_dir.iterdir()
                if classe.is_dir()
                for arquivo in classe.iterdir()
                if arquivo.suffix.lower() in IMAGE_SUFFIXES
            )

        return {
            "enabled": True,
            "pending": len(metadados),
            "labeled": rotuladas,
            "oldest_pending": capturas[0] if capturas else None,
        }

    def image_path(self, item_id: str) -> Path:
        """Caminho da imagem de um item pendente.

        Raises:
            UnknownReviewItemError: se o id não existir na fila.
        """
        pasta, _ = self._locate(item_id)
        for suffix in IMAGE_SUFFIXES:
            caminho = pasta / f"{item_id}{suffix}"
            if caminho.is_file():
                return caminho
        raise UnknownReviewItemError(f"Item {item_id} não tem imagem arquivada.")

    def label(self, item_id: str, class_name: str, reviewed_by: str | None = None) -> dict:
        """Aplica o diagnóstico do agrônomo e move o item para ``labeled/``.

        A partir daí o ``prepare_dataset`` consome o diretório como qualquer
        outro dataset, e a imagem entra no próximo treino.

        Raises:
            UnknownReviewItemError: se o id não existir na fila.
            InvalidClassNameError: se o rótulo não servir como nome de pasta.
        """
        if not CLASS_NAME.match(class_name):
            raise InvalidClassNameError(
                f"Rótulo inválido: {class_name!r}. Use letras, números e _, começando por letra."
            )

        pasta, metadados = self._locate(item_id)
        imagem = self.image_path(item_id)

        destino = self._directory / LABELED / class_name
        destino.mkdir(parents=True, exist_ok=True)
        shutil.move(str(imagem), str(destino / imagem.name))

        metadados["reviewed_label"] = class_name
        metadados["reviewed_by"] = reviewed_by
        metadados["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        # O JSON acompanha a imagem: é o rastro de quem decidiu o quê, e o que o
        # modelo tinha respondido antes de ser corrigido.
        (destino / f"{item_id}.json").write_text(
            json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pasta / f"{item_id}.json").unlink(missing_ok=True)

        logger.info(
            "Revisado: %s -> %s (o modelo havia dito %s) por %s",
            item_id,
            class_name,
            metadados.get("model_guess"),
            reviewed_by or "não informado",
        )
        return metadados

    def _all_metadata(self) -> list[dict]:
        """Lê o JSON de cada item pendente, ignorando os ilegíveis."""
        if self._directory is None:
            return []

        encontrados = []
        for caminho in (self._directory / PENDING).glob("*/*.json"):
            try:
                encontrados.append(json.loads(caminho.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                # Um item corrompido não pode derrubar a listagem inteira.
                logger.error("Metadados ilegíveis em %s: %s", caminho, exc)
        return encontrados

    def _locate(self, item_id: str) -> tuple[Path, dict]:
        """Pasta e metadados de um item pendente."""
        if self._directory is None or not ITEM_ID.match(item_id):
            raise UnknownReviewItemError(f"Item {item_id!r} não encontrado.")

        for caminho in (self._directory / PENDING).glob(f"*/{item_id}.json"):
            try:
                return caminho.parent, json.loads(caminho.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise UnknownReviewItemError(f"Metadados ilegíveis de {item_id}: {exc}") from exc

        raise UnknownReviewItemError(f"Item {item_id!r} não encontrado.")


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
