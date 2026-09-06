"""Carregamento dos pesos e execução das previsões de um modelo do catálogo.

O TensorFlow só é importado dentro de :meth:`CornDiseaseModel.load`, o que
mantém o *startup* da API rápido e permite testar as rotas sem ter o TF
instalado no ambiente.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from app.config import DEFAULT_IMG_SIZE, settings
from app.inference.novelty import NoveltyReference
from app.inference.preprocess import load_image_from_bytes, preprocess_batch
from app.schemas import ClassProbability, PredictionResponse

if TYPE_CHECKING:  # evita o ciclo de import app.model <-> app.registry
    from app.registry import ModelSpec

logger = logging.getLogger(__name__)

EMBEDDING_LAYER = "embedding"
"""Camada de onde sai a assinatura visual. Espelha ``training.model_def``, que
não é importado aqui para manter a API livre do TensorFlow."""

Upload = tuple[str, bytes]
"""Par ``(nome do arquivo, bytes da imagem)`` recebido pelas rotas."""


class ModelNotLoadedError(RuntimeError):
    """Uma previsão foi pedida sem que houvesse pesos carregados."""


class ModelConfigurationError(RuntimeError):
    """Os pesos e os rótulos carregados são incompatíveis entre si."""


class CornDiseaseModel:
    """Encapsula um modelo Keras e a tradução de índices para nomes de classe."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model: Any = None
        self._forward: Any = None
        self._novelty: NoveltyReference | None = None
        self._class_names: tuple[str, ...] = tuple(spec.class_names)
        self._img_size: int = spec.img_size

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def weights_path(self) -> Path:
        return self.spec.weights_path

    @property
    def labels_path(self) -> Path:
        return self.spec.labels_path

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def img_size(self) -> int:
        return self._img_size

    def load(self) -> None:
        """Carrega os pesos e os rótulos do disco.

        Falhas são registradas em log sem derrubar o processo: a API sobe assim
        mesmo e responde ``503`` apenas para este modelo, o que permite colocar
        um modelo em produção enquanto os outros ainda estão em coleta.
        """
        path = self.weights_path
        if not path.exists():
            logger.warning(
                "[%s] sem pesos em %s — este modelo vai responder 503. "
                "Treine com 'python -m training.train --model %s'.",
                self.key,
                path,
                self.key,
            )
            return

        import tensorflow as tf  # import tardio: custa alguns segundos no startup

        logger.info("[%s] carregando de %s ...", self.key, path)
        try:
            self.attach(tf.keras.models.load_model(path), self._read_class_names())
        except (ModelConfigurationError, OSError, ValueError) as exc:
            self._model = None
            logger.error("[%s] falha ao carregar de %s: %s", self.key, path, exc)
            return

        logger.info(
            "[%s] pronto — entrada %dx%d, %d classes: %s",
            self.key,
            self._img_size,
            self._img_size,
            len(self._class_names),
            ", ".join(self._class_names),
        )

    def attach(self, keras_model: Any, class_names: Sequence[str] | None = None) -> None:
        """Registra um modelo já instanciado.

        É o ponto de entrada usado por :meth:`load` e também pelos testes, que
        injetam um dublê para exercitar as rotas sem depender do TensorFlow.

        Raises:
            ModelConfigurationError: se a quantidade de rótulos não bater com o
                número de saídas do modelo. Mapear índices para nomes errados
                produziria previsões incorretas sem nenhum sinal de erro.
        """
        names = tuple(class_names) if class_names else tuple(self.spec.class_names)
        units = _output_units(keras_model)
        if units is not None and units != len(names):
            raise ModelConfigurationError(
                f"O modelo tem {units} saídas, mas foram informados {len(names)} rótulos "
                f"({', '.join(names)}). Verifique o {self.labels_path.name} ao lado dos pesos."
            )

        size = _square_input_size(keras_model)
        if size is None:
            logger.warning(
                "[%s] não foi possível inferir o tamanho de entrada — usando %d.",
                self.key,
                self.spec.img_size or DEFAULT_IMG_SIZE,
            )

        self._model = keras_model
        self._forward = _build_forward(keras_model)
        self._class_names = names
        self._img_size = size or self.spec.img_size or DEFAULT_IMG_SIZE
        self._novelty = self._read_novelty_reference(names)

    def unload(self) -> None:
        """Descarta o modelo em memória e volta aos rótulos de fallback."""
        self._model = None
        self._forward = None
        self._novelty = None
        self._class_names = tuple(self.spec.class_names)
        self._img_size = self.spec.img_size

    @property
    def detects_novelty(self) -> bool:
        """Se este modelo consegue apontar o que nunca viu."""
        return self._novelty is not None

    def predict(self, filename: str, image_bytes: bytes) -> PredictionResponse:
        """Classifica uma única imagem."""
        return self.predict_many([(filename, image_bytes)])[0]

    def predict_many(self, uploads: Sequence[Upload]) -> list[PredictionResponse]:
        """Classifica várias imagens em uma única passada pelo modelo.

        Raises:
            ModelNotLoadedError: se não houver pesos carregados.
            InvalidImageError: se alguma das imagens não puder ser decodificada.
        """
        if self._model is None:
            raise ModelNotLoadedError(
                f"Modelo {self.key!r} não carregado. Verifique {self.weights_path} e os logs."
            )
        if not uploads:
            return []

        images = [load_image_from_bytes(data, filename) for filename, data in uploads]
        batch = preprocess_batch(images, self._img_size)

        # Chamada direta em vez de ``predict()``: para lotes pequenos, o modo
        # eager evita o overhead por chamada do laço interno do Keras. As
        # probabilidades e os embeddings saem da *mesma* passada — extrair a
        # assinatura visual não custa uma segunda inferência.
        start = perf_counter()
        outputs, embeddings = self._forward(batch)
        elapsed_ms = (perf_counter() - start) * 1000

        probabilities = np.clip(np.asarray(outputs, dtype=np.float64), 0.0, 1.0)
        self._check_output_shape(probabilities, len(uploads))
        novelty = self._score_novelty(embeddings, len(uploads))

        per_image_ms = elapsed_ms / len(uploads)
        return [
            self._to_response(filename, row, score, per_image_ms)
            for (filename, _), row, score in zip(uploads, probabilities, novelty, strict=True)
        ]

    def _score_novelty(self, embeddings: Any, expected_rows: int) -> list[float | None]:
        """Pontua a novidade do lote, ou devolve ``None`` quando indisponível."""
        if self._novelty is None or embeddings is None:
            return [None] * expected_rows
        try:
            scores = self._novelty.score(np.asarray(embeddings, dtype=np.float64))
        except (ValueError, np.linalg.LinAlgError) as exc:
            # Novidade é um extra: se falhar, a previsão principal continua boa.
            logger.error("[%s] falha ao pontuar novidade: %s", self.key, exc)
            return [None] * expected_rows
        return [float(score) for score in scores]

    def _to_response(
        self,
        filename: str,
        probabilities: np.ndarray,
        novelty_score: float | None,
        inference_ms: float,
    ) -> PredictionResponse:
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        return PredictionResponse(
            filename=filename,
            model_key=self.key,
            predicted_class=self._class_names[best_index],
            confidence=confidence,
            abstained=confidence < settings.min_confidence,
            novelty_score=novelty_score,
            needs_review=_is_novel(novelty_score),
            probabilities=[
                ClassProbability(class_name=name, probability=float(probability))
                for name, probability in zip(self._class_names, probabilities, strict=True)
            ],
            inference_ms=inference_ms,
        )

    def _read_novelty_reference(self, names: tuple[str, ...]) -> NoveltyReference | None:
        """Carrega a referência de novidade gravada pelo ``fit_novelty``."""
        path = self.spec.novelty_path
        if not path.exists():
            logger.info(
                "[%s] sem referência de novidade em %s — /predict não vai marcar "
                "imagens para revisão. Gere com 'python -m training.fit_novelty --model %s'.",
                self.key,
                path,
                self.key,
            )
            return None

        try:
            reference = NoveltyReference.load(path)
        except ValueError as exc:
            logger.error("[%s] %s", self.key, exc)
            return None

        if reference.class_names != names:
            logger.error(
                "[%s] referência de novidade calibrada para %s, mas o modelo usa %s — ignorando.",
                self.key,
                ", ".join(reference.class_names),
                ", ".join(names),
            )
            return None

        logger.info(
            "[%s] detecção de novidade ativa (calibrada em %d imagens).",
            self.key,
            reference.fitted_on,
        )
        return reference

    def _check_output_shape(self, probabilities: np.ndarray, expected_rows: int) -> None:
        expected = (expected_rows, len(self._class_names))
        if probabilities.shape != expected:
            raise ModelConfigurationError(
                f"O modelo devolveu saída de shape {probabilities.shape}, esperado {expected}."
            )

    def _read_class_names(self) -> tuple[str, ...]:
        """Lê os rótulos gravados pelo treino, com o catálogo como fallback."""
        path = self.labels_path
        fallback = tuple(self.spec.class_names)

        if not path.exists():
            logger.warning(
                "[%s] rótulos não encontrados em %s — assumindo a ordem do catálogo (%s). "
                "Confirme que ela bate com a ordem usada no treino.",
                self.key,
                path,
                ", ".join(fallback),
            )
            return fallback

        try:
            labels = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "[%s] rótulos inválidos em %s (%s) — usando o catálogo.", self.key, path, exc
            )
            return fallback

        if not _is_label_list(labels):
            logger.error(
                "[%s] %s deveria ser uma lista de strings — usando o catálogo.", self.key, path
            )
            return fallback

        return tuple(labels)


def _is_label_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) for item in value)


def _is_novel(score: float | None) -> bool:
    """O limiar já está embutido no score (1.0); a sensibilidade o desloca."""
    if score is None:
        return False
    return score > settings.novelty_sensitivity


def _build_forward(keras_model: Any) -> Any:
    """Monta o callable ``lote -> (probabilidades, embeddings | None)``.

    Quando o modelo expõe a camada de embedding, devolve um modelo Keras de duas
    saídas: assim a assinatura visual sai de graça, na mesma passada das
    probabilidades. Modelos antigos, sem a camada nomeada, seguem funcionando —
    só não pontuam novidade.
    """
    embed = getattr(keras_model, "embed", None)
    if callable(embed):  # dublê dos testes
        return lambda batch: (keras_model(batch, training=False), embed(batch))

    try:
        embedding_output = keras_model.get_layer(EMBEDDING_LAYER).output
        dual = keras_model.__class__(keras_model.inputs, [keras_model.output, embedding_output])
    except (AttributeError, ValueError, TypeError) as exc:
        logger.warning(
            "Camada %r não encontrada (%s) — detecção de novidade indisponível "
            "para este modelo. Retreine para habilitá-la.",
            EMBEDDING_LAYER,
            exc,
        )
        return lambda batch: (keras_model(batch, training=False), None)

    return lambda batch: tuple(dual(batch, training=False))


def _square_input_size(keras_model: Any) -> int | None:
    """Extrai o lado da entrada quadrada declarada pelo modelo, se houver."""
    shape = getattr(keras_model, "input_shape", None)
    if not isinstance(shape, tuple) or len(shape) != 4:
        return None
    height, width = shape[1], shape[2]
    if height is None or width is None or height != width:
        return None
    return int(height)


def _output_units(keras_model: Any) -> int | None:
    """Extrai o número de saídas do modelo, se declarado."""
    shape = getattr(keras_model, "output_shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[1] is None:
        return None
    return int(shape[1])
