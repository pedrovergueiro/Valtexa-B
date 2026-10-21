"""Fixtures compartilhadas pelos testes.

A suíte roda sem TensorFlow instalado: as rotas são exercitadas com um dublê do
modelo Keras, o que mantém os testes rápidos e focados na camada HTTP, no
catálogo de modelos e no pré-processamento.
"""

from __future__ import annotations

import io
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app
from app.registry import get_spec, registry

DEFAULT_MODEL = "leaf-foliar"
EMBEDDING_DIM = 16
"""Dimensão do embedding falso — pequena de propósito, os testes não precisam
das 128 do modelo real."""


class FakeKerasModel:
    """Dublê com a mesma superfície que :class:`CornDiseaseModel` consome.

    Expõe ``input_shape``/``output_shape`` e é chamável, e registra os lotes
    recebidos para que os testes possam afirmar que um lote de N imagens gera
    uma única passada pelo modelo.
    """

    def __init__(
        self,
        num_classes: int = 4,
        img_size: int = 224,
        confidence: float = 1.0,
        embedding: np.ndarray | None = None,
    ) -> None:
        self.input_shape = (None, img_size, img_size, 3)
        self.output_shape = (None, num_classes)
        self.confidence = confidence
        self.embedding = embedding
        self.batch_shapes: list[tuple[int, ...]] = []

    def __call__(self, batch: np.ndarray, training: bool = False) -> np.ndarray:
        self.batch_shapes.append(batch.shape)
        # Massa na última classe, para o resultado esperado ser determinístico.
        num_classes = self.output_shape[1]
        row = np.full(num_classes, (1.0 - self.confidence) / (num_classes - 1), dtype=np.float32)
        row[-1] = self.confidence
        return np.tile(row, (batch.shape[0], 1))

    def embed(self, batch: np.ndarray) -> np.ndarray:
        """Assinatura visual falsa, fixa por instância.

        Presente porque ``CornDiseaseModel`` usa ``.embed`` como ponto de
        extensão quando o objeto anexado não é um modelo Keras de verdade.
        """
        vector = self.embedding if self.embedding is not None else np.zeros(EMBEDDING_DIM)
        return np.tile(np.asarray(vector, dtype=np.float64), (batch.shape[0], 1))


def make_image_bytes(
    size: tuple[int, int] = (320, 240),
    image_format: str = "JPEG",
    color: tuple[int, int, int] = (90, 140, 60),
    exif: Image.Exif | None = None,
) -> bytes:
    """Gera os bytes de uma imagem sintética válida."""
    buffer = io.BytesIO()
    image = Image.new("RGB", size, color)
    if exif is None:
        image.save(buffer, format=image_format)
    else:
        image.save(buffer, format=image_format, exif=exif)
    return buffer.getvalue()


def attach_fake(model_key: str, confidence: float = 1.0) -> FakeKerasModel:
    """Carrega um dublê no modelo indicado do catálogo."""
    spec = get_spec(model_key)
    double = FakeKerasModel(num_classes=spec.num_classes, confidence=confidence)
    registry.get(model_key).attach(double, spec.class_names)
    return double


@pytest.fixture
def image_bytes() -> bytes:
    return make_image_bytes()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Cliente HTTP sem executar o ``lifespan``.

    O carregamento dos pesos é controlado explicitamente pelas fixtures, para
    que cada teste declare quais modelos estão disponíveis.
    """
    registry.unload_all()
    yield TestClient(app)
    registry.unload_all()


@pytest.fixture
def fake_model(client: TestClient) -> FakeKerasModel:
    """Deixa o modelo padrão carregado durante o teste."""
    return attach_fake(DEFAULT_MODEL)


@pytest.fixture(autouse=True)
def _models_dir_isolado(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Cada teste começa com uma pasta de modelos VAZIA.

    O ``models/`` do repositório tem os pesos e as calibrações de produção (Git LFS); sem
    isto, um teste que espera "sem calibração" passava ou falhava conforme o que estivesse
    na pasta da máquina. Quem precisa de arquivos aponta ``settings.models_dir`` para o
    próprio ``tmp_path`` — esse ``monkeypatch`` roda depois e vale.
    """
    monkeypatch.setattr(settings, "models_dir", tmp_path_factory.mktemp("models_vazio"))


@pytest.fixture(autouse=True)
def _reset_confidence_threshold() -> Iterator[None]:
    """Impede que um teste de abstenção vaze o limiar para os demais."""
    original = settings.min_confidence
    yield
    settings.min_confidence = original
