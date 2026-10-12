"""Métricas da IA (brief, seções 34 e 35): inferência por modelo e HTTP por rota."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import metrics
from app.config import settings
from tests.conftest import DEFAULT_MODEL, FakeKerasModel, make_image_bytes

TOKEN = "segredo-de-metricas-123"


@pytest.fixture
def com_token() -> Iterator[None]:
    original = settings.service_token
    settings.service_token = TOKEN
    yield
    settings.service_token = original


def test_inferencia_conta_imagens_latencia_e_classes(
    client: TestClient, fake_model: FakeKerasModel
) -> None:
    antes = metrics.INFERENCIAS.valor(model=DEFAULT_MODEL)
    lat_antes = metrics.LATENCIA.contagem(model=DEFAULT_MODEL)
    resposta = client.post("/predict", files={"file": ("a.jpg", make_image_bytes(),
                                                       "image/jpeg")})
    assert resposta.status_code == 200, resposta.text
    classe = resposta.json()["predicted_class"]
    assert metrics.INFERENCIAS.valor(model=DEFAULT_MODEL) - antes == 1
    assert metrics.LATENCIA.contagem(model=DEFAULT_MODEL) - lat_antes == 1
    assert metrics.CLASSES.valor(model=DEFAULT_MODEL, label=classe) >= 1
    texto = client.get("/metrics").text
    assert f'valtexa_ia_predictions_total{{model="{DEFAULT_MODEL}",label="{classe}"}}' in texto
    assert f'valtexa_ia_model_loaded{{model="{DEFAULT_MODEL}"}} 1' in texto
    assert 'valtexa_ia_http_requests_total{method="POST",route="/predict",status="200"}' in texto


def test_modelo_sem_pesos_conta_erro_por_tipo(client: TestClient) -> None:
    from PIL import Image

    from app.model import ModelNotLoadedError
    from app.registry import registry

    # pela API, a rota recusa antes de chegar ao modelo: conta no HTTP (503)
    resposta = client.post("/predict", files={"file": ("a.jpg", make_image_bytes(),
                                                       "image/jpeg")})
    assert resposta.status_code == 503
    texto = client.get("/metrics").text
    assert 'valtexa_ia_http_requests_total{method="POST",route="/predict",status="503"}' in texto
    assert f'valtexa_ia_model_loaded{{model="{DEFAULT_MODEL}"}} 0' in texto
    # quem chama o modelo direto (rover, fusão) sem pesos: conta como erro do modelo
    antes = metrics.ERROS.valor(model=DEFAULT_MODEL, error="modelo_ausente")
    with pytest.raises(ModelNotLoadedError):
        registry.get(DEFAULT_MODEL).predict_images(["a.jpg"], [Image.new("RGB", (8, 8))])
    assert metrics.ERROS.valor(model=DEFAULT_MODEL, error="modelo_ausente") - antes == 1


def test_rota_medida_pelo_molde(client: TestClient) -> None:
    client.get("/fields/talhao-que-nao-existe/incidence")
    texto = client.get("/metrics").text
    assert "talhao-que-nao-existe" not in texto
    assert 'route="/fields/{field_id}/incidence"' in texto


@pytest.mark.usefixtures("com_token")
def test_metrics_exige_o_token_de_servico(client: TestClient) -> None:
    assert client.get("/metrics").status_code == 401
    resposta = client.get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/plain")
    # a recusa também é medida (o middleware fica por fora do token)
    assert 'route="nao_mapeada",status="401"' in resposta.text or 'status="401"' in resposta.text


def test_formato_do_histograma() -> None:
    h = metrics.Histograma("x_ms", "x", ("model",), (10, 100))
    for v in (5, 50, 500):
        h.observe(v, model="m")
    texto = h.render()
    assert 'x_ms_bucket{model="m",le="10"} 1' in texto
    assert 'x_ms_bucket{model="m",le="+Inf"} 3' in texto
    assert 'x_ms_count{model="m"} 3' in texto and 'x_ms_sum{model="m"} 555' in texto
    assert metrics.tipo_de_erro(RuntimeError()) == "falha"
