"""Token de serviço SaaS→IA: com SERVICE_TOKEN, nenhuma rota de dado fica aberta."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.service_auth import token_valido
from tests.conftest import make_image_bytes

TOKEN = "segredo-de-teste-123"


@pytest.fixture
def com_token() -> Iterator[None]:
    original = settings.service_token
    settings.service_token = TOKEN
    yield
    settings.service_token = original


def test_sem_token_configurado_tudo_segue_aberto(client: TestClient) -> None:
    assert client.get("/models").status_code == 200


@pytest.mark.usefixtures("com_token")
class TestComToken:
    def test_rota_de_dado_sem_token_e_401(self, client: TestClient) -> None:
        for rota in ("/models", "/review", "/fields"):
            resposta = client.get(rota)
            assert resposta.status_code == 401, rota
            assert resposta.headers["WWW-Authenticate"] == "Bearer"

    def test_predict_sem_token_nem_chega_a_ler_a_imagem(self, client: TestClient) -> None:
        resposta = client.post(
            "/predict", files={"file": ("a.jpg", make_image_bytes(), "image/jpeg")}
        )
        assert resposta.status_code == 401

    def test_token_errado_e_401(self, client: TestClient) -> None:
        resposta = client.get("/models", headers={"Authorization": "Bearer outro"})
        assert resposta.status_code == 401

    def test_token_certo_passa(self, client: TestClient) -> None:
        resposta = client.get("/models", headers={"Authorization": f"Bearer {TOKEN}"})
        assert resposta.status_code == 200

    def test_health_e_documentacao_continuam_abertos(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize(
    ("cabecalho", "esperado"),
    [
        (f"Bearer {TOKEN}", True),
        (f"bearer {TOKEN}", True),
        (f"Bearer {TOKEN}x", False),
        (TOKEN, False),
        (None, False),
        ("Basic abc", False),
    ],
)
def test_leitura_do_cabecalho(cabecalho: str | None, esperado: bool) -> None:
    assert token_valido(cabecalho, TOKEN) is esperado
