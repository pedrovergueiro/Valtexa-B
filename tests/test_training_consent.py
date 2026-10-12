"""Consentimento do cliente para o treino (brief, seção 32), do lado da IA.

O SaaS manda ``X-Training-Consent`` em toda inferência. Estes testes seguram a
leitura do cabeçalho: só um SIM claro libera a fila de revisão; ausente vale a
decisão de quem opera o servidor (``REVIEW_DIR``).
"""

from __future__ import annotations

import pytest

from app.review import consent_allows


@pytest.mark.parametrize("valor", ["1", "true", "SIM", " sim ", "yes"])
def test_sim_claro_libera(valor: str) -> None:
    assert consent_allows(valor) is True


@pytest.mark.parametrize("valor", ["0", "false", "nao", "não", "no", "", "talvez", "2"])
def test_qualquer_outra_coisa_e_nao(valor: str) -> None:
    assert consent_allows(valor) is False


def test_sem_cabecalho_vale_o_operador() -> None:
    assert consent_allows(None) is True
