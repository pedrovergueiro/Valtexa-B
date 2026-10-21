"""O ``merge_priors`` do dNATY medido contra o uso que o brief 7.1 sugere (rodada 4).

Ele funde **estatísticas de operadores de mutação** (``EpisodicMemory.to_prior()``): cada
prior é centrado na média e escalado ao máximo, e a média dos normalizados é o consenso. Os
testes abaixo mostram, com o pacote instalado, por que isso **não** serve para fundir
prevalência de doença entre Rovers — e o que ele faz bem. Conclusão em ``docs/FEDERADO.md``.
"""

from __future__ import annotations

import pytest

memory = pytest.importorskip("dnaty.core.memory")


def test_apaga_o_nivel_da_prevalencia() -> None:
    """2% e 20% de ferrugem viram o mesmo prior: a normalização joga fora o nível."""
    baixa = memory.merge_priors([{"scores": {"Rust": 0.02, "Healthy": 0.98}}])
    alta = memory.merge_priors([{"scores": {"Rust": 0.20, "Healthy": 0.80}}])
    assert baixa["scores"] == pytest.approx(alta["scores"])
    assert baixa["scores"] == pytest.approx({"Rust": -1.0, "Healthy": 1.0})


def test_ignora_o_tamanho_da_amostra() -> None:
    """10 plantas e 10 mil plantas pesam igual (o ``n_experiences`` só é somado)."""
    pequeno = {"scores": {"Rust": 0.5, "Healthy": 0.5 - 0.4}, "n_experiences": 10}
    grande = {"scores": {"Rust": 0.05, "Healthy": 0.95}, "n_experiences": 10_000}
    fundido = memory.merge_priors([pequeno, grande])
    # o pequeno (Rust na frente) e o grande (Healthy na frente) se anulam: empate
    assert fundido["scores"]["Rust"] == pytest.approx(0.0, abs=1e-9)
    assert fundido["n_experiences"] == 10_010


def test_serve_para_o_que_foi_feito() -> None:
    """Operadores de mutação: o que ajudou em todos os dispositivos fica na frente."""
    a = {"scores": {"alarga": 0.9, "aprofunda": 0.1, "poda": 0.5}}
    b = {"scores": {"alarga": 0.8, "aprofunda": 0.3, "poda": 0.2}}
    fundido = memory.merge_priors([a, b])
    assert max(fundido["scores"], key=fundido["scores"].get) == "alarga"
    assert fundido["merged_from"] == 2
