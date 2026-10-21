"""Levantamento aéreo importado → evidência por planta no canal do drone (brief 7.2, etapa 1).

O drone voa e o processamento do voo (fora daqui) entrega **regiões suspeitas** — anomalia de
vigor, falha, dossel diferente — e a **área coberta**. Sem a plataforma comandar o voo
(``VALTEXA-SAAS/docs/historico/RODADA4.md``, item 5), isso chega como arquivo. Aqui vira o
que o gêmeo já sabe usar:
a razão de verossimilhança (LLR) por planta, pelo canal da fonte (:data:`.epidemia.CANAIS`):

* planta com GPS dentro da cobertura **e** de uma região suspeita → sinalizada:
  ``LLR = ln(s / a)``;
* dentro da cobertura e fora de toda região suspeita → vista e não sinalizada:
  ``LLR = ln((1 - s) / (1 - a))`` — o drone também diz onde **não** viu nada;
* fora da cobertura ou sem GPS → sem evidência (não vista não é "sadia").

``s`` e ``a`` são os do canal — para o drone, **suposição de projeto** (0,45 / 0,06), até a
calibração com plantas rotuladas. Uma anomalia de vigor não é específica de doença: tratar a
região como "sinal" da doença pedida é exatamente essa suposição, e o resumo diz isso.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from app.gemeo.epidemia import Canal


def dentro(poligono: Sequence[Sequence[float]], lon: float, lat: float) -> bool:
    """Paridade de raios (borda conta como dentro); ``poligono`` em ``[[lon, lat], ...]``."""
    n = len(poligono)
    dentro_ = False
    for i in range(n):
        x1, y1 = poligono[i][0], poligono[i][1]
        x2, y2 = poligono[(i + 1) % n][0], poligono[(i + 1) % n][1]
        if min(y1, y2) <= lat <= max(y1, y2) and min(x1, x2) <= lon <= max(x1, x2):
            cruz = (x2 - x1) * (lat - y1) - (y2 - y1) * (lon - x1)
            if abs(cruz) < 1e-15:
                return True
        if (y1 > lat) != (y2 > lat):
            x = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if x > lon:
                dentro_ = not dentro_
    return dentro_


def evidencia(plant_ids: Sequence[str], lat: np.ndarray, lon: np.ndarray,
              cobertura: Sequence[Sequence[float]],
              anomalias: Sequence[Sequence[Sequence[float]]], canal: Canal
              ) -> tuple[dict[str, float], dict[str, int]]:
    """``({plant_id: LLR}, contagem)``."""
    llr_sinal = math.log(canal.sensibilidade / canal.alarme_falso)
    llr_nada = math.log((1.0 - canal.sensibilidade) / (1.0 - canal.alarme_falso))
    saida: dict[str, float] = {}
    conta = {"cobertas": 0, "sinalizadas": 0, "sem_gps": 0, "fora_da_cobertura": 0}
    for pid, la, lo in zip(plant_ids, lat, lon, strict=True):
        if not (np.isfinite(la) and np.isfinite(lo)):
            conta["sem_gps"] += 1
            continue
        if not dentro(cobertura, float(lo), float(la)):
            conta["fora_da_cobertura"] += 1
            continue
        conta["cobertas"] += 1
        if any(dentro(a, float(lo), float(la)) for a in anomalias):
            conta["sinalizadas"] += 1
            saida[pid] = round(llr_sinal, 6)
        else:
            saida[pid] = round(llr_nada, 6)
    return saida, conta
