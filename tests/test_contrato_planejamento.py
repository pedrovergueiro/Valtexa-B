"""Contrato da leitura de terreno com o planejamento robótico da plataforma (rodada 4, 8.3).

O planejamento (``VALTEXA-SAAS/valtexa/planejamento/robotico.py``) lê o que a rota de terreno
desta API grava no talhão (``TerrainInfo``): ``resolution_ok``, ``heuristic`` e, de cada
objeto, ``kind``, ``outline`` e ``traversable``. Este teste **chama** o detector de obstáculos
(``app.obstacles``) e o esquema da resposta — sem editar nenhum dos arquivos de terreno — e
falha se esses campos mudarem de nome ou de sentido.
"""

from __future__ import annotations

from app.obstacles import detect_obstacles
from app.schemas import ObjectInfo, TerrainInfo
from tests.test_obstacles import CONTORNO, cafezal

CAMPOS_TERRENO = {"resolution_ok", "heuristic", "objects"}
CAMPOS_OBJETO = {"kind", "outline", "traversable", "confidence"}


def test_campos_que_o_planejamento_le_existem() -> None:
    assert set(TerrainInfo.model_fields) >= CAMPOS_TERRENO
    assert set(ObjectInfo.model_fields) >= CAMPOS_OBJETO


def test_mata_e_intransponivel_e_carreador_e_caminho() -> None:
    import numpy as np

    lido = detect_obstacles(cafezal(carreador_px=40), np.array(CONTORNO, dtype=float))
    assert lido.resolucao_suficiente
    por_tipo = {o.tipo: o.atravessavel for o in lido.objetos}
    # o planejamento tira do Rover o trecho dentro do que não é atravessável
    assert por_tipo.get("mata") is False
    assert por_tipo.get("carreador", True) is True
    assert all(len(o.contorno_px) >= 3 for o in lido.objetos)
