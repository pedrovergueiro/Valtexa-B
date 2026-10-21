"""``training.robustez_captura``: degradação de vídeo, corte de produção do extra e a rodada."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from tests.test_continual import CLASSES, D, _npz, mundo  # noqa: F401 (fixture)
from training import robustez_captura as rc


def test_degradacao_e_deterministica_e_de_video() -> None:
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (900, 1200, 3),
                                                           dtype=np.uint8))
    a, tipo_a = rc.degrada(img, rc._rng("folha.jpg", 0))
    b, tipo_b = rc.degrada(img, rc._rng("folha.jpg", 0))
    c, _ = rc.degrada(img, rc._rng("folha.jpg", 1))
    assert tipo_a == tipo_b and np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a.resize((64, 64))), np.asarray(c.resize((64, 64))))
    assert a.mode == "RGB" and max(a.size) <= rc.LADO_TRABALHO
    assert set(tipo_a.split("+")) <= set(rc.TIPOS)
    vistos = {t for v in range(40) for t in rc.degrada(img, rc._rng("x", v))[1].split("+")}
    assert vistos == set(rc.TIPOS)


def test_corte_de_producao_do_extra(tmp_path) -> None:
    rng = np.random.default_rng(1)
    n = 40
    emb = rng.normal(size=(n, 2, D))
    extra = tmp_path / "bracol.npz"
    np.savez(extra, emb=emb, labels=np.arange(n) % 3, classes=np.array(CLASSES),
             ids=np.arange(1, n + 1), valid=np.ones(n, bool))
    pasta, holdout = tmp_path / "coffee-bracol", tmp_path / "holdout"
    for i in range(1, n + 1):
        classe = CLASSES[(i - 1) % 3]
        (pasta / classe).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(pasta / classe / f"bracol_{i}.jpg")
        if i > 25:  # as 15 últimas: o que a cabeça de produção não treinou
            (holdout / classe).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8)).save(holdout / classe / f"bracol_{i}.jpg")
    principal = tmp_path / "principal.npz"
    copia = emb[:2].copy()  # duas fotos do extra que são quase cópia do principal
    np.savez(principal, emb=np.concatenate([copia, rng.normal(size=(5, 2, D))]))
    saida = tmp_path / "alinhado.npz"
    contagem = rc.alinha_extra(extra, pasta, holdout, principal, CLASSES, 42, saida)
    assert contagem["quase_copias_fora"] == 2
    assert contagem["treino"] == 23 and contagem["validacao"] + contagem["teste"] == 15
    with np.load(saida) as z:
        assert len(z["labels"]) == 38
        holdout_nomes = {p.name for p in holdout.rglob("*.jpg")}
        for caminho, s in zip(z["paths"], z["split"], strict=True):
            assert (s > 0) == (caminho.split("/")[-1].split("\\")[-1] in holdout_nomes)


def test_rodada_no_formato_do_adapt_campo(mundo, tmp_path) -> None:  # noqa: F811
    from training import adapt_campo

    pesos = tmp_path / "m" / "coffee_foliar.keras"
    pesos.parent.mkdir()
    mundo["base"].save(pesos.with_suffix(".head.npz"))
    x_ant, y_ant = mundo["antigo"]
    x_va, y_va = mundo["antigo_val"]
    x_te, y_te = mundo["antigo_teste"]
    antigo = _npz(tmp_path / "antigo.npz", np.vstack([x_ant, x_va, x_te]),
                  np.concatenate([y_ant, y_va, y_te]),
                  split=np.concatenate([np.zeros(len(y_ant), int), np.ones(len(y_va), int),
                                        np.full(len(y_te), 2)]))
    comum = ["--model", "coffee-foliar", "--weights", str(pesos)]
    assert adapt_campo.main(["memoria", *comum, "--embeddings", str(antigo),
                             "--por-classe", "60"]) == 0
    # "degradadas": o domínio novo do mundo sintético, com corte próprio
    x_nov, y_nov = mundo["novo"]
    x_nt, y_nt = mundo["novo_teste"]
    x = np.vstack([x_nov, x_nt])
    split = np.concatenate([np.where(np.arange(len(y_nov)) % 5 == 0, 1, 0),
                            np.full(len(y_nt), 2)])
    ruins = tmp_path / "degradadas.npz"
    np.savez(ruins, emb=np.repeat(x[:, None, :], 4, axis=1),
             labels=np.concatenate([y_nov, y_nt]), classes=np.array(CLASSES),
             paths=np.array([f"f{i}.jpg#v0" for i in range(len(x))]),
             groups=np.array([f"f{i}.jpg" for i in range(len(x))]), split=split)
    codigo = rc.main(["rodada", *comum, "--degradadas", str(ruins), "--rodada", "rob1",
                      "--epocas", "40", "--taxa", "1e-2"])
    pasta = pesos.parent / "_continuo" / "coffee_foliar" / "rodadas" / "rob1"
    laudo = json.loads((pasta / "laudo.json").read_text(encoding="utf-8"))
    assert codigo == 0, laudo["portao"]["reasons"]
    assert laudo["escopo"] == "robustez-captura" and laudo["dados"]["teste"] == len(y_nt)
    assert adapt_campo.main(["promove", *comum, "--rodada", "rob1", "--aprovador", "X"]) == 0
