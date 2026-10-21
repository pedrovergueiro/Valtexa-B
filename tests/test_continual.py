"""Aprendizado contínuo da cabeça (:mod:`app.inference.continual`, ``training.adapt_campo``).

Os dados são sintéticos e pequenos (embedding de 16 dimensões, três classes) com a
estrutura do problema real: um domínio antigo, em que a cabeça de partida foi treinada, e um
domínio novo — o campo — em que as mesmas classes aparecem deslocadas. O que se testa:

* o gradiente do treino em NumPy (contra diferenças finitas, com saltos e projeção);
* o esquecimento: ajuste fino só com o campo aprende o novo e esquece o antigo; com replay,
  destilação e âncora aprende sem esquecer — e o portão distingue os dois;
* a memória (herding, cota por origem, esquecimento por consentimento) e o portão (pareado,
  margens, ganho no campo, deriva do perfil da borda);
* o ciclo inteiro pela linha de comando: memoria → adapta → promove → monitora → reverte →
  esquece, com o histórico e o perfil da borda carimbado.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.inference.continual import (
    AdaptConfig,
    GateCriteria,
    RegressionSet,
    RegressionSuite,
    ReplayMemory,
    _com_parametros,
    _parametros,
    adapt,
    backward,
    compare,
    edge_drift,
    forward,
    herding,
    paired_comparison,
    read_ledger,
    standardize,
)
from app.inference.heads import DiagnosisHead, EmbeddingMLP, softmax

CLASSES = ("Healthy", "Phoma_Leaf_Spot", "Rust")
D = 16
TORRE = "teste/torre"


def _dominio(rng, centros, n_por_classe, ruido=0.55):
    x = np.vstack([c + ruido * rng.normal(size=(n_por_classe, D)) for c in centros])
    y = np.repeat(np.arange(len(centros)), n_por_classe)
    return x, y


@pytest.fixture(scope="module")
def mundo():
    """Domínio antigo, domínio novo e a cabeça de partida (treinada só no antigo).

    O campo desloca o domínio inteiro (luz, câmera, cultivar: ``u``) e cada classe nova fica a
    meio caminho da classe antiga vizinha — a cabeça de partida acerta ~58% dele, e aprender o
    novo puxa as fronteiras para cima do antigo: o regime em que o esquecimento aparece.
    """
    rng = np.random.default_rng(0)
    antigos = rng.normal(size=(3, D)) * 1.6
    u = rng.normal(size=D)
    u = 3.0 * u / np.linalg.norm(u)
    novos = np.stack([0.5 * antigos[k] + 0.5 * antigos[(k + 1) % 3] for k in range(3)]) + u
    x_ant, y_ant = _dominio(rng, antigos, 160)
    x_ant_te, y_ant_te = _dominio(rng, antigos, 80)
    x_nov, y_nov = _dominio(rng, novos, 70)
    x_nov_te, y_nov_te = _dominio(rng, novos, 80)
    inicial = EmbeddingMLP(np.zeros(D), np.ones(D), (rng.normal(size=(24, D)) * 0.3,),
                           (np.zeros(24),), ("relu",), rng.normal(size=(3, 24)) * 0.3,
                           np.zeros(3))
    xn = x_ant / np.linalg.norm(x_ant, axis=1, keepdims=True)
    inicial = dataclasses.replace(inicial, mean=xn.mean(0), std=xn.std(0) + 1e-6)
    cabeca = DiagnosisHead(CLASSES, inicial, TORRE, {"cnn": 0.0, "head": 1.0})
    base = adapt(cabeca, x_ant, y_ant, None, x_ant_te[:60], y_ant_te[:60],
                 AdaptConfig(epocas=40, taxa=5e-3, ancora=0.0, replay_por_lote=0)).head
    antigo_val = _dominio(rng, antigos, 40)  # sorteado por último: o resto não muda
    return {"base": base, "antigo": (x_ant, y_ant), "antigo_teste": (x_ant_te, y_ant_te),
            "antigo_val": antigo_val, "novo": (x_nov, y_nov), "novo_teste": (x_nov_te, y_nov_te)}


def _acc(head, x, y) -> float:
    return float((head.mlp.probabilities(x).argmax(1) == y).mean())


def _memoria(base, x, y, por_classe=60) -> ReplayMemory:
    return ReplayMemory.empty(CLASSES, TORRE, D, por_classe).add(x, y, base, "publico:antigo",
                                                                 np.array([f"a{i}" for i in
                                                                           range(len(y))]))


class TestRede:
    def test_gradiente_confere_com_diferencas_finitas(self) -> None:
        rng = np.random.default_rng(0)
        mlp = EmbeddingMLP(
            mean=np.zeros(5), std=np.ones(5),
            weights=(rng.normal(size=(4, 5)), rng.normal(size=(3, 4)), rng.normal(size=(4, 3))),
            biases=(rng.normal(size=4), rng.normal(size=3), rng.normal(size=4)),
            activations=("gelu", "tanh", "sigmoid"),
            out_weight=rng.normal(size=(3, 4)), out_bias=rng.normal(size=3),
            skips=((1, 3, -1), (0, 2, 0)),
            skip_weights=(rng.normal(size=(3, 5)),), skip_biases=(rng.normal(size=3),))
        x = rng.normal(size=(6, 5))
        y = rng.integers(0, 3, 6)
        params = _parametros(mlp)
        xs = standardize(mlp, x)
        assert np.allclose(forward(mlp, params, xs)[0], mlp.logits(x))

        def perda(ps) -> float:
            p = softmax(forward(mlp, ps, xs)[0])
            return float(-np.log(p[np.arange(6), y]).mean())

        logits, cache = forward(mlp, params, xs)
        d = softmax(logits)
        d[np.arange(6), y] -= 1
        grads = backward(mlp, params, cache, d / 6)
        for p, g in zip(params, grads, strict=True):
            num = np.zeros_like(p)
            for j in np.ndindex(p.shape):
                orig = p[j]
                p[j] = orig + 1e-6
                a = perda(params)
                p[j] = orig - 1e-6
                b = perda(params)
                p[j] = orig
                num[j] = (a - b) / 2e-6
            assert np.abs(num - g).max() <= 1e-5 * (np.abs(num).max() + 1e-9)

    def test_parametros_ida_e_volta(self, mundo) -> None:
        mlp = mundo["base"].mlp
        de_volta = _com_parametros(mlp, _parametros(mlp))
        x = mundo["novo"][0]
        assert np.allclose(de_volta.logits(x), mlp.logits(x))


class TestEsquecimento:
    def test_partida_sabe_o_antigo_e_nao_o_novo(self, mundo) -> None:
        base = mundo["base"]
        assert _acc(base, *mundo["antigo_teste"]) > 0.95
        assert _acc(base, *mundo["novo_teste"]) < 0.8

    def test_ingenuo_esquece_proposto_nao(self, mundo) -> None:
        base = mundo["base"]
        x_nov, y_nov = mundo["novo"]
        val = (x_nov[::5], y_nov[::5])
        ingenuo = adapt(base, x_nov, y_nov, None, *val,
                        AdaptConfig(epocas=60, taxa=1e-2, ancora=0.0, destilacao=0.0,
                                    replay_por_lote=0)).head
        memoria = _memoria(base, *mundo["antigo"])  # 60 por classe
        proposto = adapt(base, x_nov, y_nov, memoria, *val,
                         AdaptConfig(epocas=60, taxa=1e-2), old_val=mundo["antigo_val"]).head
        antigo, novo = mundo["antigo_teste"], mundo["novo_teste"]
        assert _acc(ingenuo, *novo) > _acc(base, *novo) + 0.1  # aprendeu o campo...
        assert _acc(ingenuo, *antigo) < _acc(base, *antigo) - 0.05  # ...e esqueceu
        assert _acc(proposto, *novo) > _acc(base, *novo) + 0.1
        assert _acc(proposto, *antigo) >= _acc(base, *antigo) - 0.02

        suite = RegressionSuite([RegressionSet("antigo:teste", *antigo)], CLASSES, TORRE)
        campo = RegressionSet("campo", *novo)
        assert compare(base, ingenuo, suite, campo).decision == "REPROVADA"
        laudo = compare(base, proposto, suite, campo)
        assert laudo.decision == "APROVADA", laudo.reasons
        assert laudo.field["acuracia_balanceada"]["diferenca"] > 0.1

    def test_sem_epoca_melhor_a_cabeca_volta_intacta(self, mundo, tmp_path) -> None:
        base = mundo["base"]
        x, y = mundo["antigo_teste"]
        res = adapt(base, x, y, None, x, y, AdaptConfig(epocas=0))
        assert res.head is base and res.epoch == 0
        base.save(tmp_path / "a.npz")
        res.head.save(tmp_path / "b.npz")
        assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()

    def test_so_a_ultima_camada(self, mundo) -> None:
        base = mundo["base"]
        x_nov, y_nov = mundo["novo"]
        nova = adapt(base, x_nov, y_nov, None, x_nov[::5], y_nov[::5],
                     AdaptConfig(epocas=20, taxa=1e-2, camadas="ultima")).head
        assert all(np.array_equal(a, b) for a, b in zip(nova.mlp.weights, base.mlp.weights,
                                                         strict=True))
        assert not np.array_equal(nova.mlp.out_weight, base.mlp.out_weight)


class TestMemoria:
    def test_herding_acompanha_a_media_melhor_que_o_acaso(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, D)) + rng.normal(size=D)
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        media = unit.mean(0)
        escolha = herding(x, 10)
        assert len(set(escolha.tolist())) == 10
        erro_herding = np.linalg.norm(unit[escolha].mean(0) - media)
        erros_acaso = [np.linalg.norm(unit[rng.choice(300, 10, replace=False)].mean(0) - media)
                       for _ in range(50)]
        assert erro_herding < np.percentile(erros_acaso, 5)

    def test_cota_por_origem_nao_deixa_a_rodada_nova_expulsar_a_antiga(self, mundo) -> None:
        base = mundo["base"]
        x, y = mundo["antigo"]
        memoria = ReplayMemory.empty(CLASSES, TORRE, D, 20).add(x, y, base, "publico:antigo")
        assert len(memoria) == 60
        xn, yn = mundo["novo"]
        memoria = memoria.add(xn, yn, base, "campo:T1:r1", round_index=1)
        contagem = memoria.counts()
        for classe in CLASSES:
            assert contagem["publico:antigo"][classe] == 10
            assert contagem["campo:T1:r1"][classe] == 10
        # Origem pequena: a sobra vai para a outra.
        memoria = ReplayMemory.empty(CLASSES, TORRE, D, 20).add(x, y, base, "publico:antigo")
        memoria = memoria.add(xn[yn == 0][:3], yn[yn == 0][:3], base, "campo:pequeno")
        assert memoria.counts()["campo:pequeno"]["Healthy"] == 3
        assert memoria.counts()["publico:antigo"]["Healthy"] == 17

    def test_logits_da_epoca_e_esquecimento_por_consentimento(self, mundo, tmp_path) -> None:
        base = mundo["base"]
        x, y = mundo["novo"]
        refs = np.array([f"campo_T01_m1_P{i:03d}.jpg" if i % 2 else f"campo_T02_m1_P{i:03d}.jpg"
                         for i in range(len(y))])
        memoria = ReplayMemory.empty(CLASSES, TORRE, D, 500).add(x, y, base, "campo:x", refs)
        assert np.allclose(memoria.logits, base.mlp.logits(memoria.embeddings))
        memoria.save(tmp_path / "m.npz")
        lida = ReplayMemory.load(tmp_path / "m.npz")
        assert lida.class_names == CLASSES and len(lida) == len(memoria)
        assert np.allclose(lida.embeddings, memoria.embeddings, atol=1e-2)  # float16
        lida, tirados = lida.forget("campo_T01")
        assert tirados == len(y) // 2 and not any(r.startswith("campo_T01") for r in lida.refs)

    def test_torre_diferente_e_recusada(self, mundo) -> None:
        memoria = ReplayMemory.empty(CLASSES, "outra/torre", D, 10)
        with pytest.raises(ValueError, match="Torre"):
            memoria.add(*mundo["antigo"], mundo["base"], "x")


class TestPortao:
    def test_cabeca_igual_nao_tem_ganho_e_nao_passa(self, mundo) -> None:
        base = mundo["base"]
        suite = RegressionSuite([RegressionSet("antigo", *mundo["antigo_teste"])], CLASSES, TORRE)
        laudo = compare(base, base, suite, RegressionSet("campo", *mundo["novo_teste"]))
        assert laudo.decision == "REPROVADA" and any("sem ganho" in m for m in laudo.reasons)
        assert laudo.sets["antigo"]["acuracia_balanceada"]["ic"] == [0.0, 0.0]
        assert compare(base, base, suite, None).decision == "REPROVADA"

    def test_conjunto_so_de_medicao_nao_bloqueia(self, mundo) -> None:
        base = mundo["base"]
        ruim = dataclasses.replace(base, mlp=dataclasses.replace(
            base.mlp, out_bias=base.mlp.out_bias + np.array([0.0, 0.0, 50.0])))
        suite = RegressionSuite([RegressionSet("externo", *mundo["antigo_teste"], gate=False)],
                                CLASSES, TORRE)
        crit = GateCriteria(exige_ganho_campo=False)
        assert compare(base, ruim, suite, None, crit).decision == "APROVADA"
        suite.sets[0].gate = True
        laudo = compare(base, ruim, suite, None, crit)
        assert laudo.decision == "REPROVADA"
        assert any("alarme falso" in m for m in laudo.reasons)  # tudo vira Rust: sadia cai

    def test_comparacao_pareada_conta_os_desacordos(self) -> None:
        y = np.array([0, 0, 1, 1, 2, 2])
        atual = np.eye(3)[[0, 0, 1, 1, 2, 2]]
        nova = np.eye(3)[[0, 1, 1, 1, 2, 0]]
        comp = paired_comparison(atual, nova, y, CLASSES, GateCriteria(reamostragens=200))
        assert comp["concordancia"] == {"so_atual_acerta": 2, "so_nova_acerta": 0}
        assert comp["alarme_falso_sadia"]["nova"] == 0.5
        assert comp["classes"]["Rust"]["nova"] == 0.5

    def test_deriva_do_perfil_da_borda(self, mundo) -> None:
        base = mundo["base"]
        perfil = {"evidence": {"bin_edges": [0.0, 0.5, 0.9, 1.0],
                               "llr_tables": {"Rust": [-0.2, 0.5, 1.0],
                                              "Phoma_Leaf_Spot": [-0.2, 0.5, 1.0]},
                               "llr_clip": 1.0, "event_evidence": -0.04,
                               "near_miss_evidence": -0.08}}
        x, y = mundo["antigo_teste"]
        igual = edge_drift(base, base, x, y, perfil)
        assert igual["avaliado"] and not igual["recalibrar"]
        assert igual["por_doenca"]["Rust"]["subida"] == 0.0
        puxada = dataclasses.replace(base, mlp=dataclasses.replace(
            base.mlp, out_bias=base.mlp.out_bias + np.array([0.0, 0.0, 6.0])))
        deriva = edge_drift(base, puxada, x, y, perfil)
        assert deriva["recalibrar"] and deriva["por_doenca"]["Rust"]["subida"] > 0.02
        assert not edge_drift(base, base, x, y, {})["avaliado"]


# ---------------------------------------------------------------------------------------------
# A linha de comando, de ponta a ponta
# ---------------------------------------------------------------------------------------------


def _npz(caminho: Path, x, y, views=1, split=None, prefixo="f", grupos=None) -> Path:
    emb = np.repeat(x[:, None, :], views, axis=1) + (
        0.02 * np.random.default_rng(0).normal(size=(len(y), views, D)) if views > 1 else 0)
    extra = {"split": split} if split is not None else {}
    if grupos is not None:
        extra["groups"] = grupos
    np.savez(caminho, emb=emb, labels=y, classes=np.array(CLASSES),
             paths=np.array([f"{prefixo}_{i:04d}.jpg" for i in range(len(y))]), **extra)
    return caminho


def test_ciclo_pela_linha_de_comando(mundo, tmp_path, capsys) -> None:
    from training import adapt_campo

    modelos = tmp_path / "modelos"
    modelos.mkdir()
    pesos = modelos / "coffee_foliar.keras"
    mundo["base"].save(pesos.with_suffix(".head.npz"))
    sha_base = hashlib.sha256(pesos.with_suffix(".head.npz").read_bytes()).hexdigest()
    perfil = {"version": 1, "crop": "coffee",
              "head": {"encoder_model_id": TORRE, "encoder_sha256": "a" * 64,
                       "input_size": 224},
              "evidence": {"bin_edges": [0.0, 0.5, 0.9, 1.0],
                           "llr_tables": {"Rust": [-0.2, 0.5, 1.0]}, "llr_clip": 1.0,
                           "event_evidence": -0.04, "near_miss_evidence": -0.08}}
    pesos.with_suffix(".edge.json").write_text(json.dumps(perfil), encoding="utf-8")
    x_ant, y_ant = mundo["antigo"]
    x_te, y_te = mundo["antigo_teste"]
    x_va, y_va = mundo["antigo_val"]
    antigo = _npz(tmp_path / "antigo.npz", np.vstack([x_ant, x_va, x_te]),
                  np.concatenate([y_ant, y_va, y_te]),
                  split=np.concatenate([np.zeros(len(y_ant), int), np.ones(len(y_va), int),
                                        np.full(len(y_te), 2)]),
                  prefixo="publico")
    x_nov, y_nov = mundo["novo"]
    plantas = np.array([f"T01:P{i // 3:03d}" for i in range(len(y_nov))])  # 3 recortes/planta
    refs = [f"campo_T01_m1_P{i // 3:03d}_q{i}_000.jpg" for i in range(len(y_nov))]
    campo = tmp_path / "campo.npz"
    np.savez(campo, emb=np.repeat(x_nov[:, None, :], 4, axis=1), labels=y_nov,
             classes=np.array(CLASSES), paths=np.array(refs), groups=plantas)
    comum = ["--model", "coffee-foliar", "--weights", str(pesos)]

    assert adapt_campo.main(["memoria", *comum, "--embeddings", str(antigo),
                             "--por-classe", "60"]) == 0
    pasta = modelos / "_continuo" / "coffee_foliar"
    assert (pasta / "replay.npz").exists() and (pasta / "regressao.npz").exists()
    assert [s.name for s in RegressionSuite.load(pasta / "validacao.npz").sets] == [
        "antigo:validacao"]
    with pytest.raises(SystemExit, match="refaz"):
        adapt_campo.main(["memoria", *comum, "--embeddings", str(antigo)])

    codigo = adapt_campo.main(["adapta", *comum, "--campo-embeddings", str(campo),
                               "--escopo", "T01", "--rodada", "r1", "--epocas", "40",
                               "--taxa", "1e-2"])
    laudo = json.loads((pasta / "rodadas" / "r1" / "laudo.json").read_text(encoding="utf-8"))
    assert codigo == 0, laudo["portao"]["reasons"]
    assert laudo["portao"]["decision"] == "APROVADA"
    assert laudo["dados"]["campo"]["plantas"] == len(set(plantas))
    # Nenhuma planta dos dois lados do corte.
    with np.load(pasta / "rodadas" / "r1" / "campo.npz") as z:
        lados = {}
        for g, s in zip(z["groups"], z["split"], strict=True):
            lados.setdefault(str(g), set()).add(int(s))
        assert all(len(v) == 1 for v in lados.values())
    assert "APROVADA" in (pasta / "rodadas" / "r1" / "laudo.md").read_text(encoding="utf-8")

    with pytest.raises(SystemExit, match="aprovador"):
        adapt_campo.main(["promove", *comum, "--rodada", "r1"])
    assert adapt_campo.main(["promove", *comum, "--rodada", "r1", "--aprovador",
                             "Eng. Agr. Teste"]) == 0
    sha_nova = hashlib.sha256(pesos.with_suffix(".head.npz").read_bytes()).hexdigest()
    assert sha_nova != sha_base and (pasta / "cabecas" / f"{sha_base}.head.npz").exists()
    carimbado = json.loads(pesos.with_suffix(".edge.json").read_text(encoding="utf-8"))
    assert carimbado["head"]["head_sha256"] in (sha_base, sha_nova)
    assert carimbado["measured"]["continuo"]["rodada"] == "r1"
    assert (pasta / "perfis" / f"{sha_base}.edge.json").exists()
    # A procedência vai ao lado da cabeça e confere com ela.
    from app.inference.atualizacao import verifica_procedencia

    proc_path = pesos.with_suffix(".head.provenance.json")
    proc = json.loads(proc_path.read_text(encoding="utf-8"))
    assert verifica_procedencia(proc, sha_nova, TORRE) == []
    assert proc["cabeca_base"] == sha_base and proc["aprovado_por"] == "Eng. Agr. Teste"
    assert proc["tipo"] == "cabeca" and laudo["atualizacao"]["tipo"] == "cabeca"
    assert carimbado["measured"]["continuo"]["procedencia_sha256"] == hashlib.sha256(
        proc_path.read_bytes()).hexdigest()
    from app.inference.continual import RegressionSuite as Suite

    assert "campo:T01:r1" in [s.name for s in Suite.load(pasta / "regressao.npz").sets]
    assert "campo:T01:r1" in ReplayMemory.load(pasta / "replay.npz").counts()
    assert "campo:T01:r1:validacao" in [s.name for s in Suite.load(pasta / "validacao.npz").sets]

    # A mesma rodada não promove duas vezes sobre outra cabeça.
    with pytest.raises(SystemExit, match="não é a que a rodada comparou"):
        adapt_campo.main(["promove", *comum, "--rodada", "r1", "--aprovador", "X"])

    x_nt, y_nt = mundo["novo_teste"]
    novo_lote = _npz(tmp_path / "lote.npz", x_nt, y_nt, prefixo="campo_T01_m2")
    assert adapt_campo.main(["monitora", *comum, "--campo-embeddings", str(novo_lote)]) == 0
    ultimo = read_ledger(pasta / "historico.jsonl")[-1]
    assert ultimo["evento"] == "monitoramento" and ultimo["estado"] == "OK"

    assert adapt_campo.main(["reverte", *comum, "--motivo", "teste"]) == 0
    assert hashlib.sha256(pesos.with_suffix(".head.npz").read_bytes()).hexdigest() == sha_base
    assert not proc_path.exists()  # a cabeça de origem não tem procedência de adaptação
    restaurado = json.loads(pesos.with_suffix(".edge.json").read_text(encoding="utf-8"))
    assert "continuo" not in restaurado.get("measured", {})  # o perfil medido com a base

    assert adapt_campo.main(["esquece", *comum, "--fonte", "campo_T01"]) == 0
    assert "campo:T01:r1" not in [s.name for s in Suite.load(pasta / "regressao.npz").sets]
    assert not any(str(r).startswith("campo_T01")
                   for r in ReplayMemory.load(pasta / "replay.npz").refs)

    assert adapt_campo.main(["historico", *comum]) == 0
    eventos = [e["evento"] for e in read_ledger(pasta / "historico.jsonl")]
    assert eventos == ["memoria", "rodada", "promovida", "monitoramento", "revertida",
                       "esquecimento"]
    capsys.readouterr()


def test_rodada_reprovada_nao_promove(mundo, tmp_path) -> None:
    from training import adapt_campo

    modelos = tmp_path / "m"
    modelos.mkdir()
    pesos = modelos / "coffee_foliar.keras"
    mundo["base"].save(pesos.with_suffix(".head.npz"))
    x_ant, y_ant = mundo["antigo"]
    x_te, y_te = mundo["antigo_teste"]
    antigo = _npz(tmp_path / "antigo.npz", np.vstack([x_ant, x_te]),
                  np.concatenate([y_ant, y_te]),
                  split=np.concatenate([np.zeros(len(y_ant), int), np.full(len(y_te), 2)]))
    comum = ["--model", "coffee-foliar", "--weights", str(pesos)]
    adapt_campo.main(["memoria", *comum, "--embeddings", str(antigo), "--por-classe", "60"])
    x_nov, y_nov = mundo["novo"]
    campo = _npz(tmp_path / "c.npz", x_nov, y_nov, views=4,
                 grupos=np.array([f"P{i // 3}" for i in range(len(y_nov))]))
    # Ajuste fino ingênuo (sem replay, sem âncora, sem destilação): esquece o antigo.
    codigo = adapt_campo.main(["adapta", *comum, "--campo-embeddings", str(campo),
                               "--escopo", "T9", "--rodada", "ruim", "--epocas", "60",
                               "--taxa", "1e-2", "--sem-replay", "--ancora", "0",
                               "--destilacao", "0"])
    assert codigo == 2
    with pytest.raises(SystemExit, match="REPROVADA"):
        adapt_campo.main(["promove", *comum, "--rodada", "ruim", "--aprovador", "X"])
    with pytest.raises(SystemExit, match="motivo"):
        adapt_campo.main(["promove", *comum, "--rodada", "ruim", "--forcar"])


def test_poucas_plantas_nao_dividem(mundo) -> None:
    from training.adapt_campo import split_by_group

    with pytest.raises(SystemExit, match="pelo menos 4"):
        split_by_group(np.array(["a", "a", "b", "c"]), 0, 0.25, 0.15)
    tr, va, te = split_by_group(np.array([f"p{i // 2}" for i in range(40)]), 0, 0.25, 0.15)
    assert len(tr) + len(va) + len(te) == 40 and len(te) == 10


def test_import_campo_grava_a_origem_de_cada_recorte(tmp_path) -> None:
    from PIL import Image

    from training import import_campo

    originais = tmp_path / "originais"
    originais.mkdir()
    Image.new("RGB", (600, 400), (30, 140, 30)).save(originais / "r-1.jpg")
    registro = tmp_path / "registro.jsonl"
    caixas = [{"box": [i * 150, 0, i * 150 + 150, 150],
               "probabilities": {"Rust": 0.9, "Healthy": 0.1}} for i in range(2)]
    registro.write_text(json.dumps({"field_id": "T1", "pass_id": "m-1", "plant": "L03-P0001",
                                    "frame": "r-1", "tiles": caixas}) + "\n"
                        + json.dumps({"field_id": "T1", "pass_id": "m-1", "plant": "L03-P0002",
                                      "frame": "r-1", "tiles": caixas[:1]}) + "\n",
                        encoding="utf-8")
    labels = {("T1", "L03-P0001", "m-1"): "Healthy", ("T1", "L03-P0002", "m-1"): "Rust"}
    import_campo.importa(labels, [registro], originais, tmp_path / "ds")
    with (tmp_path / "ds" / "campo.csv").open(encoding="utf-8") as fh:
        linhas = list(csv.DictReader(fh))
    assert sorted((r["plant_id"], r["rotulo_planta"]) for r in linhas) == [
        ("L03-P0001", "Healthy"), ("L03-P0001", "Healthy"), ("L03-P0002", "Rust")]
    from training.adapt_campo import _grupos_do_campo

    arquivos = [tmp_path / "ds" / "Healthy" / r["arquivo"] for r in linhas]
    assert set(_grupos_do_campo(tmp_path / "ds", arquivos)) == {"T1:L03-P0001", "T1:L03-P0002"}


def test_perfil_da_borda_leva_o_sha_da_cabeca(tmp_path, monkeypatch, mundo) -> None:
    from app.config import settings
    from app.registry import get_spec
    from training import fit_edge

    monkeypatch.setattr(settings, "models_dir", tmp_path)
    (tmp_path / "framing_vision.onnx").write_bytes(b"torre de visao")
    spec = get_spec("coffee-foliar")
    assert "head_sha256" not in fit_edge.head_block(spec, "cabeca:teste")
    mundo["base"].save(spec.head_path)
    bloco = fit_edge.head_block(spec, "cabeca:teste")
    assert bloco["head_sha256"] == hashlib.sha256(spec.head_path.read_bytes()).hexdigest()


def test_api_mostra_o_historico_da_cabeca(client, tmp_path, monkeypatch, mundo) -> None:
    from app.config import settings
    from app.inference.continual import append_ledger, continual_dir
    from app.registry import get_spec

    monkeypatch.setattr(settings, "models_dir", tmp_path)
    spec = get_spec("coffee-foliar")
    vazio = client.get("/models/coffee-foliar/continuo").json()
    assert vazio == {"key": "coffee-foliar", "head_sha256": None, "history": [],
                     "memory": None, "regression_sets": [], "validation_sets": []}
    mundo["base"].save(spec.head_path)
    pasta = continual_dir(spec.weights_path)
    _memoria(mundo["base"], *mundo["antigo"]).save(pasta / "replay.npz")
    RegressionSuite([RegressionSet("antigo:teste", *mundo["antigo_teste"])], CLASSES,
                    TORRE).save(pasta / "regressao.npz")
    append_ledger(pasta / "historico.jsonl", {"evento": "rodada", "decisao": "APROVADA"})
    r = client.get("/models/coffee-foliar/continuo").json()
    assert r["head_sha256"] == hashlib.sha256(spec.head_path.read_bytes()).hexdigest()
    assert r["history"] == [{"evento": "rodada", "decisao": "APROVADA"}]
    assert r["memory"]["examples"] == 180 and r["memory"]["per_class"] == 60
    assert r["regression_sets"] == [{"name": "antigo:teste", "n": 240, "gate": True}]
    assert client.get("/models/nao-existe/continuo").status_code == 404


def test_cauda_de_doenca_em_sadia_barra_o_que_o_rover_multiplicaria() -> None:
    from app.inference.continual import _cauda

    y = np.zeros(200, dtype=np.int64)  # só sadias
    atual = np.tile([0.97, 0.02, 0.01], (200, 1))
    nova = atual.copy()
    nova[:6] = [0.05, 0.93, 0.02]  # 6 folhas sadias com doença "certa" (> 0,9)
    comp = paired_comparison(atual, nova, y, CLASSES, GateCriteria(reamostragens=300))
    c = comp["cauda_sadia"]
    assert c["atual"] == 0.0 and c["nova"] == 0.03 and c["n"] == 200
    assert _cauda("agregado", c, 0.005)  # 3 pontos: barrado
    assert not _cauda("agregado", {**c, "nova": 0.004, "ic": [-0.01, 0.01]}, 0.005)


def test_suavizacao_segura_a_confianca(mundo) -> None:
    from app.inference.continual import _alvo

    alvo = _alvo(np.array([0, 2]), 3, 0.1)
    assert np.allclose(alvo.sum(1), 1.0) and alvo[0, 0] == 0.9 and alvo[1, 0] == 0.05
    base = mundo["base"]
    x, y = mundo["novo"]
    dura = adapt(base, x, y, None, x[::5], y[::5],
                 AdaptConfig(epocas=40, taxa=1e-2, replay_por_lote=0, ancora=0.0)).head
    suave = adapt(base, x, y, None, x[::5], y[::5],
                  AdaptConfig(epocas=40, taxa=1e-2, replay_por_lote=0, ancora=0.0,
                              suavizacao=0.1)).head
    assert suave.mlp.probabilities(x).max(1).mean() < dura.mlp.probabilities(x).max(1).mean()
    cauda = lambda h: float((h.mlp.probabilities(x).max(1) > 0.98).mean())  # noqa: E731
    assert cauda(suave) < cauda(dura)
