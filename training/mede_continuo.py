"""Mede o aprendizado contínuo com embeddings reais: aprende o domínio novo? Esquece o antigo?

    python -m training.mede_continuo --head base.head.npz --embeddings coffee.npz \\
        --novo bracol.npz --externo coffee_ext.npz --saida mede_continuo.json

O protocolo simula uma fazenda nova chegando em rodadas de campo:

* a cabeça de partida (``--head``) foi treinada **só** com ``--embeddings`` (o domínio antigo);
* ``--novo`` faz o papel do campo: tira-se o que é quase cópia do antigo (cosseno > 0,97,
  como no ``fit_heads``), separa-se um **teste final** (30% das folhas, nunca visto por
  ninguém) e o resto vira ``--rodadas`` lotes de campo, um por rodada;
* cada rodada divide o seu lote como o ``adapt_campo`` faz (60/15/25 por grupo), adapta,
  passa pelo portão contra a cabeça do momento e, se aprovada, é promovida (memória e
  regressão crescem);
* estratégias: ``ingenuo`` (ajuste fino só com o campo — o que dá esquecimento), ``replay``
  (com memória, sem destilação nem âncora), ``proposto_sem_selecao`` (replay + destilação +
  L2-SP) e ``proposto`` (isso e a seleção de época consciente de regressão: o padrão do
  ``adapt_campo``). O ``ingenuo`` é promovido mesmo reprovado
  (``ingenuo_forcado``), para medir o esquecimento que o portão evitou;
* depois das rodadas do ``proposto``, uma rodada com um lote **novo** (nunca visto) e 40% dos
  rótulos embaralhados — a validação malfeita — mede se o portão recusa; e a mesma rodada com
  os rótulos certos mede se ele aceita (o controle);
* ``proposto_dnaty``: o ``proposto`` partindo da cabeça com os neurônios dormentes renascidos
  pela plasticidade do dNATY 2.2 (``training.motor_dnaty``; precisa de torch e dnaty). Entra
  em produção só se ganhar do ``proposto`` aqui. ``--estrategias`` escolhe quais rodar.

``--externo`` só mede (no café, o acervo de licença só-medição). Tudo em CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from app.inference.continual import (
    AdaptConfig,
    GateCriteria,
    RegressionSet,
    RegressionSuite,
    ReplayMemory,
    adapt,
    compare,
    metrics,
)
from app.inference.heads import DiagnosisHead
from training.adapt_campo import DUPLICATE_COSINE, load_embeddings, split_by_group
from training.motor_dnaty import Plasticidade

ESTRATEGIAS = {
    "ingenuo": (AdaptConfig(destilacao=0.0, ancora=0.0, replay_por_lote=0), False),
    "replay": (AdaptConfig(destilacao=0.0, ancora=0.0, tolerancia_validacao=None), True),
    "proposto_sem_selecao": (AdaptConfig(tolerancia_validacao=None), True),
    "proposto": (AdaptConfig(), True),
    "proposto_dnaty": (AdaptConfig(), True),
    "proposto_dnaty_puro": (AdaptConfig(), True),
}
PLASTICIDADE = {"proposto_dnaty": Plasticidade(),
                "proposto_dnaty_puro": Plasticidade(limiar_relativo=None)}
"""``puro``: a fração do dNATY sem exigir dormência (os 5% menos úteis renascem sempre)."""


def _l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def avalia(head: DiagnosisHead, conjuntos: dict) -> dict:
    return {nome: metrics(head.mlp.probabilities(x), y, head.class_names)
            for nome, (x, y) in conjuntos.items()}


def rodada(head, memoria, suite, antigo_val, lote, cfg, usa_replay, seed, nome,
           embaralha=0.0, plast: Plasticidade | None = None):
    tr, va, te = split_by_group(lote.groups, seed, 0.25, 0.15)
    y = lote.labels.copy()
    if embaralha:
        rng = np.random.default_rng(seed + 7)
        trocar = rng.random(len(y)) < embaralha
        y[trocar] = rng.integers(0, len(head.class_names), int(trocar.sum()))
    v = lote.emb.shape[1]
    inicio = time.perf_counter()
    argumentos = (head, lote.emb[tr].reshape(-1, lote.emb.shape[-1]), np.repeat(y[tr], v),
                  memoria if usa_replay else None, lote.emb[va, 0], y[va], cfg,
                  antigo_val if usa_replay else None)
    motor = None
    if plast is not None:
        from training.motor_dnaty import adapta as adapta_com_plasticidade

        res, motor = adapta_com_plasticidade(*argumentos, dataclasses.replace(plast, seed=seed))
    else:
        res = adapt(*argumentos)
    segundos = time.perf_counter() - inicio
    teste = RegressionSet(nome, lote.emb[te, 0], y[te], True, lote.refs[te])
    laudo = compare(head, res.head, suite, teste, GateCriteria(seed=seed))
    return res, laudo, (tr, va, te, y), teste, segundos, motor


def experimento(args, seed: int) -> dict:
    base = DiagnosisHead.load(args.head)
    nomes = base.class_names
    antigo = load_embeddings(args.embeddings, nomes)
    novo = load_embeddings(args.novo, nomes)
    copia = (_l2(novo.emb[:, 0]) @ _l2(antigo.emb[:, 0]).T).max(axis=1) > DUPLICATE_COSINE
    novo = novo.take(np.flatnonzero(~copia))
    rng = np.random.default_rng(seed)
    grupos = rng.permutation(np.array(sorted(set(novo.groups.tolist()))))
    n_final = int(0.3 * len(grupos))
    final = np.isin(novo.groups, grupos[:n_final])
    lotes_g = np.array_split(grupos[n_final:], args.rodadas + 1)
    lotes = [novo.take(np.flatnonzero(np.isin(novo.groups, g))) for g in lotes_g]
    lote_controle = lotes.pop()  # para a rodada de rótulos ruins e o seu controle
    conjuntos = {
        "antigo_teste": (antigo.emb[antigo.split == 2, 0], antigo.labels[antigo.split == 2]),
        "novo_teste_final": (novo.emb[final, 0], novo.labels[final]),
    }
    if args.externo:
        ext = load_embeddings(args.externo, nomes)
        conjuntos["externo"] = (ext.emb[:, 0], ext.labels)
    print(f"\n=== semente {seed} === antigo: {len(antigo)} · novo: {len(novo)} (fora "
          f"{int(copia.sum())} quase cópias) · teste final do novo: {int(final.sum())} · "
          f"rodadas: {[len(x) for x in lotes]} + controle {len(lote_controle)}")

    tr0 = antigo.split == 0
    memoria0 = ReplayMemory.empty(nomes, base.encoder, antigo.emb.shape[-1],
                                  args.por_classe).add(antigo.emb[tr0, 0], antigo.labels[tr0],
                                                       base, "publico:antigo", antigo.refs[tr0])

    def suite_inicial():
        s = RegressionSuite([], nomes, base.encoder)
        te = antigo.split == 2
        s.add(RegressionSet("antigo:teste", antigo.emb[te, 0], antigo.labels[te], True))
        if args.externo:
            s.add(RegressionSet("externo", *conjuntos["externo"], gate=False))
        return s

    saida = {"seed": seed, "base": avalia(base, conjuntos), "estrategias": {}}
    escolhidas = [(n, e) for n, e in [*ESTRATEGIAS.items(), ("ingenuo_forcado",
                                                              ESTRATEGIAS["ingenuo"])]
                  if n in args.estrategias]
    for nome, (cfg, usa_replay) in escolhidas:
        cfg = dataclasses.replace(cfg, seed=seed)
        head, memoria, suite = base, memoria0, suite_inicial()
        val0 = antigo.split == 1
        antigo_val = (antigo.emb[val0, 0], antigo.labels[val0])
        historico = []
        for r, lote in enumerate(lotes, start=1):
            res, laudo, (tr, va, _te, y), teste, seg, motor = rodada(
                head, memoria, suite, antigo_val, lote, cfg, usa_replay, seed + r,
                f"campo:r{r}", plast=PLASTICIDADE.get(nome))
            promove = laudo.decision == "APROVADA" or nome == "ingenuo_forcado"
            item = {"rodada": r, "decisao": laudo.decision, "motivos": laudo.reasons,
                    "promovida": promove, "segundos": round(seg, 1), "epoca": res.epoch,
                    "ganho_campo": laudo.field["acuracia_balanceada"],
                    "candidata": avalia(res.head, conjuntos), "motor": motor}
            historico.append(item)
            if promove:
                head = res.head
                memoria = memoria.add(lote.emb[tr, 0], y[tr], head, f"campo:r{r}",
                                      lote.refs[tr], r)
                antigo_val = (np.vstack([antigo_val[0], lote.emb[va, 0]]),
                              np.concatenate([antigo_val[1], y[va]]))
                suite.add(teste)
            forcada = " (forçada)" if promove and laudo.decision != "APROVADA" else ""
            print(f"[{nome}] r{r}: {laudo.decision}{forcada} · "
                  + " · ".join(f"{k} {v['acuracia_balanceada']:.4f}"
                               for k, v in item["candidata"].items())
                  + (f" · {laudo.reasons[0]}" if laudo.reasons else ""))
        saida["estrategias"][nome] = {"rodadas": historico, "final": avalia(head, conjuntos)}
        if nome in ("proposto", "proposto_dnaty"):  # o portão contra rótulo ruim
            sufixo = "" if nome == "proposto" else "_dnaty"
            for chave, fracao in (("rotulos_ruins", 0.4), ("rotulos_certos", 0.0)):
                res, laudo, *_ = rodada(head, memoria, suite, antigo_val, lote_controle, cfg,
                                        True, seed + 99, f"campo:{chave}", embaralha=fracao,
                                        plast=PLASTICIDADE.get(nome))
                saida[chave + sufixo] = {"decisao": laudo.decision, "motivos": laudo.reasons,
                                         "candidata": avalia(res.head, conjuntos)}
                print(f"[{chave}{sufixo}] {laudo.decision}"
                      + (f" · {laudo.reasons[0]}" if laudo.reasons else ""))
    return saida


def resumo(execucoes: list[dict]) -> dict:
    """Média e desvio entre as sementes: o estado final de cada estratégia e as decisões."""
    saida = {}
    nomes = list(execucoes[0]["estrategias"])
    for nome in ["base", *nomes]:
        finais = [e["base"] if nome == "base" else e["estrategias"][nome]["final"]
                  for e in execucoes]
        linha = {}
        for conjunto in finais[0]:
            for metrica in ("acuracia_balanceada", "alarme_falso_sadia"):
                valores = [f[conjunto].get(metrica) for f in finais]
                if all(v is not None for v in valores):
                    linha[f"{conjunto}:{metrica}"] = (round(float(np.mean(valores)), 4),
                                                       round(float(np.std(valores)), 4))
        if nome != "base":
            rodadas = [it for e in execucoes for it in e["estrategias"][nome]["rodadas"]]
            linha["aprovadas"] = f"{sum(it['decisao'] == 'APROVADA' for it in rodadas)}/" \
                                 f"{len(rodadas)}"
        saida[nome] = linha
    for chave in ("rotulos_ruins", "rotulos_certos", "rotulos_ruins_dnaty",
                  "rotulos_certos_dnaty"):
        if all(chave in e for e in execucoes):
            saida[chave] = f"{sum(e[chave]['decisao'] == 'REPROVADA' for e in execucoes)}/" \
                           f"{len(execucoes)} reprovadas"
    return saida


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--novo", type=Path, required=True)
    p.add_argument("--externo", type=Path, default=None)
    p.add_argument("--rodadas", type=int, default=3)
    p.add_argument("--por-classe", type=int, default=1000)
    p.add_argument("--sementes", type=int, nargs="+", default=[42, 7, 2026])
    p.add_argument("--saida", type=Path, default=None)
    p.add_argument("--estrategias", nargs="+", default=[*ESTRATEGIAS, "ingenuo_forcado"],
                   choices=[*ESTRATEGIAS, "ingenuo_forcado"])
    args = p.parse_args(argv)
    execucoes = [experimento(args, s) for s in args.sementes]
    final = resumo(execucoes)
    print("\n=== resumo (média ± desvio entre sementes; estado final de cada estratégia) ===")
    for nome, linha in final.items():
        print(f"{nome}: {linha}")
    if args.saida:
        args.saida.write_text(json.dumps({"resumo": final, "execucoes": execucoes},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Resultado em {args.saida}")


if __name__ == "__main__":
    main()
