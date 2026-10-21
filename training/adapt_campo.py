"""Aprendizado contínuo no campo: a cabeça aprende com o que o agrônomo validou, sem esquecer.

O ciclo (o porquê de cada peça está em :mod:`app.inference.continual`)::

    # 0) uma vez por modelo: a memória de replay e os conjuntos de regressão congelados
    python -m training.adapt_campo memoria --model coffee-foliar \\
        --extra data/coffee-bracol --externo ~/valtexa-raw/coffee-leaves-disease
    #    (com embeddings já calculados: --embeddings coffee.npz --extra-embeddings bracol.npz)

    # 1) cada rodada de campo: os recortes do training.import_campo, já confirmados
    python -m training.adapt_campo adapta --model coffee-foliar \\
        --campo data/coffee-campo --escopo fazenda-boa-vista
    #    → models/_continuo/<pesos>/rodadas/<rodada>/laudo.md, e APROVADA ou REPROVADA

    #    (--motor dnaty-plasticidade: renasce os neurônios dormentes antes — training.motor_dnaty)

    # 2) só se APROVADA: instala, guarda a anterior, alimenta a memória e o portão, e grava a
    #    procedência ao lado da cabeça (o Rover não carrega cabeça adaptada sem ela)
    python -m training.adapt_campo promove --model coffee-foliar --rodada <rodada> \\
        --aprovador "Eng. Agr. Fulana (CREA 0000)"
    #    e recalibra o que o laudo pede (conformal; perfil da borda para o Rover)

    # 3) a cada lote novo de validação do agrônomo: a cabeça em produção piorou?
    python -m training.adapt_campo monitora --model coffee-foliar --campo data/coffee-campo-novo
    python -m training.adapt_campo reverte --model coffee-foliar        # se piorou

    # consentimento retirado pelo cliente (LGPD)
    python -m training.adapt_campo esquece --model coffee-foliar --fonte campo_T01
    python -m training.adapt_campo historico --model coffee-foliar

O campo é dividido **por planta** (``campo.csv`` do ``import_campo``; sem ele, por quadro):
treino, validação (escolhe a época) e teste (o portão). O teste de cada rodada aprovada vira
um conjunto de regressão: a rodada seguinte não pode esquecê-lo.

Sem GPU: o treino é NumPy. Só calcular embeddings de fotos (``--campo``, ``--extra`` sem
``*-embeddings``) roda a torre de visão do SigLIP 2 — como o ``training.fit_heads``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.inference.atualizacao import TIPOS, classifica, procedencia
from app.inference.continual import (
    AdaptConfig,
    GateCriteria,
    RegressionSet,
    RegressionSuite,
    ReplayMemory,
    adapt,
    append_ledger,
    compare,
    continual_dir,
    edge_drift,
    file_sha256,
    metrics,
    paired_comparison,
    read_ledger,
)
from app.inference.heads import DiagnosisHead
from app.registry import MODEL_SPECS, get_spec
from training import DEFAULT_SEED
from training.motor_dnaty import MOTOR_DNATY, MOTOR_NUMPY, Plasticidade

DUPLICATE_COSINE = 0.97
"""O mesmo critério de quase cópia do ``training.fit_heads``."""


# ---------------------------------------------------------------------------------------------
# Embeddings: de fotos (torre de visão) ou de um .npz já calculado
# ---------------------------------------------------------------------------------------------


@dataclass
class Embedded:
    emb: np.ndarray
    """``(N, V, D)``: V orientações (a primeira é a foto como veio)."""
    labels: np.ndarray
    refs: np.ndarray
    groups: np.ndarray
    split: np.ndarray | None
    """0 treino, 1 validação, 2 teste (o corte do treino original), se o arquivo trouxer."""

    def take(self, idx) -> Embedded:
        idx = np.asarray(idx, dtype=np.int64)
        return Embedded(self.emb[idx], self.labels[idx], self.refs[idx], self.groups[idx],
                        None if self.split is None else self.split[idx])

    def __len__(self) -> int:
        return len(self.labels)


def load_embeddings(path: Path, class_names: tuple[str, ...]) -> Embedded:
    """``.npz`` com ``emb`` (N, V, D) ou (N, D), ``labels`` e ``classes``; opcionais: ``paths``,
    ``groups``, ``split``, ``valid``, ``is_mask`` (o formato das medições do ``fit_heads``)."""
    with np.load(path, allow_pickle=False) as z:
        emb = z["emb"].astype(np.float64)
        if emb.ndim == 2:
            emb = emb[:, None, :]
        classes = tuple(str(c) for c in z["classes"])
        faltam = sorted(set(classes) - set(class_names))
        if faltam:
            raise SystemExit(f"{path}: classes que o modelo não tem: {', '.join(faltam)}.")
        mapa = np.array([class_names.index(c) for c in classes])
        labels = mapa[z["labels"].astype(np.int64)]
        manter = np.ones(len(labels), dtype=bool)
        if "valid" in z:
            manter &= z["valid"].astype(bool)
        if "is_mask" in z:
            manter &= ~z["is_mask"].astype(bool)
        if "paths" in z:
            refs = np.array([str(p) for p in z["paths"]], dtype="<U160")
        elif "ids" in z:
            refs = np.array([f"{path.stem}:{i}" for i in z["ids"]], dtype="<U160")
        else:
            refs = np.array([f"{path.stem}:{i}" for i in range(len(labels))], dtype="<U160")
        groups = (np.array([str(g) for g in z["groups"]], dtype="<U160") if "groups" in z
                  else refs.copy())
        split = z["split"].astype(np.int64) if "split" in z else None
    saida = Embedded(emb, labels, refs, groups, split)
    return saida.take(np.flatnonzero(manter))


def _grupos_do_campo(folder: Path, arquivos: list[Path]) -> np.ndarray:
    """A planta de cada recorte (``campo.csv``); sem manifesto, o quadro (nome sem o índice)."""
    origem = {}
    manifesto = folder / "campo.csv"
    if manifesto.exists():
        with manifesto.open(newline="", encoding="utf-8") as fh:
            origem = {r["arquivo"]: f"{r['field_id']}:{r['plant_id']}" for r in csv.DictReader(fh)}
    return np.array([origem.get(a.name, a.stem.rsplit("_", 1)[0]) for a in arquivos],
                    dtype="<U160")


def embed_folder(folder: Path, class_names: tuple[str, ...], views: int) -> Embedded:
    """``<Classe>/<foto>`` pela torre de visão (pastas ``_*``, como ``_revisar``, ficam fora)."""
    from app.config import settings
    from app.framing import REFERENCE_NAME, FramingReference, vision_encoder
    from training.fit_heads import embed
    from training.prepare_dataset import is_image, looks_like_label_mask

    if not folder.is_dir():
        raise SystemExit(f"{folder} não é uma pasta.")
    arquivos, labels = [], []
    for pasta in sorted(p for p in folder.iterdir() if p.is_dir()):
        if pasta.name.startswith(("_", ".")):
            continue
        if pasta.name not in class_names:
            raise SystemExit(f"{pasta}: classe {pasta.name!r} que o modelo não tem.")
        for foto in sorted(pasta.iterdir()):
            if is_image(foto) and not looks_like_label_mask(foto):
                arquivos.append(foto)
                labels.append(class_names.index(pasta.name))
    if not arquivos:
        raise SystemExit(f"{folder}: nenhuma foto em <Classe>/.")
    reference = FramingReference.load(settings.models_dir / REFERENCE_NAME)
    encoder = vision_encoder(reference.model_id)
    emb, _ = embed(encoder, [str(a) for a in arquivos], views=views)
    return Embedded(np.asarray(emb, dtype=np.float64), np.array(labels, dtype=np.int64),
                    np.array([str(a) for a in arquivos], dtype="<U160"),
                    _grupos_do_campo(folder, arquivos), None)


def dataset_split(emb: Embedded, seed: int, val: float, test: float) -> np.ndarray:
    """O corte do treino (``training.dataset.split_indices``, mesma semente do ``fit_heads``)."""
    from training.dataset import split_indices

    treino, validacao, teste = split_indices(len(emb), seed, val, test)
    split = np.zeros(len(emb), dtype=np.int64)
    split[np.asarray(validacao, dtype=np.int64)] = 1
    split[np.asarray(teste, dtype=np.int64)] = 2
    return split


# ---------------------------------------------------------------------------------------------
# Onde mora o estado
# ---------------------------------------------------------------------------------------------


def _spec(args):
    spec = get_spec(args.model)
    if getattr(args, "weights", None) is not None:
        spec = dataclasses.replace(spec, weights_name=str(args.weights.resolve()))
    return spec


def continuo_dir(spec) -> Path:
    """``models/_continuo/<pesos>/``: memória, portão, histórico, rodadas e cabeças antigas."""
    return continual_dir(spec.weights_path)


def _cabeca_atual(spec) -> DiagnosisHead:
    if not spec.head_path.exists():
        raise SystemExit(f"{spec.head_path} não existe: o aprendizado contínuo ajusta a cabeça "
                         "(training.fit_heads) — treine-a primeiro.")
    return DiagnosisHead.load(spec.head_path)


def _agora() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------------------------
# memoria
# ---------------------------------------------------------------------------------------------


def cmd_memoria(args) -> int:
    spec = _spec(args)
    cabeca = _cabeca_atual(spec)
    pasta = continuo_dir(spec)
    if (pasta / "replay.npz").exists() and not args.refaz:
        raise SystemExit(f"{pasta / 'replay.npz'} já existe (use --refaz para recomeçar a "
                         "memória — o histórico continua).")
    nomes = cabeca.class_names
    if args.embeddings:
        principal = load_embeddings(args.embeddings, nomes)
        nome = args.embeddings.stem
    else:
        pasta_dados = args.data_dir or spec.data_dir
        principal = embed_folder(pasta_dados, nomes, views=1)
        nome = pasta_dados.name
    if principal.split is None:
        principal.split = dataset_split(principal, args.seed, args.validation_split,
                                        args.test_split)
    memoria = ReplayMemory.empty(nomes, cabeca.encoder, principal.emb.shape[-1], args.por_classe)
    suite = RegressionSuite([], nomes, cabeca.encoder)
    validacao = RegressionSuite([], nomes, cabeca.encoder)
    treino = np.flatnonzero(principal.split == 0)
    teste = np.flatnonzero(principal.split == 2)
    val = np.flatnonzero(principal.split == 1)
    if len(val):
        validacao.add(RegressionSet(f"{nome}:validacao", principal.emb[val, 0],
                                    principal.labels[val], False, principal.refs[val]))
    memoria = memoria.add(principal.emb[treino, 0], principal.labels[treino], cabeca,
                          f"publico:{nome}", principal.refs[treino])
    suite.add(RegressionSet(f"{nome}:teste", principal.emb[teste, 0], principal.labels[teste],
                            True, principal.refs[teste]))
    conhecidas = _l2(principal.emb[:, 0])
    fontes = [(p, None) for p in args.extra_embeddings] + [(None, p) for p in args.extra]
    for arquivo, pasta_extra in fontes:
        extra = (load_embeddings(arquivo, nomes) if arquivo is not None
                 else embed_folder(pasta_extra, nomes, views=1))
        nome_extra = (arquivo or pasta_extra).stem if arquivo else pasta_extra.name
        if extra.split is None:
            extra.split = dataset_split(extra, args.seed, args.validation_split, args.test_split)
        copia = (_l2(extra.emb[:, 0]) @ conhecidas.T).max(axis=1) > DUPLICATE_COSINE
        tr = np.flatnonzero((extra.split == 0) & ~copia)
        te = np.flatnonzero((extra.split == 2) & ~copia)
        va = np.flatnonzero((extra.split == 1) & ~copia)
        if len(va):
            validacao.add(RegressionSet(f"{nome_extra}:validacao", extra.emb[va, 0],
                                        extra.labels[va], False, extra.refs[va]))
        memoria = memoria.add(extra.emb[tr, 0], extra.labels[tr], cabeca, f"publico:{nome_extra}",
                              extra.refs[tr])
        suite.add(RegressionSet(f"{nome_extra}:teste", extra.emb[te, 0], extra.labels[te], True,
                                extra.refs[te]))
        print(f"Extra {nome_extra}: {int(copia.sum())} quase cópias fora · memória {len(tr)} · "
              f"teste {len(te)}")
    externos = [(p, None) for p in args.externo_embeddings] + [(None, p) for p in args.externo]
    for arquivo, pasta_ext in externos:
        ext = (load_embeddings(arquivo, nomes) if arquivo is not None
               else embed_folder(pasta_ext, nomes, views=1))
        nome_ext = arquivo.stem if arquivo is not None else pasta_ext.name
        suite.add(RegressionSet(f"{nome_ext}:externo", ext.emb[:, 0], ext.labels,
                                args.externo_bloqueia, ext.refs))
    memoria.save(pasta / "replay.npz")
    suite.save(pasta / "regressao.npz")
    validacao.save(pasta / "validacao.npz")
    resumo = {s.name: {"n": int(len(s.labels)), "bloqueia": s.gate} for s in suite.sets}
    append_ledger(pasta / "historico.jsonl", {
        "evento": "memoria", "quando": _agora(), "cabeca": file_sha256(spec.head_path),
        "memoria": memoria.counts(), "regressao": resumo, "por_classe": args.por_classe,
        "validacao": {s.name: int(len(s.labels)) for s in validacao.sets}})
    print(f"Memória: {len(memoria)} exemplos em {pasta / 'replay.npz'}")
    for origem, por in memoria.counts().items():
        print(f"  {origem}: {por}")
    print("Conjuntos de regressão (congelados):")
    for s in suite.sets:
        m = metrics(cabeca.mlp.probabilities(s.embeddings), s.labels, nomes)
        print(f"  {s.name:32s} n={len(s.labels):5d}  bloqueia={s.gate!s:5s}  "
              f"acurácia balanceada da cabeça atual {m['acuracia_balanceada']:.4f}")
    return 0


# ---------------------------------------------------------------------------------------------
# adapta
# ---------------------------------------------------------------------------------------------


def split_by_group(groups: np.ndarray, seed: int, holdout: float, val: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Treino, validação e teste por grupo (planta): nenhum grupo em dois lados do corte."""
    unicos = np.array(sorted(set(groups.tolist())))
    if len(unicos) < 4:
        raise SystemExit(f"O campo tem {len(unicos)} plantas: precisa de pelo menos 4 para "
                         "separar treino, validação e teste sem misturar a mesma planta.")
    ordem = np.random.default_rng(seed).permutation(unicos)
    n_te = max(1, round(holdout * len(ordem)))
    n_va = max(1, round(val * len(ordem)))
    teste, validacao = set(ordem[:n_te]), set(ordem[n_te:n_te + n_va])
    lado = np.array([2 if g in teste else 1 if g in validacao else 0 for g in groups])
    return np.flatnonzero(lado == 0), np.flatnonzero(lado == 1), np.flatnonzero(lado == 2)


def _carrega_estado(spec):
    pasta = continuo_dir(spec)
    if not (pasta / "replay.npz").exists():
        raise SystemExit(f"Sem memória em {pasta}: rode primeiro "
                         f"`python -m training.adapt_campo memoria --model {spec.key}`.")
    return pasta, ReplayMemory.load(pasta / "replay.npz"), RegressionSuite.load(
        pasta / "regressao.npz")


def _carrega_validacao(pasta: Path, nomes, encoder) -> RegressionSuite:
    caminho = pasta / "validacao.npz"
    return (RegressionSuite.load(caminho) if caminho.exists()
            else RegressionSuite([], tuple(nomes), encoder))


def _perfil_da_borda(spec) -> dict | None:
    caminho = spec.weights_path.with_suffix(".edge.json")
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def cmd_adapta(args) -> int:
    spec = _spec(args)
    cabeca = _cabeca_atual(spec)
    pasta, memoria, suite = _carrega_estado(spec)
    if memoria.encoder != cabeca.encoder or memoria.class_names != cabeca.class_names:
        raise SystemExit("A memória foi feita para outra torre ou outras classes: refaça-a.")
    nomes = cabeca.class_names
    campo = (load_embeddings(args.campo_embeddings, nomes) if args.campo_embeddings
             else embed_folder(args.campo, nomes, views=4))
    rodada = args.rodada or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destino = pasta / "rodadas" / rodada
    if destino.exists():
        raise SystemExit(f"A rodada {rodada} já existe em {destino}.")
    tr, va, te = split_by_group(campo.groups, args.seed, args.teste, args.validacao)
    validacao = _carrega_validacao(pasta, nomes, cabeca.encoder)
    antigo = None
    if validacao.sets:
        antigo = (np.vstack([s.embeddings for s in validacao.sets]),
                  np.concatenate([s.labels for s in validacao.sets]))
    else:
        print("Aviso: sem validação do que já se sabia (validacao.npz): a época sai só pela "
              "validação do campo, sem seleção consciente de regressão.")
    v = campo.emb.shape[1]
    cfg = AdaptConfig(epocas=args.epocas, taxa=args.taxa, destilacao=args.destilacao,
                      ancora=args.ancora, camadas=args.camadas, seed=args.seed,
                      replay_por_lote=0 if args.sem_replay else 64)
    print(f"Campo ({args.escopo}): {len(campo)} recortes de {len(set(campo.groups.tolist()))} "
          f"plantas · treino {len(tr)} x{v} · validação {len(va)} · teste {len(te)}")
    print(f"Memória: {len(memoria)} no replay · validação do que já se sabia: "
          f"{0 if antigo is None else len(antigo[1])}")
    treino_x = campo.emb[tr].reshape(-1, campo.emb.shape[-1])
    treino_y = np.repeat(campo.labels[tr], v)
    replay = None if args.sem_replay else memoria
    motor = {"motor": MOTOR_NUMPY}
    if args.motor == MOTOR_DNATY:
        from training.motor_dnaty import adapta as adapta_com_plasticidade

        resultado, motor = adapta_com_plasticidade(
            cabeca, treino_x, treino_y, replay, campo.emb[va, 0], campo.labels[va], cfg, antigo,
            Plasticidade(fracao=args.renascer, seed=args.seed))
        print(f"dNATY: {motor['renascidos']} neurônio(s) renascido(s) "
              f"{motor['por_camada']}; rank efetivo {motor['rank_efetivo_antes']} → "
              f"{motor['rank_efetivo_final']}")
    else:
        resultado = adapt(cabeca, treino_x, treino_y, replay, campo.emb[va, 0],
                          campo.labels[va], cfg, antigo)
    tipo = classifica(cabeca, resultado.head)
    if not TIPOS[tipo].continuo_pode:
        raise SystemExit(f"A candidata é uma atualização de {tipo} ({TIPOS[tipo].muda}): não "
                         "sai do ciclo contínuo. " + "; ".join(TIPOS[tipo].exige) + ".")
    teste_campo = RegressionSet(f"campo:{args.escopo}:{rodada}", campo.emb[te, 0],
                                campo.labels[te], True, campo.refs[te])
    criterios = GateCriteria(margem_acuracia=args.margem, margem_classe=args.margem_classe,
                             margem_alarme_falso=args.margem_alarme_falso, seed=args.seed)
    laudo = compare(cabeca, resultado.head, suite, teste_campo, criterios)
    perfil = _perfil_da_borda(spec)
    deriva = {"avaliado": False, "motivo": "sem perfil da borda com evidência"}
    if perfil is not None:
        sadias = [s for s in suite.sets if s.gate] + [teste_campo]
        deriva = edge_drift(cabeca, resultado.head,
                            np.vstack([s.embeddings for s in sadias]),
                            np.concatenate([s.labels for s in sadias]), perfil)
    destino.mkdir(parents=True)
    resultado.head.save(destino / "candidata.head.npz")
    lado = np.zeros(len(campo), dtype=np.int64)
    lado[va], lado[te] = 1, 2
    np.savez_compressed(destino / "campo.npz", emb=campo.emb[:, 0].astype(np.float16),
                        labels=campo.labels, refs=campo.refs, groups=campo.groups, split=lado,
                        classes=np.array(nomes))
    registro = {
        "rodada": rodada, "escopo": args.escopo, "quando": _agora(), "modelo": spec.key,
        "cabeca_anterior": file_sha256(spec.head_path),
        "cabeca_candidata": file_sha256(destino / "candidata.head.npz"),
        "dados": {"campo": {"recortes": int(len(campo)),
                            "plantas": len(set(campo.groups.tolist())),
                            "treino": int(len(tr)), "validacao": int(len(va)),
                            "teste": int(len(te)),
                            "por_classe": {nomes[k]: int((campo.labels == k).sum())
                                           for k in range(len(nomes))}},
                  "memoria": len(memoria),
                  "validacao_antigo": 0 if antigo is None else int(len(antigo[1]))},
        "treino": {"melhor_epoca": resultado.epoch, "curva": resultado.curve,
                   "config": resultado.config, "motor": motor},
        "atualizacao": {"tipo": tipo, "muda": TIPOS[tipo].muda,
                        "exige": list(TIPOS[tipo].exige),
                        "invalida": list(TIPOS[tipo].invalida)},
        "portao": laudo.to_dict(), "deriva_borda": deriva,
    }
    (destino / "laudo.json").write_text(json.dumps(registro, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    (destino / "laudo.md").write_text(laudo_md(registro), encoding="utf-8")
    append_ledger(pasta / "historico.jsonl", {
        "evento": "rodada", "quando": registro["quando"], "rodada": rodada,
        "escopo": args.escopo, "decisao": laudo.decision, "motivos": laudo.reasons,
        "cabeca_anterior": registro["cabeca_anterior"],
        "cabeca_candidata": registro["cabeca_candidata"],
        "ganho_campo": (laudo.field or {}).get("acuracia_balanceada", {}).get("diferenca")})
    print(laudo_md(registro))
    return 0 if laudo.decision == "APROVADA" else 2


def laudo_md(r: dict) -> str:
    p = r["portao"]
    linhas = [f"# Rodada {r['rodada']} — {r['modelo']} ({r['escopo']})", "",
              f"**{p['decision']}**", ""]
    for motivo in p["reasons"]:
        linhas.append(f"- {motivo}")
    linhas += ["", "| conjunto | bloqueia | atual | nova | diferença | IC 95% | alarme falso "
               "sadia (atual → nova) | cauda sadia (atual → nova) |",
               "|---|---|---|---|---|---|---|---|"]

    def linha(nome, c, bloqueia):
        a = c["acuracia_balanceada"]
        fa = c.get("alarme_falso_sadia")
        fa_txt = f"{fa['atual']:.1%} → {fa['nova']:.1%}" if fa else "—"
        ca = c.get("cauda_sadia")
        ca_txt = f"{ca['atual']:.1%} → {ca['nova']:.1%}" if ca else "—"
        return (f"| {nome} | {bloqueia} | {a['atual']:.4f} | {a['nova']:.4f} | "
                f"{a['diferenca']:+.4f} | [{a['ic'][0]:+.4f}, {a['ic'][1]:+.4f}] | {fa_txt} | "
                f"{ca_txt} |")

    for nome, c in p["sets"].items():
        linhas.append(linha(nome, c, "sim" if c["bloqueia"] else "só mede"))
    if p.get("pooled"):
        linhas.append(linha("**agregado**", p["pooled"], "não-inferioridade"))
    if p["field"]:
        linhas.append(linha("**campo (teste)**", p["field"], "ganho"))
    d = r["deriva_borda"]
    linhas += ["", "## Perfil da borda (Rover)", ""]
    if d.get("avaliado"):
        for doenca, v in d["por_doenca"].items():
            linhas.append(f"- {doenca}: LLR média em sadia {v['antes']:+.4f} → {v['depois']:+.4f}"
                          f" ({v['subida']:+.4f}; folga {d['folga']:.4f})")
        linhas.append("- **recalibrar o perfil antes de levar ao Rover**" if d["recalibrar"]
                      else "- deriva dentro da folga: o perfil continua valendo")
    else:
        linhas.append(f"- não avaliado ({d.get('motivo', 'sem tabelas de evidência')})")
    t = r["treino"]
    linhas += ["", f"Melhor época: {t['melhor_epoca']} de {t['config']['epocas']} "
               f"(validação: metade campo, metade o que já se sabia).", ""]
    m = t.get("motor") or {}
    if m.get("motor") == MOTOR_DNATY:
        linhas += [f"Motor: plasticidade do dNATY — {m['renascidos']} neurônio(s) renascido(s) "
                   f"{m['por_camada']}, rank efetivo {m['rank_efetivo_antes']} → "
                   f"{m['rank_efetivo_final']}"
                   + (" (sem ganho: a cabeça instalada fica)" if m.get("sem_ganho") else ""),
                   ""]
    a = r.get("atualizacao")
    if a:
        linhas += [f"Atualização de **{a['tipo']}** ({a['muda']}). Exige: "
                   + "; ".join(a["exige"]) + ". Invalida: "
                   + ("; ".join(a["invalida"]) or "nada") + ".", ""]
    return "\n".join(linhas) + "\n"


# ---------------------------------------------------------------------------------------------
# promove / reverte
# ---------------------------------------------------------------------------------------------


def _carimba_perfil(spec, pasta: Path, sha_antiga: str, sha_nova: str, deriva: dict,
                    rodada: str, tipo: str = "cabeca", procedencia_sha256: str | None = None
                    ) -> str:
    """O ``.edge.json`` instalado passa a dizer com que cabeça vale.

    Deriva dentro da folga: o perfil vale para a nova (``head_sha256`` = nova). Senão, ele
    fica preso à antiga — o Rover recusa a nova com ele até o ``fit_edge`` medir de novo.
    Mudança de estrutura prende sempre. ``procedencia_sha256``: o Rover só aceita a cabeça
    com o registro de procedência deste hash ao lado.
    """
    caminho = spec.weights_path.with_suffix(".edge.json")
    if not caminho.exists():
        return "sem perfil da borda instalado"
    perfil = json.loads(caminho.read_text(encoding="utf-8"))
    if not perfil.get("head"):
        return "perfil sem cabeça (CNN): nada a carimbar"
    copia = pasta / "perfis" / f"{sha_antiga}.edge.json"
    copia.parent.mkdir(parents=True, exist_ok=True)
    if not copia.exists():
        copia.write_text(caminho.read_text(encoding="utf-8"), encoding="utf-8")
    vale = deriva.get("avaliado") and not deriva.get("recalibrar") and tipo == "cabeca"
    perfil["head"]["head_sha256"] = sha_nova if vale else perfil["head"].get(
        "head_sha256") or sha_antiga
    perfil.setdefault("measured", {})["continuo"] = {
        "rodada": rodada, "cabeca": sha_nova, "perfil_vale_para_a_nova": bool(vale),
        "deriva": deriva.get("por_doenca"), "tipo": tipo,
        "procedencia_sha256": procedencia_sha256}
    caminho.write_text(json.dumps(perfil, ensure_ascii=False, indent=2), encoding="utf-8")
    return ("perfil da borda vale para a nova cabeça (deriva dentro da folga)" if vale else
            "perfil da borda preso à cabeça anterior: o Rover recusa a nova até recalibrar")


def cmd_promove(args) -> int:
    spec = _spec(args)
    pasta, memoria, suite = _carrega_estado(spec)
    destino = pasta / "rodadas" / args.rodada
    if not (destino / "laudo.json").exists():
        raise SystemExit(f"Rodada {args.rodada} não encontrada em {destino}.")
    laudo = json.loads((destino / "laudo.json").read_text(encoding="utf-8"))
    decisao = laudo["portao"]["decision"]
    if decisao != "APROVADA" and not args.forcar:
        raise SystemExit(f"Rodada {args.rodada} {decisao}: " + "; ".join(
            laudo["portao"]["reasons"]) + ". (--forcar --motivo '...' passa por cima, registrado.)")
    if args.forcar and not args.motivo:
        raise SystemExit("--forcar exige --motivo: fica no histórico.")
    if not (args.aprovador or "").strip():
        raise SystemExit("--aprovador é obrigatório: quem libera a cabeça para produção fica "
                         "na procedência que vai ao Rover.")
    atual = file_sha256(spec.head_path)
    if atual != laudo["cabeca_anterior"] and not args.forcar:
        raise SystemExit("A cabeça instalada não é a que a rodada comparou "
                         f"({atual[:12]}… ≠ {laudo['cabeca_anterior'][:12]}…): rode a rodada de "
                         "novo sobre a cabeça atual.")
    cabecas = pasta / "cabecas"
    cabecas.mkdir(parents=True, exist_ok=True)
    tipo = (laudo.get("atualizacao") or {}).get("tipo") or classifica(
        _cabeca_atual(spec), DiagnosisHead.load(destino / "candidata.head.npz"))
    if not TIPOS[tipo].continuo_pode:
        raise SystemExit(f"Atualização de {tipo}: não se promove pelo ciclo contínuo.")
    shutil.copy2(spec.head_path, cabecas / f"{atual}.head.npz")
    proc_instalada = _caminho_procedencia(spec)
    if proc_instalada.exists():
        shutil.copy2(proc_instalada, cabecas / f"{atual}.provenance.json")
    nova = DiagnosisHead.load(destino / "candidata.head.npz")
    shutil.copy2(destino / "candidata.head.npz", spec.head_path)
    sha_nova = file_sha256(spec.head_path)
    shutil.copy2(spec.head_path, cabecas / f"{sha_nova}.head.npz")

    with np.load(destino / "campo.npz", allow_pickle=False) as z:
        emb, labels = z["emb"].astype(np.float64), z["labels"].astype(np.int64)
        refs, lado = z["refs"], z["split"]
    origem = f"campo:{laudo['escopo']}:{args.rodada}"
    rodadas = [e for e in read_ledger(pasta / "historico.jsonl") if e["evento"] == "promovida"]
    # Treino vai para a memória; a validação passa a escolher a época das próximas rodadas
    # (não treina: fica sem ser decorada); o teste vira regressão (não treina nem escolhe).
    vistos = lado == 0
    memoria = memoria.add(emb[vistos], labels[vistos], nova, origem, refs[vistos],
                          round_index=len(rodadas) + 1)
    suite.add(RegressionSet(origem, emb[lado == 2], labels[lado == 2], True, refs[lado == 2]))
    validacao = _carrega_validacao(pasta, nova.class_names, nova.encoder)
    if (lado == 1).any():
        validacao.add(RegressionSet(f"{origem}:validacao", emb[lado == 1], labels[lado == 1],
                                    False, refs[lado == 1]))
    memoria.save(pasta / "replay.npz")
    suite.save(pasta / "regressao.npz")
    validacao.save(pasta / "validacao.npz")
    dados_laudo = laudo.get("dados") or {}
    # O laudo do adapta traz o bloco "campo"; o do robustez_captura, os números direto.
    campo_dados = dados_laudo.get("campo") or dados_laudo
    registro = procedencia(
        tipo=tipo, modelo=spec.key, encoder=nova.encoder, cabeca=sha_nova, cabeca_base=atual,
        rodada=args.rodada, escopo=laudo["escopo"], quando=_agora(),
        aprovado_por=args.aprovador.strip(), decisao=decisao,
        motivos=laudo["portao"]["reasons"], laudo_sha256=file_sha256(destino / "laudo.json"),
        dados={"campo_sha256": file_sha256(destino / "campo.npz"),
               **{k: campo_dados.get(k) for k in ("plantas", "recortes", "treino", "teste",
                                                   "por_classe") if k in campo_dados},
               "memoria": dados_laudo.get("memoria"), "rodadas_anteriores": len(rodadas)},
        motor=laudo["treino"].get("motor"), forcada=bool(args.forcar), motivo=args.motivo)
    proc_instalada.write_text(json.dumps(registro, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    sha_proc = file_sha256(proc_instalada)
    shutil.copy2(proc_instalada, cabecas / f"{sha_nova}.provenance.json")
    perfil = _carimba_perfil(spec, pasta, atual, sha_nova, laudo["deriva_borda"], args.rodada,
                             tipo, sha_proc)
    pendencias = [
        f"python -m training.fit_conformal --model {spec.key} --field <recortes do campo>: o "
        "conformal da API foi calibrado com a cabeça anterior",
    ]
    if "preso" in perfil:
        pendencias.append(f"python -m training.fit_edge mosaico/fit --model {spec.key}: novo "
                          ".edge.json (com head_sha256 da nova) antes de levar ao Rover")
    append_ledger(pasta / "historico.jsonl", {
        "evento": "promovida", "quando": _agora(), "rodada": args.rodada,
        "escopo": laudo["escopo"], "cabeca_anterior": atual, "cabeca_nova": sha_nova,
        "forcada": bool(args.forcar), "motivo": args.motivo, "perfil_borda": perfil,
        "tipo": tipo, "aprovado_por": registro["aprovado_por"], "procedencia_sha256": sha_proc,
        "pendencias": pendencias})
    print(f"Cabeça {sha_nova[:12]}… instalada em {spec.head_path} (a anterior, {atual[:12]}…, "
          f"em {cabecas}).")
    print(f"Memória: +{int(vistos.sum())} de {origem}; validação: +{int((lado == 1).sum())}; "
          f"regressão: +{int((lado == 2).sum())} ({origem}).")
    print(f"Perfil da borda: {perfil}.")
    print(f"Procedência: {proc_instalada.name} ({sha_proc[:12]}…), liberada por "
          f"{registro['aprovado_por']} — vai ao Rover ao lado da cabeça.")
    print("Pendências antes de pôr em produção:")
    for p in pendencias:
        print(f"  - {p}")
    return 0


def _caminho_procedencia(spec) -> Path:
    """``<pesos>.head.provenance.json``: o Rover procura ``<sha>.head.provenance.json``."""
    return spec.head_path.with_suffix(".provenance.json")


def _cabeca_que_entrou_com(pasta: Path, sha: str) -> dict | None:
    for e in reversed(read_ledger(pasta / "historico.jsonl")):
        if e["evento"] in ("promovida", "revertida") and e.get("cabeca_nova") == sha:
            return e
    return None


def cmd_reverte(args) -> int:
    spec = _spec(args)
    pasta = continuo_dir(spec)
    atual = file_sha256(spec.head_path)
    alvo = args.para
    if alvo is None:
        entrada = _cabeca_que_entrou_com(pasta, atual)
        if entrada is None:
            raise SystemExit("A cabeça instalada não veio de uma promoção registrada: diga "
                             "--para <sha>.")
        alvo = entrada["cabeca_anterior"]
    candidatos = sorted((pasta / "cabecas").glob(f"{alvo}*.head.npz"))
    if len(candidatos) != 1:
        raise SystemExit(f"Cabeça {alvo} não encontrada (ou ambígua) em {pasta / 'cabecas'}.")
    shutil.copy2(spec.head_path, pasta / "cabecas" / f"{atual}.head.npz")
    shutil.copy2(candidatos[0], spec.head_path)
    alvo = file_sha256(spec.head_path)
    # A procedência acompanha a cabeça: a de origem (sem procedência) volta sem nenhuma.
    proc = _caminho_procedencia(spec)
    if proc.exists():
        shutil.copy2(proc, pasta / "cabecas" / f"{atual}.provenance.json")
        proc.unlink()
    guardada = pasta / "cabecas" / f"{alvo}.provenance.json"
    if guardada.exists():
        shutil.copy2(guardada, proc)
    perfil = pasta / "perfis" / f"{alvo}.edge.json"
    restaurado = False
    if perfil.exists():
        shutil.copy2(perfil, spec.weights_path.with_suffix(".edge.json"))
        restaurado = True
    append_ledger(pasta / "historico.jsonl", {
        "evento": "revertida", "quando": _agora(), "cabeca_anterior": atual,
        "cabeca_nova": alvo, "motivo": args.motivo, "perfil_restaurado": restaurado})
    print(f"Cabeça {alvo[:12]}… de volta em {spec.head_path}"
          + (" (com o perfil da borda medido com ela)." if restaurado else "."))
    print("A memória e o portão continuam com as rodadas (são dado validado, não a cabeça).")
    return 0


# ---------------------------------------------------------------------------------------------
# monitora / esquece / historico
# ---------------------------------------------------------------------------------------------


def cmd_monitora(args) -> int:
    spec = _spec(args)
    pasta = continuo_dir(spec)
    atual = _cabeca_atual(spec)
    nomes = atual.class_names
    novo = (load_embeddings(args.campo_embeddings, nomes) if args.campo_embeddings
            else embed_folder(args.campo, nomes, views=1))
    x, y = novo.emb[:, 0], novo.labels
    sha = file_sha256(spec.head_path)
    m_atual = metrics(atual.mlp.probabilities(x), y, nomes)
    entrada = _cabeca_que_entrou_com(pasta, sha)
    comp, estado = None, "sem cabeça anterior para comparar"
    if entrada is not None and (pasta / "cabecas" / f"{entrada['cabeca_anterior']}.head.npz"
                                ).exists():
        anterior = DiagnosisHead.load(pasta / "cabecas" /
                                      f"{entrada['cabeca_anterior']}.head.npz")
        comp = paired_comparison(anterior.mlp.probabilities(x), atual.mlp.probabilities(x), y,
                                 nomes, GateCriteria(seed=args.seed))
        dif = comp["acuracia_balanceada"]
        fa = comp.get("alarme_falso_sadia")
        if dif["ic"][1] < 0 or (fa and fa["ic"][0] > 0):
            estado = "REGRESSAO"
        elif dif["diferenca"] < -args.margem:
            estado = "ATENCAO"
        else:
            estado = "OK"
    append_ledger(pasta / "historico.jsonl", {
        "evento": "monitoramento", "quando": _agora(), "cabeca": sha, "estado": estado,
        "n": int(len(y)), "metricas": m_atual, "contra_anterior": comp})
    fa = m_atual.get("alarme_falso_sadia")
    print(f"Cabeça em produção {sha[:12]}… em {len(y)} recortes novos: acurácia balanceada "
          f"{m_atual['acuracia_balanceada']:.4f}, alarme falso em sadia "
          + ("— (sem folha sadia no lote)" if fa is None else f"{fa:.1%}"))
    if comp is not None:
        d = comp["acuracia_balanceada"]
        print(f"Contra a anterior: {d['diferenca']:+.4f} (IC 95% [{d['ic'][0]:+.4f}, "
              f"{d['ic'][1]:+.4f}]) → {estado}")
    else:
        print(estado)
    if estado == "REGRESSAO":
        print(f"A cabeça em produção é pior que a anterior neste lote: "
              f"python -m training.adapt_campo reverte --model {spec.key}")
        return 3
    return 0


def cmd_esquece(args) -> int:
    spec = _spec(args)
    pasta, memoria, suite = _carrega_estado(spec)
    memoria, tirados_m = memoria.forget(args.fonte)
    tirados_r = suite.forget(args.fonte)
    validacao = _carrega_validacao(pasta, memoria.class_names, memoria.encoder)
    tirados_r += validacao.forget(args.fonte)
    memoria.save(pasta / "replay.npz")
    suite.save(pasta / "regressao.npz")
    validacao.save(pasta / "validacao.npz")
    tirados_c = 0
    for campo in (pasta / "rodadas").glob("*/campo.npz"):
        with np.load(campo, allow_pickle=False) as z:
            dados = {k: z[k] for k in z.files}
        fica = np.array([not str(r).startswith(args.fonte) and args.fonte not in str(r)
                         for r in dados["refs"]], dtype=bool)
        if (~fica).any():
            tirados_c += int((~fica).sum())
            np.savez_compressed(campo, **{k: (v[fica] if k != "classes" else v)
                                          for k, v in dados.items()})
    append_ledger(pasta / "historico.jsonl", {
        "evento": "esquecimento", "quando": _agora(), "fonte": args.fonte,
        "memoria": tirados_m, "regressao": tirados_r, "rodadas": tirados_c})
    print(f"Fora: {tirados_m} da memória, {tirados_r} da regressão e validação, {tirados_c} das "
          "rodadas.")
    print("A cabeça já treinada com esse dado continua com o que aprendeu: para tirar a "
          "influência, reverta para a cabeça anterior à rodada e adapte de novo sem ele.")
    return 0


def cmd_historico(args) -> int:
    spec = _spec(args)
    for e in read_ledger(continuo_dir(spec) / "historico.jsonl"):
        detalhe = {k: v for k, v in e.items() if k in ("rodada", "escopo", "decisao", "estado",
                                                       "ganho_campo", "fonte", "motivo")}
        cabeca = e.get("cabeca_nova") or e.get("cabeca_candidata") or e.get("cabeca") or ""
        print(f"{e['quando']}  {e['evento']:14s} {cabeca[:12]:12s}  "
              f"{json.dumps(detalhe, ensure_ascii=False)}")
    return 0


# ---------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aprendizado contínuo da cabeça com dado de campo.")
    sub = p.add_subparsers(dest="comando", required=True)

    def base(nome: str, ajuda: str) -> argparse.ArgumentParser:
        s = sub.add_parser(nome, help=ajuda)
        s.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
        s.add_argument("--weights", type=Path, default=None)
        s.add_argument("--seed", type=int, default=DEFAULT_SEED)
        return s

    m = base("memoria", "monta a memória de replay e os conjuntos de regressão")
    m.add_argument("--data-dir", type=Path, default=None)
    m.add_argument("--embeddings", type=Path, default=None)
    m.add_argument("--extra", type=Path, action="append", default=[])
    m.add_argument("--extra-embeddings", type=Path, action="append", default=[])
    m.add_argument("--externo", type=Path, action="append", default=[],
                   help="Acervo só para medir (licença só-medição).")
    m.add_argument("--externo-embeddings", type=Path, action="append", default=[])
    m.add_argument("--externo-bloqueia", action="store_true")
    m.add_argument("--por-classe", type=int, default=1000,
                   help="Vagas por classe na memória (768 floats16 cada: 1.000 x 5 classes "
                        "= 7,7 MB). Medido: 300 guarda ~60%% do treino do café e esquece mais.")
    m.add_argument("--validation-split", type=float, default=0.2)
    m.add_argument("--test-split", type=float, default=0.15)
    m.add_argument("--refaz", action="store_true")

    a = base("adapta", "treina uma candidata com o campo e passa pelo portão")
    origem = a.add_mutually_exclusive_group(required=True)
    origem.add_argument("--campo", type=Path, help="<Classe>/<recorte> (training.import_campo).")
    origem.add_argument("--campo-embeddings", type=Path)
    a.add_argument("--escopo", required=True, help="Fazenda, talhão ou região da rodada.")
    a.add_argument("--rodada", default=None)
    a.add_argument("--teste", type=float, default=0.25, help="Fração das plantas no teste.")
    a.add_argument("--validacao", type=float, default=0.15)
    a.add_argument("--epocas", type=int, default=25)
    a.add_argument("--taxa", type=float, default=5e-4)
    a.add_argument("--destilacao", type=float, default=1.0)
    a.add_argument("--ancora", type=float, default=1e-3)
    a.add_argument("--camadas", choices=("todas", "ultima"), default="todas")
    a.add_argument("--sem-replay", action="store_true", help="Só para comparação: esquece.")
    a.add_argument("--motor", choices=(MOTOR_NUMPY, MOTOR_DNATY), default=MOTOR_NUMPY,
                   help="dnaty-plasticidade: renasce os neurônios dormentes (dNATY 2.2) antes "
                        "do ajuste; precisa de torch e dnaty.")
    a.add_argument("--renascer", type=float, default=Plasticidade.fracao,
                   help="Fração máxima de neurônios por camada que renasce (motor dNATY).")
    a.add_argument("--margem", type=float, default=0.02,
                   help="Não-inferioridade no agregado dos conjuntos que bloqueiam.")
    a.add_argument("--margem-classe", type=float, default=0.05)
    a.add_argument("--margem-alarme-falso", type=float, default=0.02)

    pr = base("promove", "instala a candidata aprovada")
    pr.add_argument("--rodada", required=True)
    pr.add_argument("--forcar", action="store_true")
    pr.add_argument("--motivo", default=None)
    pr.add_argument("--aprovador", default=None,
                    help="Quem libera (nome e registro): obrigatório, vai na procedência.")

    r = base("reverte", "volta para a cabeça anterior")
    r.add_argument("--para", default=None, help="SHA-256 (ou início dele) da cabeça.")
    r.add_argument("--motivo", default=None)

    mo = base("monitora", "a cabeça em produção contra a anterior, em dado novo do agrônomo")
    origem = mo.add_mutually_exclusive_group(required=True)
    origem.add_argument("--campo", type=Path)
    origem.add_argument("--campo-embeddings", type=Path)
    mo.add_argument("--margem", type=float, default=0.02)

    e = base("esquece", "tira da memória e do portão o dado de uma origem (consentimento)")
    e.add_argument("--fonte", required=True, help="Começo da origem ou do nome do arquivo.")

    base("historico", "o que aconteceu com a cabeça deste modelo")
    return p.parse_args(argv)


COMANDOS = {"memoria": cmd_memoria, "adapta": cmd_adapta, "promove": cmd_promove,
            "reverte": cmd_reverte, "monitora": cmd_monitora, "esquece": cmd_esquece,
            "historico": cmd_historico}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return COMANDOS[args.comando](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
