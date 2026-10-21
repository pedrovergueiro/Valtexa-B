"""Cabeça robusta à captura ruim: a foto pior não pode virar "ferrugem".

Medido em 16/10/2026 (vídeos de lavoura e as próprias folhas de teste degradadas): quando a
foto piora, a cabeça do café não fica em dúvida — ela responde **Rust**. Cercóspora limpa
acertava 98%; reduzida a 160 px e recomprimida, 35% (25 de 54 viraram ferrugem). Bicho-mineiro
com borrão de movimento: 98% → 40%. Em vídeo de lavoura, 89% dos quadros aceitos pelo portão
saíam ferrugem, inclusive planta sadia e cereja madura.

Este script ensina a cabeça a ler a foto ruim, sem desaprender a boa, com a mesma máquina do
aprendizado contínuo (:mod:`app.inference.continual`):

1. ``embeddings`` (GPU, um job): cada foto do treino ganha ``--variantes`` versões degradadas
   como quadro de vídeo — resolução baixa e JPEG forte, borrão de movimento, desfoque,
   exposição, ruído (:func:`degrada`; sorteio determinístico por arquivo) — e passa pela torre
   do SigLIP 2 nas quatro orientações do treino. Grava também o BRACOL com o corte **exato**
   da cabeça de produção: ``--extra-holdout`` é a pasta que o ``fit_heads --holdout`` gravou,
   com as fotos que ela **não** treinou;
2. ``rodada`` (CPU): ajuste fino da cabeça em produção com as degradadas do **treino**,
   replay do limpo, destilação e âncora; a época sai da validação limpa + validação degradada;
   o portão compara com a cabeça atual no teste **limpo** (não pode piorar) e no teste
   **degradado** (tem de ganhar). Grava a rodada em ``models/_continuo/<pesos>/rodadas/``, no
   formato do ``adapt_campo`` — ``promove``, ``reverte`` e o histórico valem para ela.

    python -m training.adapt_campo memoria --model coffee-foliar \\
        --embeddings coffee.npz --extra-embeddings bracol_alinhado.npz \\
        --externo-embeddings coffee_ext.npz
    python -m training.robustez_captura embeddings --model coffee-foliar \\
        --embeddings coffee.npz --extra-dir data/coffee-bracol \\
        --extra-holdout data/coffee-bracol-holdout --extra-embeddings bracol.npz \\
        --alinhado bracol_alinhado.npz --saida degradadas.npz
    python -m training.robustez_captura rodada --model coffee-foliar --degradadas degradadas.npz
    python -m training.adapt_campo promove --model coffee-foliar --rodada <rodada> \\
        --aprovador "<quem libera>"

O que **não** se usa: quadros de vídeo da internet (só medem; ver MODELS.md). As degradações
são sintéticas e sobre fotos de licença compatível — as do próprio dataset do modelo.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.inference.continual import (
    AdaptConfig,
    GateCriteria,
    RegressionSet,
    adapt,
    append_ledger,
    compare,
    edge_drift,
    file_sha256,
)
from app.registry import MODEL_SPECS
from training.adapt_campo import (
    DUPLICATE_COSINE,
    _cabeca_atual,
    _carrega_estado,
    _carrega_validacao,
    _perfil_da_borda,
    _spec,
    laudo_md,
    load_embeddings,
)

TIPOS = ("resolucao", "movimento", "desfoque", "exposicao", "ruido")
LADO_TRABALHO = 640
"""As degradações rodam com o lado maior em 640 px — o tamanho de um quadro de vídeo 720p."""


def _rng(nome: str, variante: int) -> np.random.Generator:
    semente = int(hashlib.sha256(f"{nome}#{variante}".encode()).hexdigest()[:12], 16)
    return np.random.default_rng(semente)


def _jpeg(img: Image.Image, qualidade: int) -> Image.Image:
    saida = io.BytesIO()
    img.save(saida, "JPEG", quality=int(qualidade))
    return Image.open(io.BytesIO(saida.getvalue())).convert("RGB")


def _movimento(img: Image.Image, comprimento: int, angulo: int) -> Image.Image:
    """Borrão de movimento: média de cópias deslocadas ao longo da direção."""
    a = np.asarray(img, dtype=np.float32)
    dx, dy = {0: (1, 0), 45: (1, 1), 90: (0, 1), 135: (-1, 1)}[angulo]
    soma = np.zeros_like(a)
    for k in range(comprimento):
        d = k - comprimento // 2
        soma += np.roll(np.roll(a, d * dy, axis=0), d * dx, axis=1)
    return Image.fromarray(np.clip(soma / comprimento, 0, 255).astype(np.uint8))


def degrada(img: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Uma ou duas degradações sorteadas e, no fim, um JPEG de vídeo. Devolve a imagem e o nome."""
    img = img.convert("RGB")
    img.thumbnail((LADO_TRABALHO, LADO_TRABALHO))
    feitas = []
    for tipo in rng.choice(TIPOS, size=int(rng.integers(1, 3)), replace=False):
        if tipo == "resolucao":
            lado = int(rng.integers(128, 360))
            pequena = img.copy()
            pequena.thumbnail((lado, lado))
            img = _jpeg(pequena, int(rng.integers(12, 50)))
        elif tipo == "movimento":
            img = _movimento(img, int(rng.integers(5, 16)), int(rng.choice([0, 45, 90, 135])))
        elif tipo == "desfoque":
            img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(1.0, 2.5))))
        elif tipo == "exposicao":
            img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.4, 1.7)))
            img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.6, 1.3)))
            img = ImageEnhance.Color(img).enhance(float(rng.uniform(0.6, 1.4)))
        else:
            a = np.asarray(img, dtype=np.float32)
            a += rng.normal(0.0, float(rng.uniform(5.0, 20.0)), a.shape)
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        feitas.append(str(tipo))
    return _jpeg(img, int(rng.integers(25, 80))), "+".join(feitas)


# ---------------------------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------------------------


def alinha_extra(extra_npz: Path, extra_dir: Path, holdout_dir: Path, principal: Path,
                 class_names: tuple[str, ...], seed: int, saida: Path) -> dict[str, int]:
    """O ``--extra`` com o corte da cabeça de produção: holdout = validação + teste.

    O holdout do ``fit_heads`` junta validação e teste do extra; aqui ele é dividido por um
    sorteio fixo por arquivo (40% validação, 60% teste). O resto do extra é treino, menos as
    quase cópias do principal (cosseno > 0,97), que ficam fora de tudo.
    """
    z = np.load(extra_npz, allow_pickle=False)
    ids = [str(i) for i in z["ids"]] if "ids" in z.files else None
    if ids is None:
        raise SystemExit(f"{extra_npz}: sem 'ids' para casar com {extra_dir}.")
    holdout = {p.name for p in holdout_dir.rglob("*.jpg")}
    arquivos = {p.name: p for p in extra_dir.rglob("*.jpg")}
    prefixo = extra_dir.name.split("-")[-1]  # data/coffee-bracol -> bracol_<id>.jpg
    nomes = [f"{prefixo}_{i}.jpg" for i in ids]
    faltam = sum(n not in arquivos for n in nomes)
    if faltam:
        raise SystemExit(f"{faltam} fotos de {extra_npz} sem arquivo em {extra_dir}.")
    classes = tuple(str(c) for c in z["classes"])
    mapa = np.array([class_names.index(c) for c in classes])
    rotulos = mapa[z["labels"].astype(np.int64)]
    ok = z["valid"].astype(bool) if "valid" in z.files else np.ones(len(ids), bool)
    p = np.load(principal, allow_pickle=False)
    conhecidas = p["emb"][:, 0]
    conhecidas = conhecidas / np.linalg.norm(conhecidas, axis=1, keepdims=True)
    e0 = z["emb"][:, 0] / np.linalg.norm(z["emb"][:, 0], axis=1, keepdims=True)
    copia = (e0 @ conhecidas.T).max(axis=1) > DUPLICATE_COSINE
    split = np.zeros(len(ids), dtype=np.int64)
    for k, nome in enumerate(nomes):
        if nome in holdout:
            split[k] = 1 if _rng(nome, seed).random() < 0.4 else 2
    manter = ok & ~(copia & (split == 0))
    caminhos = np.array([str(arquivos[n]) for n in nomes])
    np.savez_compressed(saida, emb=z["emb"][manter], labels=rotulos[manter],
                        classes=np.array(class_names), paths=caminhos[manter],
                        split=split[manter])
    return {"treino": int(((split == 0) & manter).sum()),
            "validacao": int((split == 1).sum()), "teste": int((split == 2).sum()),
            "quase_copias_fora": int((copia & (split == 0)).sum())}


def cmd_embeddings(args) -> int:
    from app.framing import IMG_SIZE, REFERENCE_NAME, FramingReference, vision_encoder
    from app.inference.preprocess import load_image_from_bytes
    from training.fit_heads import four_views

    spec = _spec(args)
    cabeca = _cabeca_atual(spec)
    nomes = cabeca.class_names
    fontes = [load_embeddings(args.embeddings, nomes)]
    if args.extra_embeddings:
        print("BRACOL alinhado ao corte de produção:",
              alinha_extra(args.extra_embeddings, args.extra_dir, args.extra_holdout,
                           args.embeddings, nomes, args.seed, args.alinhado), flush=True)
        fontes.append(load_embeddings(args.alinhado, nomes))
    reference = FramingReference.load(spec.weights_path.parent / REFERENCE_NAME)
    encoder = vision_encoder(reference.model_id)
    if reference.model_id != cabeca.encoder:
        raise SystemExit(f"Torre {reference.model_id} ≠ a da cabeça ({cabeca.encoder}).")
    embs, labels, refs, splits, tipos = [], [], [], [], []
    inicio = time.perf_counter()
    for fonte in fontes:
        for k in range(len(fonte)):
            caminho = Path(str(fonte.refs[k]))
            try:
                img = load_image_from_bytes(caminho.read_bytes(), caminho.name)
            except Exception as exc:  # foto ilegível no dataset: fica de fora, e avisa
                print(f"\n  {caminho}: {exc}", flush=True)
                continue
            for v in range(args.variantes):
                ruim, tipo = degrada(img, _rng(caminho.name, v))
                vistas = four_views(ruim, IMG_SIZE)
                embs.append(np.asarray(encoder(vistas), dtype=np.float16))
                labels.append(int(fonte.labels[k]))
                refs.append(f"{caminho.name}#v{v}")
                splits.append(int(fonte.split[k]))
                tipos.append(tipo)
            if len(refs) % 200 < args.variantes:
                print(f"\r  {len(refs)} degradadas ({time.perf_counter() - inicio:.0f}s)",
                      end="", flush=True)
    print()
    grupos = np.array([r.split("#")[0] for r in refs])
    np.savez_compressed(args.saida, emb=np.stack(embs), labels=np.array(labels),
                        classes=np.array(nomes), paths=np.array(refs), groups=grupos,
                        split=np.array(splits), tipos=np.array(tipos))
    print(f"{len(refs)} embeddings degradados em {args.saida} "
          f"({time.perf_counter() - inicio:.0f}s) · por corte "
          f"{dict(zip(*np.unique(splits, return_counts=True), strict=True))}")
    return 0


# ---------------------------------------------------------------------------------------------
# rodada
# ---------------------------------------------------------------------------------------------


TOL_VALIDACAO = {"acuracia": 0.01, "classe": 0.03, "cauda": 0.005, "alarme": 0.01}
"""Admissão na validação limpa: metade (ou menos) das margens do portão, como a seleção de
época do ``adapt`` — o portão mede em outro conjunto."""


def selecao_na_validacao(atual, nova, limpa: tuple, degradada: tuple) -> dict:
    """A dose (suavização, destilação) sai da validação, nunca dos conjuntos do portão.

    Admissível: na validação limpa, acurácia balanceada sem cair mais de 1 ponto, nenhuma
    classe mais de 3, cauda de doença em sadia sem subir mais de 0,5 ponto e alarme falso sem
    subir mais de 1. Entre as admissíveis, vence a melhor na validação degradada. A rodada 2
    de 16/10/2026 (suavização 0,1 fixa) mostrou o porquê: curou a cauda e o alarme falso, mas
    perdeu 5 pontos de ferrugem — o portão recusou.
    """
    from app.inference.continual import HEALTHY, balanced_accuracy

    nomes = atual.class_names
    h = nomes.index(HEALTHY)
    x, y = limpa
    pa, pn = atual.mlp.probabilities(x), nova.mlp.probabilities(x)
    xa, ya = degradada
    motivos = []
    ba = balanced_accuracy(y, pa.argmax(1), len(nomes))
    bn = balanced_accuracy(y, pn.argmax(1), len(nomes))
    if bn < ba - TOL_VALIDACAO["acuracia"]:
        motivos.append(f"acurácia limpa {ba:.3f}→{bn:.3f}")
    for k, nome in enumerate(nomes):
        s = y == k
        if s.sum() >= 30:
            ra, rn = (pa[s].argmax(1) == k).mean(), (pn[s].argmax(1) == k).mean()
            if rn < ra - TOL_VALIDACAO["classe"]:
                motivos.append(f"{nome} {ra:.3f}→{rn:.3f}")
    sad = y == h

    def cauda(p):
        return float((np.delete(p[sad], h, axis=1).max(1) > 0.9).mean())

    def alarme(p):
        return float((p[sad].argmax(1) != h).mean())

    if cauda(pn) > cauda(pa) + TOL_VALIDACAO["cauda"]:
        motivos.append(f"cauda sadia {cauda(pa):.3f}→{cauda(pn):.3f}")
    if alarme(pn) > alarme(pa) + TOL_VALIDACAO["alarme"]:
        motivos.append(f"alarme falso {alarme(pa):.3f}→{alarme(pn):.3f}")
    deg = balanced_accuracy(ya, nova.mlp.probabilities(xa).argmax(1), len(nomes))
    return {"admissivel": not motivos, "motivos": motivos, "limpa": round(bn, 4),
            "degradada": round(deg, 4), "cauda": round(cauda(pn), 4),
            "resumo": f"validação limpa {bn:.3f} (atual {ba:.3f}) · degradada {deg:.3f} · "
                      f"cauda {cauda(pn):.3f}"}


def cmd_rodada(args) -> int:
    spec = _spec(args)
    cabeca = _cabeca_atual(spec)
    pasta, memoria, suite = _carrega_estado(spec)
    validacao = _carrega_validacao(pasta, cabeca.class_names, cabeca.encoder)
    if not validacao.sets:
        raise SystemExit("Sem validação do que já se sabia (validacao.npz): refaça a memória.")
    ruins = load_embeddings(args.degradadas, cabeca.class_names)
    if ruins.split is None:
        raise SystemExit(f"{args.degradadas} sem 'split'.")
    tr, va, te = (np.flatnonzero(ruins.split == s) for s in (0, 1, 2))
    rodada = args.rodada or datetime.now(UTC).strftime("robustez-%Y%m%dT%H%M%SZ")
    destino = pasta / "rodadas" / rodada
    if destino.exists():
        raise SystemExit(f"A rodada {rodada} já existe em {destino}.")
    antigo = (np.vstack([s.embeddings for s in validacao.sets]),
              np.concatenate([s.labels for s in validacao.sets]))
    v = ruins.emb.shape[1]
    print(f"Degradadas: treino {len(tr)} x{v} · validação {len(va)} · teste {len(te)} · "
          f"memória {len(memoria)} · validação limpa {len(antigo[1])}", flush=True)
    candidatas = []
    for suav in args.suavizacao:
        for dest in args.destilacao:
            cfg = AdaptConfig(epocas=args.epocas, taxa=args.taxa, destilacao=dest,
                              ancora=args.ancora, suavizacao=suav, seed=args.seed)
            inicio = time.perf_counter()
            res = adapt(cabeca, ruins.emb[tr].reshape(-1, ruins.emb.shape[-1]),
                        np.repeat(ruins.labels[tr], v), memoria, ruins.emb[va, 0],
                        ruins.labels[va], cfg, antigo)
            sel = selecao_na_validacao(cabeca, res.head, antigo,
                                       (ruins.emb[va, 0], ruins.labels[va]))
            candidatas.append((res, sel))
            print(f"  suavização {suav} · destilação {dest}: época {res.epoch} "
                  f"({time.perf_counter() - inicio:.0f}s) · {sel['resumo']}"
                  + ("" if sel["admissivel"] else f" · fora: {'; '.join(sel['motivos'])}"),
                  flush=True)
    admissiveis = [c for c in candidatas if c[1]["admissivel"]]
    resultado, escolha = max(admissiveis or candidatas,
                             key=lambda c: c[1]["degradada"] if admissiveis else c[1]["limpa"])
    print(f"Escolhida na validação: suavização {resultado.config['suavizacao']}, destilação "
          f"{resultado.config['destilacao']}" + ("" if admissiveis else
                                                " (nenhuma admissível: a de melhor validação "
                                                "limpa vai ao portão, que deve recusar)"),
          flush=True)
    escopo = "robustez-captura"
    teste = RegressionSet(f"campo:{escopo}:{rodada}", ruins.emb[te, 0], ruins.labels[te], True,
                          ruins.refs[te])
    laudo = compare(cabeca, resultado.head, suite, teste,
                    GateCriteria(seed=args.seed))
    perfil = _perfil_da_borda(spec)
    deriva = {"avaliado": False, "motivo": "sem perfil da borda com evidência"}
    if perfil is not None:
        sets = [s for s in suite.sets if s.gate] + [teste]
        deriva = edge_drift(cabeca, resultado.head, np.vstack([s.embeddings for s in sets]),
                            np.concatenate([s.labels for s in sets]), perfil)
    destino.mkdir(parents=True)
    resultado.head.save(destino / "candidata.head.npz")
    np.savez_compressed(destino / "campo.npz", emb=ruins.emb[:, 0].astype(np.float16),
                        labels=ruins.labels, refs=ruins.refs, groups=ruins.groups,
                        split=ruins.split, classes=np.array(cabeca.class_names))
    registro = {
        "rodada": rodada, "escopo": escopo, "quando": datetime.now(UTC).isoformat(
            timespec="seconds"), "modelo": spec.key,
        "cabeca_anterior": file_sha256(spec.head_path),
        "cabeca_candidata": file_sha256(destino / "candidata.head.npz"),
        "dados": {"degradadas": str(args.degradadas), "treino": int(len(tr)),
                  "validacao": int(len(va)), "teste": int(len(te)), "memoria": len(memoria),
                  "validacao_antigo": int(len(antigo[1]))},
        "treino": {"melhor_epoca": resultado.epoch, "curva": resultado.curve,
                   "config": resultado.config,
                   "selecao": [{"config": {k: c.config[k] for k in ("suavizacao", "destilacao")},
                                "epoca": c.epoch, **{k: v for k, v in sel.items()
                                                     if k != "resumo"}}
                               for c, sel in candidatas]},
        "portao": laudo.to_dict(), "deriva_borda": deriva,
    }
    (destino / "laudo.json").write_text(json.dumps(registro, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    (destino / "laudo.md").write_text(laudo_md(registro), encoding="utf-8")
    append_ledger(pasta / "historico.jsonl", {
        "evento": "rodada", "quando": registro["quando"], "rodada": rodada, "escopo": escopo,
        "decisao": laudo.decision, "motivos": laudo.reasons,
        "cabeca_anterior": registro["cabeca_anterior"],
        "cabeca_candidata": registro["cabeca_candidata"],
        "ganho_campo": (laudo.field or {}).get("acuracia_balanceada", {}).get("diferenca")})
    print(laudo_md(registro))
    return 0 if laudo.decision == "APROVADA" else 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cabeça robusta à captura ruim.")
    sub = p.add_subparsers(dest="comando", required=True)

    def base(nome: str, ajuda: str) -> argparse.ArgumentParser:
        s = sub.add_parser(nome, help=ajuda)
        s.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
        s.add_argument("--weights", type=Path, default=None)
        s.add_argument("--seed", type=int, default=42)
        return s

    e = base("embeddings", "fotos degradadas pela torre de visão (GPU)")
    e.add_argument("--embeddings", type=Path, required=True,
                   help="Embeddings limpos do dataset do modelo, com 'split' e 'paths'.")
    e.add_argument("--extra-embeddings", type=Path, default=None)
    e.add_argument("--extra-dir", type=Path, default=None)
    e.add_argument("--extra-holdout", type=Path, default=None,
                   help="A pasta do fit_heads --holdout: o que a cabeça de produção não treinou.")
    e.add_argument("--alinhado", type=Path, default=None,
                   help="Onde gravar o extra limpo com o corte de produção.")
    e.add_argument("--variantes", type=int, default=2)
    e.add_argument("--saida", type=Path, required=True)

    r = base("rodada", "adapta, passa pelo portão e grava a rodada (CPU)")
    r.add_argument("--degradadas", type=Path, required=True)
    r.add_argument("--rodada", default=None)
    r.add_argument("--epocas", type=int, default=25)
    r.add_argument("--taxa", type=float, default=5e-4)
    r.add_argument("--destilacao", type=float, nargs="+", default=[1.0, 3.0])
    r.add_argument("--ancora", type=float, default=1e-3)
    r.add_argument("--suavizacao", type=float, nargs="+", default=[0.0, 0.03, 0.06],
                   help="Suavização de rótulo: segura a cauda das sadias (ver GateCriteria). "
                        "Com vários valores (e de --destilacao), a dose sai da validação.")
    args = p.parse_args(argv)
    if args.comando == "embeddings" and args.extra_embeddings and not (
            args.extra_dir and args.extra_holdout and args.alinhado):
        p.error("--extra-embeddings pede --extra-dir, --extra-holdout e --alinhado.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return {"embeddings": cmd_embeddings, "rodada": cmd_rodada}[args.comando](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
