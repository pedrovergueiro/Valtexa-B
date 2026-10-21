"""Ajusta e mede a novidade no embedding da torre de visão (:mod:`app.inference.novidade_cabeca`).

    python -m training.fit_novidade_cabeca --treino coffee.npz --extra bracol.npz \\
        --externo coffee_ext.npz --recortes recortes_emb.npz \\
        --saida models/coffee_foliar.novidade.npz --relatorio novidade.json

* ``--treino``: os embeddings do treino da cabeça (``split`` 0 = ajuste, 1 = calibração,
  2 = teste; só a primeira orientação, sem máscaras);
* ``--extra``: acervo que também entrou no treino da cabeça (no café de produção, o BRACOL),
  dividido por grupo (``ids``) em 60/20/20 do mesmo jeito;
* ``--externo``: acervo que a cabeça não viu (só mede: a taxa marcada nele é o custo da troca
  de domínio no tempo do agrônomo);
* ``--recortes``: recortes reais de outro domínio (vídeo do YouTube, pelo motor do Rover), com
  a cena do portão, a classe e a probabilidade da cabeça: mede quanto do **erro confiante**
  a novidade pega.

Os componentes da PCA são fixados antes de olhar os recortes (``--componentes``, 64);
``--varre`` só relata outros valores, para transparência.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.inference.novidade_cabeca import COMPONENTES, PERCENTIL, ReferenciaDeNovidade, ajusta

ENCODER_PADRAO = "google/siglip2-base-patch16-224"


def _carrega(caminho: Path, nomes: tuple[str, ...] | None = None):
    with np.load(caminho, allow_pickle=False) as z:
        classes = tuple(str(c) for c in z["classes"])
        ok = z["valid"].astype(bool) if "valid" in z.files else np.ones(len(z["labels"]), bool)
        if "is_mask" in z.files:
            ok &= ~z["is_mask"].astype(bool)
        emb = z["emb"][:, 0] if z["emb"].ndim == 3 else z["emb"]
        y = z["labels"].astype(np.int64)
        split = z["split"].astype(np.int64) if "split" in z.files else None
        grupos = z["ids"] if "ids" in z.files else (z["paths"] if "paths" in z.files else None)
    if nomes is not None and classes != nomes:
        mapa = np.array([nomes.index(c) for c in classes])
        y = mapa[y]
    return classes, emb[ok].astype(np.float32), y[ok], None if split is None else split[ok], \
        None if grupos is None else np.asarray(grupos)[ok]


def _por_grupo(grupos: np.ndarray) -> np.ndarray:
    """0/1/2 (60/20/20) por grupo, estável entre execuções (hash do id)."""
    h = np.array([int(hashlib.sha256(str(g).encode()).hexdigest()[:8], 16) % 100 for g in grupos])
    return np.where(h < 60, 0, np.where(h < 80, 1, 2))


def _taxa(ref: ReferenciaDeNovidade, emb: np.ndarray, y: np.ndarray | None = None) -> dict:
    if not len(emb):
        return {"n": 0}
    novo = ref.novos(emb)
    saida = {"n": int(len(emb)), "marcados": round(float(novo.mean()), 4)}
    if y is not None:
        saida["por_classe"] = {ref.class_names[k]: round(float(novo[y == k].mean()), 4)
                               for k in range(len(ref.class_names)) if (y == k).any()}
    return saida


def _recortes(ref: ReferenciaDeNovidade, caminho: Path) -> dict:
    with np.load(caminho, allow_pickle=False) as z:
        emb, cena, escopo = z["emb"], z["cena"].astype(str), z["escopo"].astype(bool)
        classe, prob = z["classe"].astype(str), z["prob"].astype(np.float64)
    novo = ref.novos(emb)
    saida = {"n": int(len(emb)), "marcados": round(float(novo.mean()), 4), "por_cena": {}}
    for c, n in Counter(cena.tolist()).most_common():
        m = cena == c
        saida["por_cena"][c] = {"n": int(n), "marcados": round(float(novo[m].mean()), 4)}
    dentro = escopo
    saida["no_escopo_do_portao"] = {"n": int(dentro.sum()),
                                     "marcados": round(float(novo[dentro].mean()), 4)}
    # O erro que motivou isto: recorte que o portão deixa passar e a cabeça chama de doença
    # com confiança alta.
    confiante = dentro & (classe != "Healthy") & (prob >= 0.9)
    saida["doenca_confiante_no_escopo"] = {
        "n": int(confiante.sum()),
        "marcados": round(float(novo[confiante].mean()), 4) if confiante.any() else None,
        "por_classe": {c: round(float(novo[confiante & (classe == c)].mean()), 4)
                       for c in sorted(set(classe[confiante].tolist()))}}
    return saida


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--treino", type=Path, required=True)
    p.add_argument("--extra", type=Path, action="append", default=[])
    p.add_argument("--externo", type=Path, action="append", default=[])
    p.add_argument("--recortes", type=Path, default=None)
    p.add_argument("--encoder", default=ENCODER_PADRAO)
    p.add_argument("--componentes", type=int, default=COMPONENTES)
    p.add_argument("--percentil", type=float, default=PERCENTIL)
    p.add_argument("--varre", type=int, nargs="*", default=[])
    p.add_argument("--saida", type=Path, default=None)
    p.add_argument("--relatorio", type=Path, default=None)
    args = p.parse_args(argv)

    nomes, emb, y, split, _g = _carrega(args.treino)
    if split is None:
        raise SystemExit(f"{args.treino}: sem 'split' (0 ajuste, 1 calibração, 2 teste).")
    partes = {k: ([emb[split == k]], [y[split == k]]) for k in (0, 1, 2)}
    nomes_extra = []
    for caminho in args.extra:
        _c, e, ye, _s, g = _carrega(caminho, nomes)
        lado = _por_grupo(g if g is not None else np.arange(len(e)))
        for k in (0, 1, 2):
            partes[k][0].append(e[lado == k])
            partes[k][1].append(ye[lado == k])
        nomes_extra.append(caminho.name)
    junta = {k: (np.vstack(v[0]), np.concatenate(v[1])) for k, v in partes.items()}

    def roda(componentes: int) -> tuple[ReferenciaDeNovidade, dict]:
        ref = ajusta(*junta[0], *junta[1], nomes, args.encoder, componentes=componentes,
                     percentil=args.percentil)
        r = {"componentes": componentes, "medido": ref.medido,
             "teste_do_dominio": _taxa(ref, *junta[2])}
        for caminho in args.externo:
            _c, e, ye, _s, _g2 = _carrega(caminho, nomes)
            r[f"externo:{caminho.name}"] = _taxa(ref, e, ye)
        if args.recortes:
            r["recortes"] = _recortes(ref, args.recortes)
        return ref, r

    ref, relatorio = roda(args.componentes)
    relatorio = {"escolhido": relatorio, "extra": nomes_extra, "percentil": args.percentil,
                 "varredura": [roda(k)[1] for k in args.varre if k != args.componentes]}
    print(json.dumps(relatorio, ensure_ascii=False, indent=1))
    if args.saida:
        ref.save(args.saida)
        print(f"Referência em {args.saida}")
    if args.relatorio:
        args.relatorio.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
