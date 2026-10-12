"""Calibra a Edge AI do Rover para um modelo: ``<pesos>.edge.json`` (+ ``.categories.json``).

    # 1) Probabilidades do ONNX sozinho em cada recorte, planta a planta (CPU basta)
    python -m training.fit_edge collect --model banana-foliar \\
        --data-dir data/banana-foliar-campo \\
        --plantseg ~/valtexa-raw/plantseg/plantseg/images/test \\
        --out medicoes/edge/banana-foliar.jsonl
    python -m training.fit_edge collect --model citrus-foliar \\
        --data-dir data/citrus-foliar-campo --data-dir-classes Healthy \\
        --plantseg ~/valtexa-raw/plantseg/plantseg/images/test \\
        --out medicoes/edge/citrus-foliar.jsonl
    # Café: pelo caminho do Rover com cabeça (SigLIP + head.npz, GPU), mais o BRACOL separado
    python -m training.fit_edge collect --model coffee-foliar \\
        --plantseg ~/valtexa-raw/plantseg/plantseg/images/test \\
        --extra-dir data/coffee-bracol-holdout --out medicoes/edge/coffee-foliar.jsonl
    # 2) Limiares por validação cruzada, gravados ao lado dos pesos
    python -m training.fit_edge fit --model banana-foliar --frames medicoes/edge/banana-foliar.jsonl
    # 3) A fusão por planta dos recortes que o Rover manda (<pesos>.edge-fusion.json)
    python -m training.fit_edge fusao --model banana-foliar \\
        --frames medicoes/edge/banana-foliar.jsonl
    # 4) Em campo: o registro do Rover na IA (FIELDS_TILE_LOG=true) + o rótulo do agrônomo
    python -m training.fit_edge fit --model coffee-foliar \\
        --frames fields/plantas/registro/T01.jsonl --labels rotulos.csv

Por que existe
--------------
No Rover (``VALTEXA-ROVER``) a Edge AI decide, quadro a quadro e sem internet, o que
guardar em qualidade original e mandar primeiro (doença, praga, anomalia) e o que vira
só metadado ou imagem reduzida. Ela roda o ONNX do modelo sem o portão de
enquadramento. Nos modelos com cabeça sobre o embedding (o café), roda também a torre
de visão do SigLIP 2 em ONNX com a cabeça (``collect --head``): no café a CNN sozinha
acerta 6,5% num acervo externo. Os limiares precisam ser medidos com esse mesmo caminho.

O que se mede (fixado antes de olhar)
-------------------------------------
Plantas de ``training.fit_fusion.build_plants`` (as mesmas da calibração da fusão): três
fotos por planta, uma por andar de câmera, recortes de 150 mm a 270 mm.

* Um recorte é **achado** quando a maior probabilidade de doença/praga ≥ ``t``; o quadro
  é **evento** com pelo menos ``k`` achados.
* Uma planta doente é **pega** quando ao menos um dos seus quadros é evento — é o que
  garante que a nuvem receba o original para diagnosticar e fundir.
* Custo: a fração de **quadros sadios** que viram evento, porque cada um sobe em
  qualidade original. Meta padrão: ≤ 5 % (``--orcamento``).
* Escolha: o par (``t``, ``k``) de maior fração de plantas doentes pegas, com o custo
  na meta, nas partes de treino; números fora da amostra numa validação cruzada por
  planta. ``near_miss`` (baixa confiança, original guardado para validação) e
  ``anomalia`` saem de quantis dos quadros conhecidos (ver ``fit``).

**Regra por evidência (a que o Rover usa quando o perfil a traz).** Medido em 07/10/2026
na banana: o modelo põe probabilidade de doença ≥ 0,99 em recortes de folha sadia com
tanta frequência que a regra de contagem acima não cabe no orçamento (33% dos quadros
sadios viravam evento fora da amostra, no canto da grade). A regra por evidência é a da
fusão por planta, num quadro só: cada recorte vira a log-razão de verossimilhança da
faixa da sua probabilidade (``EVIDENCE_EDGES``, mais fina perto de 1, onde as plantas se
separam), limitada a ``±clip``; o escore do quadro é a maior, entre as doenças, da média
nos recortes vezes o tamanho efetivo ``n / (1 + (n - 1) rho)`` (o desconto de correlação
da fusão). Evento: escore > ``event_evidence``, o quantil ``1 - orçamento`` dos quadros
sadios. Tabelas, ``clip`` e limiar são refeitos em cada parte de treino; a parte de fora
só mede.

Fusão dos recortes do Rover (``fusao``)
---------------------------------------
O Rover manda, em cada resultado, as probabilidades de todos os recortes (o quadro normal
sobe reduzido e a nuvem não consegue rodar o modelo de novo nele). A nuvem funde esses
recortes por planta (``POST /fields/{id}/frames``). A calibração da fusão da nuvem
(``training.fit_fusion``) foi medida só nos recortes que o portão de enquadramento
aceitou; no Rover todo recorte conta. ``fusao`` converte estes mesmos quadros para o
formato do ``fit_fusion`` (``in_scope`` nulo: todo recorte conta) e roda o mesmo ajuste,
gravando ``<pesos>.edge-fusion.json``.

Limite: são fotos de folha agrupadas, não o Rover no pomar; as sadias dos citros são
folhas sobre papel. A validação de campo deve remedir com o registro do Rover.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.registry import MODEL_SPECS, get_spec
from training import DEFAULT_SEED

HEALTHY = "Healthy"
PROFILE_VERSION = 1
T_GRID = tuple(round(x, 3) for x in np.arange(0.50, 0.995, 0.01))
K_GRID = (1, 2, 3)
PEST_WORDS = ("miner", "mite", "praga", "pest", "insect")
EVIDENCE_EDGES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 0.99,
                  0.995, 0.999, 1.0)
CLIP_GRID = (0.5, 1.0, 2.0, 3.0)
RHO_GRID = (0.02, 0.1, 0.3, 1.0)


# ---------------------------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------------------------


def collect(args: argparse.Namespace) -> None:
    import onnxruntime as ort

    from app.inference.preprocess import load_image_from_bytes, preprocess_image
    from app.rover import plan_tiles
    from training.fit_fusion import CAMERA_HEIGHTS_MM, DISTANCE_MM, _as_frame, build_plants

    spec = get_spec(args.model)
    onnx_path = spec.weights_path.with_suffix(".onnx")
    if not onnx_path.exists():
        raise SystemExit(f"{onnx_path} não existe: exporte com training.export_onnx.")
    labels = tuple(json.loads(spec.labels_path.read_text(encoding="utf-8")))
    opcoes = ort.SessionOptions()
    opcoes.intra_op_num_threads = args.threads
    sessao = ort.InferenceSession(str(onnx_path), opcoes, providers=["CPUExecutionProvider"])
    entrada = sessao.get_inputs()[0].name
    tamanho = int(sessao.get_inputs()[0].shape[1])
    cabeca = _head_for(spec, args.head, labels)
    args.data_dir = args.data_dir or spec.data_dir
    args.camera_heights = list(CAMERA_HEIGHTS_MM)
    plantas = build_plants(args, labels, spec.crop)
    for extra in args.extra_dir or []:
        plantas += extra_plants(extra, labels, len(CAMERA_HEIGHTS_MM), args.max_plants,
                                args.max_healthy)
    print(f"{spec.key}: {len(plantas)} plantas; caminho: {cabeca[2] if cabeca else 'cnn'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    parcial = args.out.with_suffix(".parcial")
    inicio = time.perf_counter()
    with parcial.open("w", encoding="utf-8") as fh:
        for n, (pid, label, grupo, fotos) in enumerate(plantas):
            for foto, altura in zip(fotos, CAMERA_HEIGHTS_MM, strict=True):
                imagem = load_image_from_bytes(_as_frame(foto), foto.name)
                base = {"model_key": spec.key, "labels": list(labels), "plant": pid,
                        "label": label, "group": grupo, "camera_height_mm": altura,
                        "width": imagem.size[0], "height": imagem.size[1]}
                try:
                    recortes = plan_tiles(imagem.size, DISTANCE_MM, altura, crop=spec.crop)
                except ValueError as exc:
                    fh.write(json.dumps({**base, "refused": str(exc), "probs": []}) + "\n")
                    continue
                pedacos = [imagem.crop(t.box) for t in recortes]
                head, encode, caminho = cabeca or (None, None, "cnn")
                cnn = None
                if head is None or head.mix.get("cnn", 0.0) > 0:
                    lote = np.concatenate([preprocess_image(c, tamanho) for c in pedacos])
                    cnn = np.concatenate([sessao.run(None, {entrada: lote[i:i + 32]})[0]
                                          for i in range(0, len(lote), 32)])
                if head is None:
                    probs = cnn
                else:
                    emb = np.concatenate([encode(pedacos[i:i + 64])
                                          for i in range(0, len(pedacos), 64)])
                    if cnn is None:  # mistura sem a CNN: o valor não entra na conta
                        cnn = np.full((len(pedacos), len(labels)), 1.0 / len(labels))
                    probs = head.combine(cnn, emb)
                fh.write(json.dumps({**base, "path": caminho,
                                     "probs": np.round(probs, 5).tolist(),
                                     "heights": [t.center_height_mm for t in recortes]}) + "\n")
            if n % 50 == 0:
                print(f"  {n + 1}/{len(plantas)} em {time.perf_counter() - inicio:.0f}s",
                      flush=True)
    parcial.replace(args.out)
    print(f"Gravado em {args.out}")


def _head_for(spec: Any, quando: str, labels: tuple[str, ...]):
    """``(cabeça, codificador, caminho)`` quando o modelo tem cabeça sobre o embedding.

    O Rover roda a torre de visão pelo ONNX com o redimensionamento do Pillow; aqui a
    torre do ``transformers`` (GPU, se houver) faz a mesma conta — cosseno 1,00000 entre
    as duas, medido em 01/10/2026 (:func:`app.framing.onnx_vision_encoder`).
    """
    if quando == "nao" or (quando == "auto" and not spec.head_path.exists()):
        return None
    from app.framing import vision_encoder
    from app.inference.heads import DiagnosisHead

    head = DiagnosisHead.load(spec.head_path)
    if head.class_names != labels:
        raise SystemExit(f"{spec.head_path.name}: classes {head.class_names} != {labels}.")
    return head, vision_encoder(head.encoder), f"cabeca:{head.encoder}"


def extra_plants(root: Path, labels: tuple[str, ...], per_plant: int, max_plants: int,
                 max_healthy: int) -> list[tuple[str, str, str, list[Path]]]:
    """Trios de fotos de um acervo à parte (todas as fotos; a fonte é o nome da pasta).

    Para acervos que o modelo **não** viu no treino — no café, o BRACOL separado da
    cabeça (``data/coffee-bracol-holdout``).
    """
    plantas = []
    for nome in labels:
        pasta = Path(root) / nome
        if not pasta.is_dir():
            continue
        fotos = sorted(f for f in pasta.iterdir()
                       if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        teto = max_healthy if nome == HEALTHY else max_plants
        for k in range(min(teto, len(fotos) // per_plant)):
            plantas.append((f"{Path(root).name}-{nome}-{k:03d}", nome, Path(root).name,
                            fotos[k * per_plant:(k + 1) * per_plant]))
    return plantas


# ---------------------------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------------------------


def load_frames(
    paths: list[Path], default_labels: list[str] | None = None
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Plantas → quadros, cada quadro com a maior prob. de doença e a maior prob. por recorte."""
    labels: list[str] | None = None
    plantas: dict[str, dict[str, Any]] = {}
    for path in paths:
        for linha in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(linha)
            if labels is None:
                labels = rec.get("labels") or default_labels
            planta = plantas.setdefault(rec["plant"], {"label": rec["label"],
                                                       "group": rec["group"], "frames": []})
            if not rec["probs"]:
                planta["frames"].append(None)  # recusado pela geometria
                continue
            p = np.asarray(rec["probs"], dtype=float)
            sadio = labels.index(HEALTHY)
            doenca = np.delete(p, sadio, axis=1).max(axis=1)
            planta["frames"].append({"doenca": doenca, "confianca": p.max(axis=1)})
    if labels is None:
        raise SystemExit("Nenhum quadro lido.")
    return labels, plantas


def frame_event(frame: dict | None, t: float, k: int) -> bool:
    return frame is not None and int((frame["doenca"] >= t).sum()) >= k


def rates(plantas: dict, ids: list[str], t: float, k: int) -> dict[str, float]:
    """Plantas doentes pegas, quadros doentes evento e quadros sadios evento."""
    pegas = doentes = q_doentes = q_doentes_ev = q_sadios = q_sadios_ev = 0
    for pid in ids:
        p = plantas[pid]
        eventos = [frame_event(f, t, k) for f in p["frames"] if f is not None]
        if p["label"] == HEALTHY:
            q_sadios += len(eventos)
            q_sadios_ev += sum(eventos)
        else:
            doentes += 1
            pegas += any(eventos)
            q_doentes += len(eventos)
            q_doentes_ev += sum(eventos)
    return {
        "plantas_doentes_pegas": pegas / doentes if doentes else math.nan,
        "quadros_doentes_evento": q_doentes_ev / q_doentes if q_doentes else math.nan,
        "quadros_sadios_evento": q_sadios_ev / q_sadios if q_sadios else math.nan,
        "n_plantas_doentes": doentes,
        "n_quadros_sadios": q_sadios,
    }


def choose(plantas: dict, ids: list[str], orcamento: float) -> tuple[float, int]:
    melhor = None
    for k in K_GRID:
        for t in T_GRID:
            r = rates(plantas, ids, t, k)
            if r["quadros_sadios_evento"] > orcamento:
                continue
            chave = (r["plantas_doentes_pegas"], r["quadros_doentes_evento"],
                     -r["quadros_sadios_evento"], -k)
            if melhor is None or chave > melhor[0]:
                melhor = (chave, t, k)
    if melhor is None:
        return 0.99, max(K_GRID)
    return melhor[1], melhor[2]


def folds(plantas: dict, k: int, seed: int) -> dict[str, int]:
    from training.fit_fusion import fixed_folds

    ids = sorted(plantas)
    fixas = fixed_folds([plantas[p]["group"] for p in ids], k)
    if fixas is not None:  # mosaico: a parte vem do grupo de fotos
        return {p: int(f) for p, f in zip(ids, fixas, strict=True)}
    rng = np.random.default_rng(seed)
    por_estrato: dict[tuple, list[str]] = defaultdict(list)
    for pid, p in sorted(plantas.items()):
        por_estrato[(p["label"], p["group"])].append(pid)
    parte = {}
    deslocamento = 0
    for chave in sorted(por_estrato):
        for i, pid in enumerate(rng.permutation(por_estrato[chave])):
            parte[str(pid)] = (i + deslocamento) % k
        deslocamento += len(por_estrato[chave])
    return parte


def evidence_plants(paths: list[Path], labels: list[str]) -> tuple[tuple[str, ...], list]:
    """Plantas no formato de ``training.fit_fusion`` (faixas em ``EVIDENCE_EDGES``)."""
    from training.fit_fusion import Frame, Plant, bin_index

    doencas = tuple(rotulo for rotulo in labels if rotulo != HEALTHY)
    colunas = [labels.index(d) for d in doencas]
    plantas: dict[str, Any] = {}
    for path in paths:
        for linha in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(linha)
            planta = plantas.setdefault(rec["plant"], Plant(rec["plant"], rec["label"],
                                                            rec["group"]))
            if not rec["probs"]:
                continue  # recusado pela geometria: não é evento em nenhuma regra
            p = np.asarray(rec["probs"], dtype=float)[:, colunas]
            planta.frames.append(Frame(camera=str(rec.get("camera_height_mm")), probs=p,
                                       bins=bin_index(p, EVIDENCE_EDGES),
                                       weights=np.ones(len(p)), n_review=0))
    return doencas, [plantas[k] for k in sorted(plantas)]


def frame_scores(plant: Any, tables: np.ndarray, clip: float, rho: float) -> np.ndarray:
    """Escore de cada quadro: max_d da média nos recortes da LLR limitada da faixa, vezes
    o tamanho efetivo ``n / (1 + (n - 1) rho)`` — o desconto de correlação da fusão. Com
    ``rho`` = 1 é a média (uma lesão some num quadro grande); perto de 0, a soma."""
    cols = np.arange(tables.shape[0])[None, :]
    saida = []
    for f in plant.frames:
        n = len(f.bins)
        media = np.clip(tables[cols, f.bins], -clip, clip).mean(axis=0)
        saida.append(float(media.max()) * n / (1.0 + (n - 1) * rho))
    return np.array(saida)


def choose_evidence(plants: list, doencas: tuple[str, ...], orcamento: float,
                    orcamento_validacao: float) -> dict[str, Any]:
    """Tabelas, ``clip`` e limiares que mais pegam plantas doentes com o custo na meta."""
    from training.fit_fusion import fit_tables

    tables = fit_tables(plants, doencas, len(EVIDENCE_EDGES) - 1)
    melhor = None
    for clip in CLIP_GRID:
        for rho in RHO_GRID:
            por_grupo: dict[str, list] = defaultdict(list)
            for p in plants:
                if not p.diseased:
                    por_grupo[p.group].append(frame_scores(p, tables, clip, rho))
            if not por_grupo:
                raise SystemExit("Sem quadros sadios para medir o custo.")
            sadios = np.concatenate([np.concatenate(v) for v in por_grupo.values()])
            # Evento é escore ESTRITAMENTE acima, e o orçamento vale em CADA grupo de
            # sadias (fonte; no mosaico, grupo de fotos): os quadros sadios variam muito
            # de um acervo para outro, e o limiar da média estourava o orçamento fora da
            # amostra (café, 07/10/2026: 20,6% dos quadros sadios contra 5% de meta).
            limiar = max(float(np.quantile(np.concatenate(v), 1 - orcamento, method="higher"))
                         for v in por_grupo.values())
            doentes = [frame_scores(p, tables, clip, rho) > limiar
                       for p in plants if p.diseased]
            pegas = np.mean([bool(ev.any()) for ev in doentes])
            quadros = float(np.mean(np.concatenate(doentes))) if doentes else 0.0
            if melhor is None or (pegas, quadros, -clip, rho) > melhor[0]:
                quase = float(np.quantile(
                    sadios, max(0.0, 1 - orcamento - orcamento_validacao), method="higher"))
                melhor = ((pegas, quadros, -clip, rho), {"tables": tables, "clip": clip, "rho": rho,
                                                "event": limiar, "near": min(quase, limiar)})
    return melhor[1]


def frame_paths(paths: list[Path]) -> set[str]:
    """Por qual caminho os quadros foram classificados (``cnn`` ou ``cabeca:<torre>``)."""
    vistos: set[str] = set()
    for path in paths:
        for linha in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(linha)
            if rec.get("probs"):  # quadro recusado pela geometria não foi classificado
                vistos.add(rec.get("path") or "cnn")
    return vistos


def head_block(spec: Any, caminho: str) -> dict[str, Any] | None:
    """O que o Rover precisa para refazer o caminho com cabeça: a torre e seu hash.

    O Rover recebe a torre de visão ONNX (``models/framing_vision.onnx``) como
    ``<modelo>.encoder.onnx`` e confere o SHA-256 antes de carregar; a cabeça vai como
    ``<modelo>.head.npz``, com a mistura gravada nela.
    """
    if caminho == "cnn":
        return None
    import hashlib

    from app.framing import VISION_ONNX_NAME

    torre = spec.weights_path.parent / VISION_ONNX_NAME
    if not torre.exists():
        raise SystemExit(f"{torre} não existe: exporte com training.export_onnx --framing.")
    sha = hashlib.sha256()
    with torre.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            sha.update(bloco)
    return {"encoder_model_id": caminho.removeprefix("cabeca:"),
            "encoder_sha256": sha.hexdigest(), "input_size": 224}


def categories(labels: list[str]) -> dict[str, str]:
    saida = {}
    for rotulo in labels:
        nome = rotulo.lower()
        if rotulo == HEALTHY:
            saida[rotulo] = "NORMAL"
        elif any(p in nome for p in PEST_WORDS):
            saida[rotulo] = "POSSIVEL_PRAGA"
        else:
            saida[rotulo] = "POSSIVEL_DOENCA"
    return saida


def registro_to_frames(paths: list[Path], labels: dict, model_labels: list[str]) -> list[dict]:
    """O registro do Rover na IA (``FIELDS_TILE_LOG``) no formato do ``collect``.

    O registro tem uma linha por (quadro, planta), com os recortes daquela planta. A
    decisão do Rover é por **quadro**: aqui os recortes voltam a ser o quadro inteiro, e o
    quadro entra em cada planta rotulada que ele tocou — no cafezal, com pés a 0,7 m, um
    quadro de ~1 m toca dois ou três. Planta sem rótulo não entra (o quadro continua
    valendo para as outras). ``labels``: o CSV de ``training.fit_fusion.read_labels``.
    """
    quadros: dict[tuple, dict[str, Any]] = {}
    for path in paths:
        for linha in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(linha)
            chave = (rec["field_id"], rec["pass_id"], rec["frame"])
            q = quadros.setdefault(chave, {"plants": [], "probs": [],
                                           "camera_height_mm": rec.get("camera_height_mm")})
            q["plants"].append(str(rec["plant"]))
            for t in rec.get("tiles") or []:
                q["probs"].append([float(t["probabilities"].get(c, 0.0)) for c in model_labels])
    saida = []
    for (field_id, pass_id, _), q in sorted(quadros.items()):
        for planta in sorted(set(q["plants"])):
            rotulo = labels.get((field_id, planta, pass_id)) or labels.get((field_id, planta, None))
            if rotulo is None:
                continue
            saida.append({"plant": f"{field_id}|{pass_id}|{planta}", "label": rotulo,
                          "group": f"talhao:{field_id}", "labels": model_labels,
                          "camera_height_mm": q["camera_height_mm"], "probs": q["probs"]})
    return saida


# ---------------------------------------------------------------------------------------------
# mosaico (café)
# ---------------------------------------------------------------------------------------------


def mosaic_photos(args: argparse.Namespace, spec: Any,
                  labels: tuple[str, ...]) -> list[tuple[Path, str]]:
    """As fotos que o modelo não viu: o teste do acervo e os acervos à parte, inteiros."""
    from training.dataset import index_by_class, split_indices
    from training.prepare_dataset import looks_like_label_mask

    paths, idx, names = index_by_class(args.data_dir or spec.data_dir)
    if tuple(names) != labels:
        raise SystemExit(f"Classes {names} diferem do modelo {labels}.")
    _, _, teste = split_indices(len(paths), args.seed, args.validation_split, args.test_split)
    fotos = [(Path(paths[i]), names[idx[i]]) for i in teste
             if not looks_like_label_mask(Path(paths[i]))]
    for raiz in args.extra_dir or []:
        for nome in labels:
            pasta = Path(raiz) / nome
            if pasta.is_dir():
                fotos += [(f, nome) for f in sorted(pasta.iterdir())
                          if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    return fotos


def mosaic_frames(probs: dict[str, np.ndarray], pools: dict[str, np.ndarray], *,
                  tiles: int, incidences: list[float], plants: int, healthy: int,
                  rng: np.random.Generator, labels: list[str], path: str,
                  n_pools: int, per_plant: int = 3) -> list[dict[str, Any]]:
    """Quadros sintéticos do Rover no cafezal: cada recorte é uma folha fotografada.

    A 270 mm, o recorte de 150 mm mostra cerca de uma folha de café — o que cada foto de
    folha em close mostra. Um quadro tem ``tiles`` recortes; numa planta doente, cada um
    é uma folha doente com probabilidade ``incidência`` (a fração de folhas com a doença,
    como o agrônomo mede) e sadia no resto. ``probs[classe]`` são as probabilidades de
    cada foto (pela cabeça), ``pools[classe]`` o grupo de cada foto: uma planta usa fotos
    de um grupo só.
    """
    linhas = []
    for grupo in range(n_pools):
        def do_grupo(classe: str, grupo: int = grupo) -> np.ndarray:
            return probs[classe][pools[classe] == grupo]

        sadias = do_grupo(HEALTHY)
        if not len(sadias):
            continue
        for k in range(healthy):
            for q in range(per_plant):
                escolha = sadias[rng.integers(0, len(sadias), tiles)]
                linhas.append({"plant": f"g{grupo}-Healthy-{k:03d}", "label": HEALTHY,
                               "group": f"dobra:{grupo}", "path": path, "labels": labels,
                               "camera_height_mm": (450.0, 1000.0, 1540.0)[q % 3],
                               "incidence": 0.0, "probs": np.round(escolha, 5).tolist()})
        for classe in sorted(c for c in probs if c != HEALTHY):
            doentes = do_grupo(classe)
            if not len(doentes):
                continue
            for inc in incidences:
                for k in range(plants):
                    for q in range(per_plant):
                        doente = rng.random(tiles) < inc
                        escolha = np.where(doente[:, None],
                                           doentes[rng.integers(0, len(doentes), tiles)],
                                           sadias[rng.integers(0, len(sadias), tiles)])
                        linhas.append({
                            "plant": f"g{grupo}-{classe}-{inc:g}-{k:03d}", "label": classe,
                            "group": f"dobra:{grupo}", "path": path, "labels": labels,
                            "camera_height_mm": (450.0, 1000.0, 1540.0)[q % 3],
                            "incidence": inc, "probs": np.round(escolha, 5).tolist()})
    return linhas


def mosaico(args: argparse.Namespace) -> None:
    """Classifica cada foto uma vez (o caminho do Rover) e monta os quadros do mosaico."""
    import hashlib

    from PIL import Image

    spec = get_spec(args.model)
    labels = tuple(json.loads(spec.labels_path.read_text(encoding="utf-8")))
    cabeca = _head_for(spec, args.head, labels)
    if cabeca is None:
        raise SystemExit("O mosaico mede o caminho com cabeça: o modelo não tem head.npz.")
    head, encode, caminho = cabeca
    if head.mix.get("cnn", 0.0) > 0:
        raise SystemExit("Mistura com a CNN no mosaico ainda não implementada.")
    fotos = mosaic_photos(args, spec, labels)
    print(f"{spec.key}: {len(fotos)} fotos; caminho {caminho}", flush=True)
    probs: dict[str, list] = defaultdict(list)
    grupos: dict[str, list] = defaultdict(list)
    for i in range(0, len(fotos), 64):
        lote = fotos[i:i + 64]
        imagens = []
        for f, _ in lote:
            with Image.open(f) as im:
                imagens.append(im.convert("RGB"))
        p = head.combine(np.full((len(lote), len(labels)), 1.0 / len(labels)), encode(imagens))
        for (f, classe), linha in zip(lote, p, strict=True):
            probs[classe].append(linha)
            # Grupo pela foto (estável entre execuções), não pela ordem.
            grupos[classe].append(int(hashlib.sha256(f.name.encode()).hexdigest(), 16)
                                  % args.folds)
    acerto = {c: float(np.mean(np.argmax(v, axis=1) == labels.index(c)))
              for c, v in probs.items()}
    print("acerto por foto (folha inteira):",
          {c: f"{a:.3f} (n={len(probs[c])})" for c, a in sorted(acerto.items())}, flush=True)
    linhas = mosaic_frames({c: np.array(v) for c, v in probs.items()},
                           {c: np.array(v) for c, v in grupos.items()},
                           tiles=args.recortes, incidences=args.incidencias, plants=args.plantas,
                           healthy=args.sadias, rng=np.random.default_rng(args.seed),
                           labels=list(labels), path=caminho, n_pools=args.folds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x) + "\n" for x in linhas), encoding="utf-8")
    print(f"{len(linhas)} quadros em {args.out}")


def registro_path(spec: Any) -> str:
    """O caminho com que o Rover classificou: o do perfil instalado, ou o do modelo."""
    instalado = spec.weights_path.with_suffix(".edge.json")
    if instalado.exists():
        cabeca = json.loads(instalado.read_text(encoding="utf-8")).get("head")
        return f"cabeca:{cabeca['encoder_model_id']}" if cabeca else "cnn"
    if spec.head_path.exists():
        from app.inference.heads import DiagnosisHead

        return f"cabeca:{DiagnosisHead.load(spec.head_path).encoder}"
    return "cnn"


def fit(args: argparse.Namespace) -> None:
    spec = get_spec(args.model)
    if getattr(args, "labels", None):
        # Registro de campo + rótulo do agrônomo: vira o formato do collect, num arquivo.
        from training.fit_fusion import read_labels

        do_modelo = (json.loads(spec.labels_path.read_text(encoding="utf-8"))
                     if spec.labels_path.exists() else list(spec.class_names))
        linhas = registro_to_frames(args.frames, read_labels(args.labels), do_modelo)
        if not linhas:
            raise SystemExit("Nenhum quadro do registro tem planta rotulada no CSV.")
        # O Rover classificou pelo caminho do perfil instalado (no café, com a cabeça):
        # o perfil recalibrado tem de manter esse caminho.
        caminho = registro_path(spec)
        for linha in linhas:
            linha["path"] = caminho
        convertido = (args.out or spec.weights_path).with_suffix(".registro.jsonl")
        convertido.write_text("".join(json.dumps(x) + "\n" for x in linhas), encoding="utf-8")
        print(f"{len(linhas)} pares (quadro, planta rotulada) do registro de campo")
        args.frames = [convertido]
    # A ordem que vale é a da saída do modelo (``.labels.json``); no café ela difere da
    # ordem de exibição do catálogo (``class_names``).
    if spec.labels_path.exists():
        do_modelo = json.loads(spec.labels_path.read_text(encoding="utf-8"))
        labels, plantas = load_frames(args.frames, do_modelo)
    else:  # sem os pesos aqui: vale a ordem do JSONL, se as classes forem as do catálogo
        labels, plantas = load_frames(args.frames, list(spec.class_names))
        do_modelo = labels if set(labels) == set(spec.class_names) else list(spec.class_names)
    if do_modelo != labels:
        raise SystemExit(f"Rótulos do JSONL {labels} diferem do modelo {do_modelo}.")
    ids = sorted(plantas)
    n_doentes = sum(plantas[p]["label"] != HEALTHY for p in ids)
    n_sadias = len(ids) - n_doentes
    if n_doentes < 10 or n_sadias < 10:
        raise SystemExit(f"Poucas plantas: {n_doentes} doentes, {n_sadias} sadias.")

    # Validação cruzada por planta: os números do relatório são fora da amostra.
    parte = folds(plantas, args.folds, args.seed)
    escolhas = []
    fora = {"pegas": 0, "doentes": 0, "q_doentes_ev": 0, "q_doentes": 0, "q_sadios_ev": 0,
            "q_sadios": 0}
    for i in range(args.folds):
        treino = [p for p in ids if parte[p] != i]
        teste = [p for p in ids if parte[p] == i]
        t, k = choose(plantas, treino, args.orcamento)
        escolhas.append({"event_threshold": t, "min_event_tiles": k})
        r = rates(plantas, teste, t, k)
        fora["doentes"] += r["n_plantas_doentes"]
        fora["pegas"] += round(r["plantas_doentes_pegas"] * r["n_plantas_doentes"]) \
            if r["n_plantas_doentes"] else 0
        for pid in teste:
            p = plantas[pid]
            for f in p["frames"]:
                if f is None:
                    continue
                ev = frame_event(f, t, k)
                if p["label"] == HEALTHY:
                    fora["q_sadios"] += 1
                    fora["q_sadios_ev"] += ev
                else:
                    fora["q_doentes"] += 1
                    fora["q_doentes_ev"] += ev

    t, k = choose(plantas, ids, args.orcamento)

    # Regra por evidência: validação cruzada com as mesmas partes, tudo refeito no treino.
    doencas, ev_plantas = evidence_plants(args.frames, labels)
    incidencias = {}  # mosaico: a fração de folhas doentes de cada planta sintética
    for path in args.frames:
        for linha in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(linha)
            if "incidence" in rec:
                incidencias[rec["plant"]] = rec["incidence"]
    ev_fora = {"pegas": 0, "doentes": 0, "q_doentes_ev": 0, "q_doentes": 0,
               "q_sadios_ev": 0, "q_sadios": 0}
    ev_escolhas = []
    for i in range(args.folds):
        treino = [p for p in ev_plantas if parte[p.plant_id] != i]
        teste = [p for p in ev_plantas if parte[p.plant_id] == i]
        esc = choose_evidence(treino, doencas, args.orcamento, args.orcamento_validacao)
        ev_escolhas.append({"llr_clip": esc["clip"], "tile_correlation": esc["rho"],
                            "event_evidence": round(esc["event"], 4)})
        for p in teste:
            ev = frame_scores(p, esc["tables"], esc["clip"], esc["rho"]) > esc["event"]
            if p.diseased:
                ev_fora["doentes"] += 1
                ev_fora["pegas"] += bool(ev.any())
                inc = incidencias.get(p.plant_id)
                if inc is not None:
                    por_inc = ev_fora.setdefault("por_incidencia", {}).setdefault(
                        f"{inc:g}", [0, 0])
                    por_inc[0] += bool(ev.any())
                    por_inc[1] += 1
                ev_fora["q_doentes"] += len(ev)
                ev_fora["q_doentes_ev"] += int(ev.sum())
            else:
                ev_fora["q_sadios"] += len(ev)
                ev_fora["q_sadios_ev"] += int(ev.sum())
    evid = choose_evidence(ev_plantas, doencas, args.orcamento, args.orcamento_validacao)
    final = rates(plantas, ids, t, k)
    # near miss: o maior recorte de doença dos quadros sadios que NÃO são evento. Escolhido
    # para que só ``--orcamento-validacao`` deles a mais suba em original para validação.
    restantes = [float(f["doenca"].max()) for p in plantas.values() if p["label"] == HEALTHY
                 for f in p["frames"] if f is not None and not frame_event(f, t, k)]
    near = float(np.quantile(restantes, 1 - args.orcamento_validacao)) if restantes else None
    near = min(near, t) if near is not None else None
    # anomalia: confiança média nos recortes abaixo do 1 % mais baixo dos quadros conhecidos.
    # No mosaico (cada recorte é uma foto limpa de folha) a confiança média é alta demais
    # para valer em quadro de verdade — medido em 07/10/2026: quase todo quadro sadio de
    # cafezal montado saía ANOMALIA. Sem quadro de campo, a regra fica desligada.
    mosaico = any(p["group"].startswith("dobra:") for p in plantas.values())
    medias = [float(f["confianca"].mean()) for p in plantas.values() for f in p["frames"]
              if f is not None]
    anomalia = (float(np.quantile(medias, args.quantil_anomalia))
                if medias and not mosaico else None)

    medido = {
        "orcamento_quadros_sadios": args.orcamento,
        "fora_da_amostra": {
            "partes": args.folds,
            "plantas_doentes_pegas": f"{fora['pegas']}/{fora['doentes']}",
            "quadros_doentes_evento": f"{fora['q_doentes_ev']}/{fora['q_doentes']}",
            "quadros_sadios_evento": f"{fora['q_sadios_ev']}/{fora['q_sadios']}",
            "escolhas_por_parte": escolhas,
        },
        "no_conjunto_inteiro": {k2: (round(v, 4) if isinstance(v, float) else v)
                                for k2, v in final.items()},
        "evidencia_fora_da_amostra": {
            "partes": args.folds,
            "plantas_doentes_pegas": f"{ev_fora['pegas']}/{ev_fora['doentes']}",
            "quadros_doentes_evento": f"{ev_fora['q_doentes_ev']}/{ev_fora['q_doentes']}",
            "quadros_sadios_evento": f"{ev_fora['q_sadios_ev']}/{ev_fora['q_sadios']}",
            "escolhas_por_parte": ev_escolhas,
            **({"plantas_doentes_pegas_por_incidencia": {
                k: f"{v[0]}/{v[1]}" for k, v in sorted(ev_fora["por_incidencia"].items(),
                                                       key=lambda kv: float(kv[0]))}}
               if "por_incidencia" in ev_fora else {}),
        },
        "plantas": {"doentes": n_doentes, "sadias": n_sadias},
        "dados": [str(p) for p in args.frames],
        "data": datetime.now(UTC).isoformat(timespec="seconds"),
        "limite": "fotos de folha agrupadas, não o Rover no pomar; validar em campo",
    }
    caminhos = frame_paths(args.frames)
    if len(caminhos) > 1:
        raise SystemExit(f"Quadros de caminhos diferentes misturados: {sorted(caminhos)}.")
    cabeca = head_block(spec, next(iter(caminhos), "cnn"))
    perfil = {
        "version": PROFILE_VERSION,
        "crop": spec.crop,
        **({"head": cabeca} if cabeca else {}),
        "healthy_label": HEALTHY,
        "event_threshold": t,
        "min_event_tiles": k,
        "near_miss_threshold": None if near is None else round(near, 4),
        "anomaly_mean_confidence_below": None if anomalia is None else round(anomalia, 4),
        "tile_mm": 150.0,
        "overlap": 0.25,
        "hfov_deg": 127.0,
        "vfov_deg": 110.0,
        "default_distance_mm": 270.0,
        "tile_batch": 32,
        "tile_probabilities": "full",
        "evidence": {
            "bin_edges": list(EVIDENCE_EDGES),
            "llr_tables": {d: [round(float(v), 5) for v in evid["tables"][j]]
                           for j, d in enumerate(doencas)},
            "llr_clip": evid["clip"],
            "tile_correlation": evid["rho"],
            "event_evidence": round(evid["event"], 6),
            "near_miss_evidence": round(evid["near"], 6),
        },
        "measured": medido,
    }
    destino = args.out or spec.weights_path.with_suffix(".edge.json")
    destino.write_text(json.dumps(perfil, indent=2, ensure_ascii=False), encoding="utf-8")
    nome = destino.name.removesuffix(".edge.json").removesuffix(".json")
    cats = destino.with_name(nome + ".categories.json")
    cats.write_text(json.dumps(categories(labels), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    fa = medido["fora_da_amostra"]
    print(f"{spec.key}: {n_doentes} plantas doentes, {n_sadias} sadias")
    print(f"  fora da amostra ({args.folds} partes): plantas doentes pegas "
          f"{fa['plantas_doentes_pegas']}, quadros doentes evento {fa['quadros_doentes_evento']}, "
          f"quadros sadios evento {fa['quadros_sadios_evento']}")
    ef = medido["evidencia_fora_da_amostra"]
    print(f"  evidência, fora da amostra: plantas doentes pegas {ef['plantas_doentes_pegas']}, "
          f"quadros doentes evento {ef['quadros_doentes_evento']}, "
          f"quadros sadios evento {ef['quadros_sadios_evento']}")
    if "plantas_doentes_pegas_por_incidencia" in ef:
        print(f"  por incidência (fração de folhas doentes): "
              f"{ef['plantas_doentes_pegas_por_incidencia']}")
    print(f"  evidência no perfil: clip {evid['clip']}, rho {evid['rho']}, "
          f"evento > {evid['event']:.3f}, "
          f"validação > {evid['near']:.3f}")
    print(f"  (contagem) evento com {k} recorte(s) ≥ {t:.2f}; validação ≥ {near}; "
          f"anomalia < {anomalia}")
    print(f"Gravado em {destino} e {cats}")


# ---------------------------------------------------------------------------------------------
# fusao
# ---------------------------------------------------------------------------------------------


def to_fusion_lines(paths: list[Path], default_labels: list[str]) -> list[dict[str, Any]]:
    """Quadros da borda no formato que ``training.fit_fusion.load_plants`` lê."""
    saida = []
    for path in paths:
        for linha in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(linha)
            labels = rec.get("labels") or default_labels
            recortes = []
            for probs in rec.get("probs") or []:
                melhor = int(np.argmax(probs))
                recortes.append({
                    "box": [0, 0, 0, 0],
                    "predicted_class": labels[melhor],
                    "probabilities": dict(zip(labels, map(float, probs), strict=True)),
                    "confidence": float(probs[melhor]),
                    "in_scope": None,  # sem portão no Rover: todo recorte conta
                })
            saida.append({"plant": rec["plant"], "label": rec["label"], "group": rec["group"],
                          "camera_height_mm": rec.get("camera_height_mm", 0.0),
                          "tiles": recortes})
    return saida


def fusao(args: argparse.Namespace) -> None:
    from training import fit_fusion

    spec = get_spec(args.model)
    labels = json.loads(spec.labels_path.read_text(encoding="utf-8"))
    out = args.out or spec.edge_fusion_path
    convertido = out.with_suffix(".frames.jsonl")
    linhas = to_fusion_lines(args.frames, labels)
    convertido.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in linhas), encoding="utf-8"
    )
    print(f"{len(linhas)} quadros convertidos em {convertido}")
    argv = ["fit", "--model", spec.key, "--frames", str(convertido), "--out", str(out),
            "--seed", str(args.seed)]
    try:
        fit_fusion.fit(fit_fusion.parse_args(argv))
    finally:
        convertido.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibração da Edge AI do Rover")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect", help="Probabilidades do ONNX sozinho, recorte a recorte.")
    c.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    c.add_argument("--data-dir", type=Path, default=None)
    c.add_argument("--data-dir-classes", nargs="+", default=None)
    c.add_argument("--plantseg", type=Path, default=None)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--max-plants", type=int, default=40)
    c.add_argument("--max-healthy", type=int, default=300)
    c.add_argument("--threads", type=int, default=3)
    c.add_argument("--head", choices=("auto", "sim", "nao"), default="auto",
                   help="Classificar pela cabeça sobre o embedding (auto: se o modelo tiver).")
    c.add_argument("--extra-dir", type=Path, nargs="+", default=None,
                   help="Acervos à parte que o modelo não viu (todas as fotos, por classe).")
    c.add_argument("--seed", type=int, default=DEFAULT_SEED)
    c.add_argument("--validation-split", type=float, default=0.15)
    c.add_argument("--test-split", type=float, default=0.15)

    f = sub.add_parser("fit", help="Limiares por validação cruzada → <pesos>.edge.json.")
    f.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    f.add_argument("--frames", type=Path, nargs="+", required=True)
    f.add_argument("--out", type=Path, default=None, help="Padrão: ao lado dos pesos.")
    f.add_argument("--orcamento", type=float, default=0.05,
                   help="Fração máxima de quadros sadios que vira evento (sobe em original).")
    f.add_argument("--orcamento-validacao", type=float, default=0.05,
                   help="Fração extra de quadros sadios guardada como baixa confiança.")
    f.add_argument("--quantil-anomalia", type=float, default=0.01)
    f.add_argument("--labels", type=Path, default=None,
                   help="CSV field_id,plant_id,label[,pass_id]: --frames é o registro de campo.")
    f.add_argument("--folds", type=int, default=5)
    f.add_argument("--seed", type=int, default=0)

    m = sub.add_parser("mosaico", help="Café: quadros do Rover montados com fotos de folha.")
    m.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    m.add_argument("--data-dir", type=Path, default=None)
    m.add_argument("--extra-dir", type=Path, nargs="+", default=None)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--head", choices=("auto", "sim"), default="sim")
    m.add_argument("--recortes", type=int, default=70,
                   help="Recortes por quadro (a 270 mm num quadro de 12 MP: ~70).")
    m.add_argument("--incidencias", type=float, nargs="+", default=[0.05, 0.1, 0.2],
                   help="Frações de folhas doentes nas plantas doentes.")
    m.add_argument("--plantas", type=int, default=12,
                   help="Plantas doentes por classe, incidência e grupo de fotos.")
    m.add_argument("--sadias", type=int, default=60, help="Plantas sadias por grupo de fotos.")
    m.add_argument("--folds", type=int, default=5, help="Grupos de fotos sem repetição.")
    m.add_argument("--seed", type=int, default=DEFAULT_SEED)
    m.add_argument("--validation-split", type=float, default=0.15)
    m.add_argument("--test-split", type=float, default=0.15)

    u = sub.add_parser("fusao", help="Fusão por planta dos recortes do Rover → .edge-fusion.json.")
    u.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    u.add_argument("--frames", type=Path, nargs="+", required=True)
    u.add_argument("--out", type=Path, default=None, help="Padrão: ao lado dos pesos.")
    u.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "collect":
        collect(args)
    elif args.command == "fusao":
        fusao(args)
    elif args.command == "mosaico":
        mosaico(args)
    else:
        fit(args)


if __name__ == "__main__":
    main(sys.argv[1:])
