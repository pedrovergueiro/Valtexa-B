"""Aprendizado contínuo da cabeça de diagnóstico: aprender com o campo sem esquecer.

A cabeça (:mod:`app.inference.heads`) lê o embedding do SigLIP 2 de cada recorte; a torre e
a CNN ficam congeladas. É ela que aprende com o que o agrônomo valida no campo — o dado que
nenhum acervo público tem: o cafezal do cliente, na luz e no ângulo do Rover. O risco de
treinar de novo com dado novo é o **esquecimento catastrófico** (McCloskey & Cohen, 1989): a
rede melhora no talhão e piora em tudo que já sabia. Três peças evitam isso:

1. **Adaptação supervisionada local** (:func:`adapt`). Ajuste fino da cabeça em produção, em
   NumPy (CPU, segundos — roda no servidor da IA ou na Jetson), só com recortes de rótulo
   confirmado. A perda soma:

   * entropia cruzada balanceada por classe nos recortes do campo e nos do replay;
   * **destilação** nos exemplos do replay contra os *logits* gravados quando eles entraram na
     memória (DER++, Buzzega et al., NeurIPS 2020; LwF, Li & Hoiem, 2016): a resposta antiga
     é o alvo, não só o rótulo;
   * **âncora L2-SP** (Li et al., ICML 2018): penalidade na distância aos pesos de partida.

   A época que fica é a melhor na validação — metade campo, metade o que já se sabia —,
   entre as que não pioram a validação do que já se sabia (nem em acurácia nem em alarme
   falso em sadia): a seleção já respeita o que o portão vai cobrar, sem nunca olhar os
   conjuntos dele. Essa validação é a **original** (o split que a cabeça nunca treinou, o
   mesmo que escolheu a arquitetura no ``fit_heads``) mais a validação de cada rodada
   promovida — não a memória, que a cabeça decorou: medido em 15/10/2026, a seleção pela
   memória nunca recusou uma época.

2. **Replay** (:class:`ReplayMemory`). Embeddings (não fotos) de exemplos anteriores, por
   classe, escolhidos por *herding* (iCaRL, Rebuffi et al., CVPR 2017: os que mantêm a média
   da classe), com a origem de cada um. Cada rodada aprovada entra na memória com cota por
   origem: a rodada nova não apaga a anterior, e o acervo público não some.

3. **Monitoramento de regressões** (:class:`RegressionSuite`, :func:`compare`). Conjuntos
   congelados — o teste isolado do treino, o de outros acervos, o de cada rodada de campo já
   aprovada — e um teste **pareado** (bootstrap nos mesmos exemplos) da candidata contra a
   cabeça em produção. No agregado, o ônus da prova é da candidata (não-inferioridade: o
   limite inferior do IC de 95% acima de menos a margem); em cada conjunto, em cada classe e
   no alarme falso em folha sadia, nenhuma piora demonstrada nem acima da margem
   (:class:`GateCriteria` explica o porquê das duas formas). E ela precisa ganhar no campo.

O portão e a seleção medem a leitura da cabeça (``head.mlp``), que é o que muda. No café
(mistura ``cnn=0, head=1``) ela é a resposta inteira; num modelo com mistura (citros: 25% de
CNN), a resposta final também tem a CNN, que não muda aqui.

A cabeça nova muda as probabilidades: tudo calibrado sobre elas (conformal, perfil da borda
``.edge.json``, fusão por planta) passa a valer para a cabeça antiga. :func:`edge_drift` mede
quanto a evidência do perfil da borda se moveria nas folhas sadias; o ``training.adapt_campo``
diz o que recalibrar antes de levar a cabeça ao Rover — que recusa um perfil medido com outra
cabeça (``head.head_sha256``).

Só NumPy. O ``training.adapt_campo`` é a linha de comando.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from app.inference.heads import ACTIVATIONS, DiagnosisHead, EmbeddingMLP, _erf, softmax

HEALTHY = "Healthy"


# ---------------------------------------------------------------------------------------------
# A rede em NumPy, com gradiente
# ---------------------------------------------------------------------------------------------


def _derivada(nome: str, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Derivada da ativação em ``z`` (``a`` é a ativação já calculada)."""
    if nome == "relu":
        return (z > 0).astype(z.dtype)
    if nome == "tanh":
        return 1.0 - a * a
    if nome == "sigmoid":
        return a * (1.0 - a)
    if nome == "gelu":
        cdf = 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
        return cdf + z * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    raise ValueError(f"ativação sem derivada: {nome}")


def _parametros(mlp: EmbeddingMLP) -> list[np.ndarray]:
    """Os pesos treináveis, numa ordem fixa (cópias em float64)."""
    saida = []
    for w, b in zip(mlp.weights, mlp.biases, strict=True):
        saida += [np.array(w, dtype=np.float64), np.array(b, dtype=np.float64)]
    saida += [np.array(mlp.out_weight, dtype=np.float64), np.array(mlp.out_bias, dtype=np.float64)]
    for w, b in zip(mlp.skip_weights, mlp.skip_biases, strict=True):
        saida += [np.array(w, dtype=np.float64), np.array(b, dtype=np.float64)]
    return saida


def _com_parametros(mlp: EmbeddingMLP, params: list[np.ndarray]) -> EmbeddingMLP:
    n = len(mlp.weights)
    k = len(mlp.skip_weights)
    return replace(
        mlp,
        weights=tuple(params[2 * i] for i in range(n)),
        biases=tuple(params[2 * i + 1] for i in range(n)),
        out_weight=params[2 * n],
        out_bias=params[2 * n + 1],
        skip_weights=tuple(params[2 * n + 2 + 2 * j] for j in range(k)),
        skip_biases=tuple(params[2 * n + 3 + 2 * j] for j in range(k)),
    )


def standardize(mlp: EmbeddingMLP, embeddings: np.ndarray) -> np.ndarray:
    """A entrada da rede: embedding cru → L2 → padronizado pela média e desvio **do treino
    original** (congelados: a memória e o campo ficam no mesmo espaço)."""
    x = np.asarray(embeddings, dtype=np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return (x - mlp.mean) / mlp.std


def forward(mlp: EmbeddingMLP, params: list[np.ndarray], x: np.ndarray):
    """Logits e o que o gradiente precisa. Mesma conta de :meth:`EmbeddingMLP.logits`."""
    n = len(mlp.weights)
    saidas = [x]
    zs, acts, aplicados = [], [], []
    for i in range(n):
        w, b = params[2 * i], params[2 * i + 1]
        z = saidas[-1] @ w.T + b
        a = ACTIVATIONS[mlp.activations[i]](z)
        h = a
        usados = []
        for s, (origem, destino, proj) in enumerate(mlp.skips):
            if destino != i + 1 or origem >= len(saidas):
                continue
            atalho = saidas[origem]
            if proj >= 0:
                atalho = atalho @ params[2 * n + 2 + 2 * proj].T + params[2 * n + 3 + 2 * proj]
            if atalho.shape == h.shape:
                h = h + atalho
                usados.append(s)
        zs.append(z)
        acts.append(a)
        aplicados.append(usados)
        saidas.append(h)
    logits = saidas[-1] @ params[2 * n].T + params[2 * n + 1]
    return logits, (saidas, zs, acts, aplicados)


def backward(mlp: EmbeddingMLP, params: list[np.ndarray], cache, dlogits: np.ndarray
             ) -> list[np.ndarray]:
    """Gradiente de cada parâmetro dado o gradiente nos logits."""
    saidas, zs, acts, aplicados = cache
    n = len(mlp.weights)
    grads = [np.zeros_like(p) for p in params]
    grads[2 * n] = dlogits.T @ saidas[-1]
    grads[2 * n + 1] = dlogits.sum(axis=0)
    g = [np.zeros_like(s) for s in saidas]
    g[-1] = dlogits @ params[2 * n]
    for i in range(n - 1, -1, -1):
        gh = g[i + 1]
        for s in aplicados[i]:
            origem, _destino, proj = mlp.skips[s]
            if proj < 0:
                g[origem] = g[origem] + gh
            else:
                grads[2 * n + 2 + 2 * proj] += gh.T @ saidas[origem]
                grads[2 * n + 3 + 2 * proj] += gh.sum(axis=0)
                g[origem] = g[origem] + gh @ params[2 * n + 2 + 2 * proj]
        dz = gh * _derivada(mlp.activations[i], zs[i], acts[i])
        grads[2 * i] = dz.T @ saidas[i]
        grads[2 * i + 1] = dz.sum(axis=0)
        g[i] = g[i] + dz @ params[2 * i]
    return grads


# ---------------------------------------------------------------------------------------------
# Memória de replay
# ---------------------------------------------------------------------------------------------


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def herding(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Os ``k`` índices cuja média acompanha a média da classe (iCaRL, algoritmo 4)."""
    x = _l2(embeddings)
    n = len(x)
    if k >= n:
        return np.arange(n)
    media = x.mean(axis=0)
    escolhidos: list[int] = []
    soma = np.zeros_like(media)
    livre = np.ones(n, dtype=bool)
    for t in range(1, k + 1):
        candidatos = (soma[None, :] + x) / t
        dist = np.linalg.norm(media[None, :] - candidatos, axis=1)
        dist[~livre] = np.inf
        i = int(np.argmin(dist))
        escolhidos.append(i)
        livre[i] = False
        soma += x[i]
    return np.array(escolhidos, dtype=np.int64)


@dataclass
class ReplayMemory:
    """Exemplos anteriores, como embedding, com o rótulo, a origem e os logits da época."""

    embeddings: np.ndarray
    """``(N, D)`` crus (float16 no arquivo)."""
    labels: np.ndarray
    logits: np.ndarray
    """``(N, C)``: o que a cabeça em produção dizia quando o exemplo entrou (DER++)."""
    sources: np.ndarray
    """Origem: ``publico:<dataset>`` ou ``campo:<escopo>:<rodada>``."""
    refs: np.ndarray
    """Arquivo de origem (rastreável; :meth:`forget` tira por consentimento retirado)."""
    rounds: np.ndarray
    class_names: tuple[str, ...]
    encoder: str
    per_class: int

    def __len__(self) -> int:
        return len(self.labels)

    @classmethod
    def empty(cls, class_names, encoder: str, dim: int, per_class: int) -> ReplayMemory:
        c = len(class_names)
        return cls(np.zeros((0, dim)), np.zeros(0, np.int64), np.zeros((0, c)),
                   np.zeros(0, dtype="<U96"), np.zeros(0, dtype="<U160"),
                   np.zeros(0, np.int64), tuple(class_names), encoder, per_class)

    def add(self, embeddings: np.ndarray, labels: np.ndarray, head: DiagnosisHead,
            source: str, refs: np.ndarray | None = None, round_index: int = 0) -> ReplayMemory:
        """Junta exemplos novos e reescolhe, por classe, com cota igual por origem.

        A cota por origem é o que impede a rodada nova de expulsar as anteriores: numa classe
        com ``per_class`` vagas e três origens, cada uma fica com um terço (o que sobra de uma
        origem pequena vai para as outras), escolhido por *herding* dentro da origem.
        """
        if head.encoder != self.encoder:
            raise ValueError(f"Torre {head.encoder} diferente da memória ({self.encoder}).")
        embeddings = np.asarray(embeddings, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        logits = head.mlp.logits(embeddings)
        refs = np.asarray(refs if refs is not None else [""] * len(labels), dtype="<U160")
        todos = {
            "embeddings": np.vstack([self.embeddings, embeddings]),
            "labels": np.concatenate([self.labels, labels]),
            "logits": np.vstack([self.logits, logits]),
            "sources": np.concatenate([self.sources, np.full(len(labels), source, "<U96")]),
            "refs": np.concatenate([self.refs, refs]),
            "rounds": np.concatenate([self.rounds, np.full(len(labels), round_index, np.int64)]),
        }
        manter: list[int] = []
        for c in range(len(self.class_names)):
            idx = np.flatnonzero(todos["labels"] == c)
            if len(idx) <= self.per_class:
                manter += idx.tolist()
                continue
            origens = sorted(set(todos["sources"][idx]))
            grupos = {o: idx[todos["sources"][idx] == o] for o in origens}
            cotas = {o: 0 for o in origens}
            vagas = self.per_class
            abertas = list(origens)
            while vagas > 0 and abertas:  # água enchendo: cota igual, sobra redistribuída
                parte = max(1, vagas // len(abertas))
                for o in list(abertas):
                    extra = min(parte, len(grupos[o]) - cotas[o], vagas)
                    cotas[o] += extra
                    vagas -= extra
                    if cotas[o] >= len(grupos[o]):
                        abertas.remove(o)
                    if vagas == 0:
                        break
            for o in origens:
                escolha = herding(todos["embeddings"][grupos[o]], cotas[o])
                manter += grupos[o][escolha].tolist()
        manter = np.array(sorted(manter), dtype=np.int64)
        return replace(self, **{k: v[manter] for k, v in todos.items()})

    def forget(self, prefix: str) -> tuple[ReplayMemory, int]:
        """Tira tudo cuja origem ou arquivo começa com ``prefix`` (consentimento retirado)."""
        fora = np.array([s.startswith(prefix) or r.startswith(prefix)
                         for s, r in zip(self.sources, self.refs, strict=True)], dtype=bool)
        fica = ~fora
        return replace(self, embeddings=self.embeddings[fica], labels=self.labels[fica],
                       logits=self.logits[fica], sources=self.sources[fica],
                       refs=self.refs[fica], rounds=self.rounds[fica]), int(fora.sum())

    def counts(self) -> dict[str, dict[str, int]]:
        saida: dict[str, dict[str, int]] = {}
        for s, y in zip(self.sources, self.labels, strict=True):
            saida.setdefault(str(s), {}).setdefault(self.class_names[int(y)], 0)
            saida[str(s)][self.class_names[int(y)]] += 1
        return saida

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, embeddings=self.embeddings.astype(np.float16), labels=self.labels,
            logits=self.logits.astype(np.float32), sources=self.sources, refs=self.refs,
            rounds=self.rounds, class_names=np.array(self.class_names),
            encoder=np.array(self.encoder), per_class=np.array(self.per_class))

    @classmethod
    def load(cls, path: Path) -> ReplayMemory:
        with np.load(path, allow_pickle=False) as z:
            return cls(z["embeddings"].astype(np.float64), z["labels"].astype(np.int64),
                       z["logits"].astype(np.float64), z["sources"], z["refs"],
                       z["rounds"].astype(np.int64), tuple(str(c) for c in z["class_names"]),
                       str(z["encoder"]), int(z["per_class"]))


# ---------------------------------------------------------------------------------------------
# Conjuntos de regressão
# ---------------------------------------------------------------------------------------------


@dataclass
class RegressionSet:
    name: str
    embeddings: np.ndarray
    labels: np.ndarray
    gate: bool = True
    """``False``: só mede (acervo de licença só-medição, como o externo do café)."""
    refs: np.ndarray | None = None


@dataclass
class RegressionSuite:
    """Conjuntos congelados que nenhuma candidata pode piorar. Nunca treinam nem escolhem."""

    sets: list[RegressionSet] = field(default_factory=list)
    class_names: tuple[str, ...] = ()
    encoder: str = ""

    def add(self, s: RegressionSet) -> None:
        if any(x.name == s.name for x in self.sets):
            raise ValueError(f"Conjunto de regressão {s.name!r} já existe.")
        self.sets.append(s)

    def forget(self, prefix: str) -> int:
        tirados = 0
        for s in self.sets:
            if s.refs is None:
                continue
            fica = np.array([not str(r).startswith(prefix) for r in s.refs], dtype=bool)
            tirados += int((~fica).sum())
            s.embeddings, s.labels, s.refs = s.embeddings[fica], s.labels[fica], s.refs[fica]
        self.sets = [s for s in self.sets if len(s.labels) and not s.name.startswith(prefix)]
        return tirados

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "names": np.array([s.name for s in self.sets]),
            "gates": np.array([s.gate for s in self.sets]),
            "class_names": np.array(self.class_names),
            "encoder": np.array(self.encoder),
        }
        for i, s in enumerate(self.sets):
            arrays[f"emb_{i}"] = s.embeddings.astype(np.float16)
            arrays[f"lab_{i}"] = s.labels.astype(np.int64)
            arrays[f"ref_{i}"] = (np.asarray(s.refs, dtype="<U160") if s.refs is not None
                                  else np.zeros(0, dtype="<U160"))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> RegressionSuite:
        with np.load(path, allow_pickle=False) as z:
            sets = []
            for i, (nome, gate) in enumerate(zip(z["names"], z["gates"], strict=True)):
                refs = z[f"ref_{i}"]
                sets.append(RegressionSet(str(nome), z[f"emb_{i}"].astype(np.float64),
                                          z[f"lab_{i}"].astype(np.int64), bool(gate),
                                          refs if len(refs) else None))
            return cls(sets, tuple(str(c) for c in z["class_names"]), str(z["encoder"]))


# ---------------------------------------------------------------------------------------------
# Métricas e comparação pareada
# ---------------------------------------------------------------------------------------------


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    recalls = [float((pred[y == c] == c).mean()) for c in range(n_classes) if (y == c).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    confianca = probs.max(axis=1)
    acerto = probs.argmax(axis=1) == y
    bordas = np.linspace(0.0, 1.0, bins + 1)
    erro = 0.0
    for a, b in itertools.pairwise(bordas):
        dentro = (confianca > a) & (confianca <= b)
        if dentro.any():
            erro += dentro.mean() * abs(float(acerto[dentro].mean() - confianca[dentro].mean()))
    return float(erro)


def metrics(probs: np.ndarray, y: np.ndarray, class_names: tuple[str, ...]) -> dict:
    """Acurácia, acurácia balanceada, revocação por classe, alarme falso em sadia, ECE e NLL."""
    pred = probs.argmax(axis=1)
    c = len(class_names)
    saida = {
        "n": int(len(y)),
        "acuracia": round(float((pred == y).mean()), 4),
        "acuracia_balanceada": round(balanced_accuracy(y, pred, c), 4),
        "ece": round(expected_calibration_error(probs, y), 4),
        "nll": round(float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean()), 4),
        "revocacao": {class_names[k]: {"n": int((y == k).sum()),
                                       "valor": round(float((pred[y == k] == k).mean()), 4)}
                      for k in range(c) if (y == k).any()},
    }
    if HEALTHY in class_names:
        h = class_names.index(HEALTHY)
        sadias = y == h
        doentes = ~sadias
        if sadias.any():
            saida["alarme_falso_sadia"] = round(float((pred[sadias] != h).mean()), 4)
        if doentes.any():
            saida["doenca_perdida"] = round(float((pred[doentes] == h).mean()), 4)
    return saida


@dataclass(frozen=True)
class GateCriteria:
    """Quando a candidata pode substituir a cabeça em produção.

    Dois tipos de critério, porque os conjuntos têm tamanhos muito diferentes:

    * **não-inferioridade no agregado** (o ônus da prova é da candidata): juntando todos os
      conjuntos que bloqueiam, o limite inferior do IC de 95% da diferença de acurácia
      balanceada não pode ficar abaixo de ``-margem_acuracia``. O agregado tem amostra para
      decidir; um conjunto pequeno sozinho não teria — com 116 folhas sadias no teste do café,
      o IC de uma diferença de duas folhas já tem ~3 pontos de largura, e uma margem menor que
      a resolução recusaria toda candidata, boa ou ruim;
    * **piora demonstrada** (o IC inteiro do lado ruim) em cada conjunto. Só a demonstrada:
      o teste de uma rodada de campo tem dezenas de folhas, e uma margem pontual de 3% ali é
      uma folha e meia — medido no BRACOL (15/10/2026): a rodada recusada por "cair 6,6%"
      num teste de 49 folhas **melhorava** de 86,0% para 89,6% no teste final de 336;
    * no agregado, por classe e no alarme falso em sadia, a piora demonstrada **e** a piora
      pontual acima da margem (aí a amostra resolve).
    """

    margem_acuracia: float = 0.02
    """Não-inferioridade da acurácia balanceada no agregado (limite inferior do IC)."""
    margem_classe: float = 0.05
    """Queda pontual máxima da revocação de cada classe (agregada, com ``min_n_classe``)."""
    min_n_classe: int = 30
    margem_alarme_falso: float = 0.02
    """Aumento pontual máximo do alarme falso em folha sadia (agregado e no campo) — no
    café, o erro que mais custa: calda numa planta sadia."""
    limiar_cauda: float = 0.9
    margem_cauda: float = 0.005
    """Cauda de doença em folha sadia: a fração de folhas sadias com P(doença) acima de
    ``limiar_cauda`` não pode subir mais que isto (nem de forma demonstrada), no agregado e no
    campo. É o erro que o **Rover** multiplica: um quadro tem ~70 recortes, e um só recorte
    sadio com doença "certa" vira evento. Medido em 16/10/2026: uma cabeça com a mesma acurácia
    por folha, mas cauda de 1,2% → 2,4% nas sadias, levou o alarme falso do Rover no mosaico de
    103/900 para 163/900 quadros e a fusão por planta de 631/720 para 432/720 — e o portão
    sem este critério a aprovou."""
    exige_ganho_campo: bool = True
    """A candidata tem de ganhar no campo (diferença média > 0): sem ganho, não há por que
    trocar o que já foi medido."""
    confianca: float = 0.95
    reamostragens: int = 2000
    seed: int = 0


def _ic(valores: np.ndarray, confianca: float) -> tuple[float, float]:
    a = (1 - confianca) / 2
    return float(np.quantile(valores, a)), float(np.quantile(valores, 1 - a))


def paired_comparison(p_atual: np.ndarray, p_nova: np.ndarray, y: np.ndarray,
                      class_names: tuple[str, ...], criteria: GateCriteria) -> dict:
    """Bootstrap pareado (os mesmos exemplos nas duas) das diferenças (nova menos atual)."""
    rng = np.random.default_rng(criteria.seed)
    c = len(class_names)
    pa, pn = p_atual.argmax(axis=1), p_nova.argmax(axis=1)
    n = len(y)
    reamostras = rng.integers(0, n, size=(criteria.reamostragens, n))
    dif_bal = np.array([balanced_accuracy(y[r], pn[r], c) - balanced_accuracy(y[r], pa[r], c)
                        for r in reamostras])
    dif_bal = dif_bal[np.isfinite(dif_bal)]
    saida = {
        "acuracia_balanceada": {
            "atual": round(balanced_accuracy(y, pa, c), 4),
            "nova": round(balanced_accuracy(y, pn, c), 4),
            "diferenca": round(balanced_accuracy(y, pn, c) - balanced_accuracy(y, pa, c), 4),
            "ic": [round(v, 4) for v in _ic(dif_bal, criteria.confianca)],
        },
        "concordancia": {"so_atual_acerta": int(((pa == y) & (pn != y)).sum()),
                         "so_nova_acerta": int(((pa != y) & (pn == y)).sum())},
        "classes": {},
    }
    for k in range(c):
        idx = np.flatnonzero(y == k)
        if not len(idx):
            continue
        ra = rng.integers(0, len(idx), size=(criteria.reamostragens, len(idx)))
        acerto_a = (pa[idx] == k).astype(np.float64)
        acerto_n = (pn[idx] == k).astype(np.float64)
        dif = acerto_n[ra].mean(axis=1) - acerto_a[ra].mean(axis=1)
        saida["classes"][class_names[k]] = {
            "n": int(len(idx)), "atual": round(float(acerto_a.mean()), 4),
            "nova": round(float(acerto_n.mean()), 4),
            "ic": [round(v, 4) for v in _ic(dif, criteria.confianca)],
        }
    if HEALTHY in class_names:
        h = class_names.index(HEALTHY)
        idx = np.flatnonzero(y == h)
        if len(idx):
            ra = rng.integers(0, len(idx), size=(criteria.reamostragens, len(idx)))
            fa_a = (pa[idx] != h).astype(np.float64)
            fa_n = (pn[idx] != h).astype(np.float64)
            dif = fa_n[ra].mean(axis=1) - fa_a[ra].mean(axis=1)
            saida["alarme_falso_sadia"] = {
                "atual": round(float(fa_a.mean()), 4), "nova": round(float(fa_n.mean()), 4),
                "ic": [round(v, 4) for v in _ic(dif, criteria.confianca)]}
            doenca_a = np.delete(p_atual[idx], h, axis=1).max(axis=1)
            doenca_n = np.delete(p_nova[idx], h, axis=1).max(axis=1)
            ca = (doenca_a > criteria.limiar_cauda).astype(np.float64)
            cn = (doenca_n > criteria.limiar_cauda).astype(np.float64)
            dif = cn[ra].mean(axis=1) - ca[ra].mean(axis=1)
            saida["cauda_sadia"] = {
                "limiar": criteria.limiar_cauda, "n": int(len(idx)),
                "atual": round(float(ca.mean()), 4), "nova": round(float(cn.mean()), 4),
                "ic": [round(v, 4) for v in _ic(dif, criteria.confianca)]}
    return saida


def edge_drift(atual: DiagnosisHead, nova: DiagnosisHead, embeddings: np.ndarray,
               labels: np.ndarray, profile: dict) -> dict:
    """Quanto a evidência do perfil da borda se move nas folhas **sadias** com a cabeça nova.

    O alarme falso do Rover é a média da LLR (tabelas do ``.edge.json``) nos recortes sadios;
    o limiar de evento foi medido com a cabeça atual. ``recalibrar`` quando a média de alguma
    doença sobe mais que a metade da faixa entre quase-evento e evento.
    """
    regra = profile.get("evidence") or {}
    tabelas = regra.get("llr_tables") or {}
    if not tabelas or HEALTHY not in atual.class_names:
        return {"avaliado": False}
    bordas = np.asarray(regra["bin_edges"], dtype=np.float64)
    clip = float(regra["llr_clip"])
    sadias = np.asarray(labels) == atual.class_names.index(HEALTHY)
    if not sadias.any():
        return {"avaliado": False}
    x = np.asarray(embeddings, dtype=np.float64)[sadias]
    pa, pn = atual.mlp.probabilities(x), nova.mlp.probabilities(x)

    def llr(p: np.ndarray, tabela) -> float:
        i = np.clip(np.searchsorted(bordas, p, side="right") - 1, 0, len(bordas) - 2)
        return float(np.clip(np.asarray(tabela)[i], -clip, clip).mean())

    evento = regra.get("event_evidence")
    quase = regra.get("near_miss_evidence")
    folga = (abs(evento - quase) / 2 if evento is not None and quase is not None else 0.05 * clip)
    por_doenca = {}
    for doenca, tabela in tabelas.items():
        k = atual.class_names.index(doenca)
        antes, depois = llr(pa[:, k], tabela), llr(pn[:, k], tabela)
        por_doenca[doenca] = {"antes": round(antes, 4), "depois": round(depois, 4),
                              "subida": round(depois - antes, 4)}
    subida = max(v["subida"] for v in por_doenca.values())
    return {"avaliado": True, "n_sadias": int(sadias.sum()), "por_doenca": por_doenca,
            "folga": round(folga, 4), "recalibrar": bool(subida > folga)}


@dataclass
class GateReport:
    decision: str
    """``APROVADA`` ou ``REPROVADA``."""
    reasons: list[str]
    sets: dict[str, dict]
    field: dict | None
    criteria: dict
    pooled: dict | None = None
    """Todos os conjuntos que bloqueiam, juntos (a não-inferioridade e as classes)."""

    def to_dict(self) -> dict:
        return asdict(self)


def _piora(rotulo: str, comp: dict, margem: float) -> list[str]:
    """Queda demonstrada (IC inteiro abaixo de zero) ou pontual acima da margem."""
    d = comp["diferenca"] if "diferenca" in comp else comp["nova"] - comp["atual"]
    if comp["ic"][1] < 0:
        return [f"{rotulo}: piora demonstrada ({d:+.1%}, IC 95% [{comp['ic'][0]:+.1%}, "
                f"{comp['ic'][1]:+.1%}])"]
    if d < -margem:
        return [f"{rotulo}: cai {-d:.1%} (margem {margem:.0%})"]
    return []


def _cauda(rotulo: str, c: dict | None, margem: float) -> list[str]:
    if not c:
        return []
    d = c["nova"] - c["atual"]
    texto = (f"cauda de doença em sadia (P > {c['limiar']:.2f}) {c['atual']:.1%} → "
             f"{c['nova']:.1%}")
    if c["ic"][0] > 0:
        return [f"{rotulo}: {texto}, aumento demonstrado — o Rover multiplica isto por recorte"]
    if d > margem:
        return [f"{rotulo}: {texto} (margem {margem:.1%}) — o Rover multiplica isto por recorte"]
    return []


def _alarme(rotulo: str, fa: dict | None, margem: float) -> list[str]:
    if not fa:
        return []
    d = fa["nova"] - fa["atual"]
    if fa["ic"][0] > 0:
        return [f"{rotulo}: alarme falso em sadia sobe, demonstrado ({fa['atual']:.1%} → "
                f"{fa['nova']:.1%})"]
    if d > margem:
        return [f"{rotulo}: alarme falso em sadia sobe {d:.1%} ({fa['atual']:.1%} → "
                f"{fa['nova']:.1%}; margem {margem:.0%})"]
    return []


def compare(atual: DiagnosisHead, nova: DiagnosisHead, suite: RegressionSuite,
            campo: RegressionSet | None, criteria: GateCriteria = GateCriteria()) -> GateReport:
    """O portão: a candidata contra a cabeça em produção, nos mesmos exemplos."""
    nomes = atual.class_names
    motivos: list[str] = []
    conjuntos: dict[str, dict] = {}

    def comparacao(x, y) -> dict:
        pa, pn = atual.mlp.probabilities(x), nova.mlp.probabilities(x)
        comp = paired_comparison(pa, pn, y, nomes, criteria)
        comp["metricas_atual"] = metrics(pa, y, nomes)
        comp["metricas_nova"] = metrics(pn, y, nomes)
        return comp

    for s in suite.sets:
        comp = comparacao(s.embeddings, s.labels)
        comp["bloqueia"] = s.gate
        conjuntos[s.name] = comp
        if s.gate:  # margem infinita: só a piora demonstrada (ver GateCriteria)
            motivos += _piora(f"{s.name}: acurácia balanceada", comp["acuracia_balanceada"],
                              float("inf"))
    bloqueiam = [s for s in suite.sets if s.gate]
    agregado = None
    if bloqueiam:
        agregado = comparacao(np.vstack([s.embeddings for s in bloqueiam]),
                              np.concatenate([s.labels for s in bloqueiam]))
        agregado["conjuntos"] = [s.name for s in bloqueiam]
        baixo = agregado["acuracia_balanceada"]["ic"][0]
        if baixo < -criteria.margem_acuracia:
            motivos.append(f"agregado: não-inferioridade não demonstrada — a acurácia "
                           f"balanceada pode cair {-baixo:.1%} (margem "
                           f"{criteria.margem_acuracia:.0%})")
        for classe, v in agregado["classes"].items():
            if v["n"] >= criteria.min_n_classe:
                motivos += _piora(f"agregado: revocação de {classe}", v, criteria.margem_classe)
        motivos += _alarme("agregado", agregado.get("alarme_falso_sadia"),
                           criteria.margem_alarme_falso)
        motivos += _cauda("agregado", agregado.get("cauda_sadia"), criteria.margem_cauda)
    relatorio_campo = None
    if campo is not None and len(campo.labels):
        relatorio_campo = comparacao(campo.embeddings, campo.labels)
        ganho = relatorio_campo["acuracia_balanceada"]["diferenca"]
        if criteria.exige_ganho_campo and ganho <= 0:
            motivos.append(f"campo ({campo.name}): sem ganho ({ganho:+.1%}) — não há por que "
                           "trocar a cabeça medida")
        motivos += _alarme(f"campo ({campo.name})", relatorio_campo.get("alarme_falso_sadia"),
                           criteria.margem_alarme_falso)
        motivos += _cauda(f"campo ({campo.name})", relatorio_campo.get("cauda_sadia"),
                          criteria.margem_cauda)
    elif criteria.exige_ganho_campo:
        motivos.append("sem conjunto de campo para medir o ganho")
    return GateReport("REPROVADA" if motivos else "APROVADA", motivos, conjuntos,
                      relatorio_campo, asdict(criteria), agregado)


# ---------------------------------------------------------------------------------------------
# A adaptação
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptConfig:
    epocas: int = 25
    lote: int = 64
    taxa: float = 5e-4
    """Adam. Pequena: a cabeça já sabe; o campo ajusta."""
    replay_por_lote: int = 64
    """Exemplos da memória em cada lote (a proporção 1:1 é a do DER++)."""
    destilacao: float = 1.0
    """Peso da destilação nos exemplos da memória (0 = só o rótulo: replay simples)."""
    temperatura: float = 2.0
    ancora: float = 1e-3
    """Peso da L2-SP (distância aos pesos de partida)."""
    suavizacao: float = 0.0
    """Suavização de rótulo na entropia cruzada (alvo ``1 - s`` na classe certa, ``s`` dividido
    entre as outras). Segura a confiança: sem ela, treinar com foto ruim empurrou a cauda das
    sadias para cima (ver ``GateCriteria.margem_cauda``)."""
    camadas: str = "todas"
    """``todas`` ou ``ultima`` (só a camada de saída aprende)."""
    tolerancia_validacao: float | None = 0.01
    """Seleção consciente de regressão: só vale a época em que a acurácia balanceada na
    validação do que já se sabia não caiu mais que isto — metade da margem de
    não-inferioridade do portão (fator de segurança 2: a validação é uma estimativa, e o
    portão mede em outro conjunto). ``None`` desliga (a época sai só pela média)."""
    tolerancia_alarme_falso: float = 0.01
    """Idem para o aumento do alarme falso em sadia nessa validação (metade da margem do
    portão)."""
    seed: int = 0


@dataclass
class AdaptResult:
    head: DiagnosisHead
    epoch: int
    curve: list[dict]
    config: dict


def _alvo(y: np.ndarray, c: int, suavizacao: float) -> np.ndarray:
    """One-hot, ou suavizado: ``1 - s`` na classe certa e ``s / (c - 1)`` em cada outra."""
    alvo = np.full((len(y), c), suavizacao / max(1, c - 1))
    alvo[np.arange(len(y)), y] = 1.0 - suavizacao
    return alvo


def _pesos_de_classe(y: np.ndarray, c: int) -> np.ndarray:
    contagem = np.bincount(y, minlength=c).astype(np.float64)
    pesos = np.where(contagem > 0, contagem.sum() / np.maximum(contagem, 1) / max(1, (
        contagem > 0).sum()), 0.0)
    return pesos


def adapt(head: DiagnosisHead, x_campo: np.ndarray, y_campo: np.ndarray,
          memoria: ReplayMemory | None, x_val: np.ndarray, y_val: np.ndarray,
          cfg: AdaptConfig = AdaptConfig(),
          old_val: tuple[np.ndarray, np.ndarray] | None = None) -> AdaptResult:
    """Ajusta a cabeça ao campo com replay, destilação e âncora; devolve a melhor época.

    ``x_campo`` são embeddings crus (já com as orientações do TTA como aumento, se houver);
    ``x_val``/``y_val`` é a validação do campo. ``old_val`` (embeddings, rótulos) é a
    validação do que já se sabia, que a cabeça **não** treinou; sem ela, só o campo escolhe
    a época e não há seleção consciente de regressão.
    """
    rng = np.random.default_rng(cfg.seed)
    mlp = head.mlp
    c = len(head.class_names)
    p0 = _parametros(mlp)
    params = [p.copy() for p in p0]
    n_camadas = len(mlp.weights)
    treinaveis = (list(range(len(params))) if cfg.camadas == "todas"
                  else [2 * n_camadas, 2 * n_camadas + 1])
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    passo = 0

    xs = standardize(mlp, x_campo)
    ys = np.asarray(y_campo, dtype=np.int64)
    pesos = _pesos_de_classe(ys, c)
    if memoria is not None and len(memoria):
        xm = standardize(mlp, memoria.embeddings)
        ym = memoria.labels
        tm = memoria.logits
        pesos_m = _pesos_de_classe(ym, c)
        idx_por_classe = [np.flatnonzero(ym == k) for k in range(c)]
        idx_por_classe = [i for i in idx_por_classe if len(i)]
    xv = standardize(mlp, x_val) if len(y_val) else None
    xmv = ymv = None
    if old_val is not None and len(old_val[1]):
        xmv, ymv = standardize(mlp, old_val[0]), np.asarray(old_val[1], dtype=np.int64)

    sadia = head.class_names.index(HEALTHY) if HEALTHY in head.class_names else None

    def avalia(ps) -> dict:
        m = _com_parametros(mlp, ps)
        saida = {}
        if xv is not None:
            pred = softmax(forward(m, ps, xv)[0]).argmax(1)
            saida["campo"] = balanced_accuracy(np.asarray(y_val), pred, c)
        if xmv is not None:
            pred = softmax(forward(m, ps, xmv)[0]).argmax(1)
            saida["antigo"] = balanced_accuracy(ymv, pred, c)
            if sadia is not None and (ymv == sadia).any():
                saida["antigo_alarme_falso"] = float((pred[ymv == sadia] != sadia).mean())
        valores = [saida[k] for k in ("campo", "antigo") if k in saida and np.isfinite(saida[k])]
        saida["escolha"] = float(np.mean(valores)) if valores else 0.0
        return saida

    curva = [{"epoca": 0, **avalia(params)}]
    inicio = curva[0]

    def admissivel(ponto: dict) -> bool:
        tol = cfg.tolerancia_validacao
        if tol is None or "antigo" not in ponto:
            return True
        if ponto["antigo"] < inicio["antigo"] - tol:
            return False
        fa = ponto.get("antigo_alarme_falso")
        return fa is None or fa <= inicio["antigo_alarme_falso"] + cfg.tolerancia_alarme_falso

    melhor, melhor_epoca = [p.copy() for p in params], 0
    temp = cfg.temperatura
    for epoca in range(1, cfg.epocas + 1):
        ordem = rng.permutation(len(ys))
        for ini in range(0, len(ordem), cfg.lote):
            lote = ordem[ini:ini + cfg.lote]
            xb, yb = xs[lote], ys[lote]
            wb = pesos[yb]
            logits, cache = forward(mlp, params, xb)
            p = softmax(logits)
            d = p - _alvo(yb, c, cfg.suavizacao)
            d *= wb[:, None] / max(1.0, wb.sum())
            grads = backward(mlp, params, cache, d)
            if memoria is not None and len(memoria) and cfg.replay_por_lote > 0:
                # Amostra balanceada por classe: a classe rara do passado não some da memória.
                classes = rng.integers(0, len(idx_por_classe), cfg.replay_por_lote)
                mb = np.array([idx_por_classe[k][rng.integers(0, len(idx_por_classe[k]))]
                               for k in classes])
                lm, cm = forward(mlp, params, xm[mb])
                pm = softmax(lm)
                dm = pm - _alvo(ym[mb], c, cfg.suavizacao)
                wm = pesos_m[ym[mb]]
                dm *= wm[:, None] / max(1.0, wm.sum())
                if cfg.destilacao > 0:
                    alvo = softmax(tm[mb] / temp)
                    dm += cfg.destilacao * temp * (softmax(lm / temp) - alvo) / len(mb)
                for i, g in enumerate(backward(mlp, params, cm, dm)):
                    grads[i] += g
            passo += 1
            for i in treinaveis:
                g = grads[i] + 2.0 * cfg.ancora * (params[i] - p0[i])
                m1[i] = b1 * m1[i] + (1 - b1) * g
                m2[i] = b2 * m2[i] + (1 - b2) * g * g
                mh = m1[i] / (1 - b1 ** passo)
                vh = m2[i] / (1 - b2 ** passo)
                params[i] = params[i] - cfg.taxa * mh / (np.sqrt(vh) + eps)
        ponto = {"epoca": epoca, **avalia(params)}
        ponto["admissivel"] = admissivel(ponto)
        curva.append(ponto)
        if ponto["admissivel"] and ponto["escolha"] > curva[melhor_epoca]["escolha"] + 1e-9:
            melhor, melhor_epoca = [p.copy() for p in params], epoca
    # Nenhuma época melhor que a de partida: a cabeça volta intacta — os mesmos bytes, o
    # mesmo SHA-256 (o perfil da borda e o portão sabem que nada mudou).
    nova = head if melhor_epoca == 0 else replace(
        head, mlp=_com_parametros(mlp, melhor), trained_on=head.trained_on + int(len(ys)))
    return AdaptResult(nova, melhor_epoca, [{k: (round(v, 4) if isinstance(v, float) else v)
                                             for k, v in p.items()} for p in curva], asdict(cfg))


# ---------------------------------------------------------------------------------------------
# Apoio
# ---------------------------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def continual_dir(weights_path: Path) -> Path:
    """``models/_continuo/<pesos>/``: memória, portão, histórico, rodadas e cabeças antigas."""
    return Path(weights_path).parent / "_continuo" / Path(weights_path).stem


def status(weights_path: Path, head_path: Path, limit: int = 50) -> dict:
    """O que a API mostra: a cabeça em produção, o histórico, a memória e o portão."""
    pasta = continual_dir(weights_path)
    head = Path(head_path)
    saida: dict = {"head_sha256": file_sha256(head) if head.exists() else None,
                   "history": read_ledger(pasta / "historico.jsonl")[-limit:],
                   "memory": None, "regression_sets": [], "validation_sets": []}
    if (pasta / "validacao.npz").exists():
        saida["validation_sets"] = [
            {"name": s.name, "n": int(len(s.labels))}
            for s in RegressionSuite.load(pasta / "validacao.npz").sets]
    if (pasta / "replay.npz").exists():
        memoria = ReplayMemory.load(pasta / "replay.npz")
        saida["memory"] = {"examples": len(memoria), "per_class": memoria.per_class,
                           "by_source": memoria.counts()}
    if (pasta / "regressao.npz").exists():
        saida["regression_sets"] = [
            {"name": s.name, "n": int(len(s.labels)), "gate": bool(s.gate)}
            for s in RegressionSuite.load(pasta / "regressao.npz").sets]
    return saida


def head_sha256(head: DiagnosisHead, scratch: Path) -> str:
    """SHA-256 dos bytes que :meth:`DiagnosisHead.save` grava (o que vai ao Rover)."""
    head.save(scratch)
    return file_sha256(scratch)


def append_ledger(path: Path, entrada: dict) -> None:
    """Uma linha por acontecimento: rodada, promoção, reversão, monitoramento, esquecimento."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entrada, ensure_ascii=False, default=str) + "\n")


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(linha) for linha in path.read_text(encoding="utf-8").splitlines()
            if linha.strip()]
