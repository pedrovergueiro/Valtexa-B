"""A plasticidade do dNATY no aprendizado contínuo da cabeça (Nível 2, 5.1).

Rede ajustada rodada após rodada perde a capacidade de aprender coisa nova: neurônios
param de disparar e o rank das features cai (Dohare et al., Nature 2024). O dNATY 2.2
traz o remédio do *continual backprop* para a ``DynamicMLP`` — que é exatamente a cabeça
do café (exportada com o BatchNorm dobrado, :meth:`EmbeddingMLP.from_dnaty`):

1. :func:`para_dnaty` remonta a ``DynamicMLP`` a partir da cabeça (BatchNorm identidade:
   média 0, variância ``1 - eps``, escala 1 — o caminho inverso do ``from_dnaty``);
2. o ``NeuronUtilityTracker`` do dNATY mede a utilidade de cada neurônio oculto
   (``|ativação| x ||pesos de saída||``) no campo e na memória de replay;
3. o ``reinit_dormant`` do dNATY renasce os dormentes: pesos de entrada novos, de saída
   **zerados** — a função da rede só perde a contribuição desprezível do dormente;
4. só as linhas e colunas renascidas voltam para a cabeça em float64 (o resto fica com os
   mesmos bits) e o treinador de sempre (:func:`app.inference.continual.adapt`: replay,
   destilação, L2-SP, seleção consciente de regressão) ajusta a partir dela;
5. o portão compara com a cabeça **instalada**, não com a renascida.

Por cima da fração do dNATY, ``limiar_relativo`` exige que o neurônio esteja dormente de
fato (utilidade abaixo dessa fração da mediana da camada): a cabeça do café é pequena e
treinada, e renascer neurônio útil é esquecer.

Precisa de ``torch`` e do pacote ``dnaty`` (licença BSL-1.1: uso comercial pede licença
própria). Só treino: nem a API nem o Rover importam isto.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from app.inference.continual import AdaptConfig, AdaptResult, ReplayMemory, adapt, standardize
from app.inference.heads import DiagnosisHead, EmbeddingMLP

MOTOR_NUMPY = "numpy"
MOTOR_DNATY = "dnaty-plasticidade"


@dataclass(frozen=True)
class Plasticidade:
    fracao: float = 0.05
    """No máximo esta fração dos neurônios de cada camada renasce por rodada (a do dNATY)."""
    limiar_relativo: float | None = 0.1
    """Só renasce quem tem utilidade abaixo disto vezes a mediana da camada. ``None`` = a
    fração pura do dNATY (os menos úteis, dormentes ou não)."""
    max_amostras: int = 4096
    seed: int = 0


def _importa():
    try:
        import torch
        from dnaty.core.arch import DynamicMLP
        from dnaty.training.plasticity import (
            NeuronUtilityTracker,
            effective_rank,
            reinit_dormant,
        )
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise SystemExit(f"O motor {MOTOR_DNATY} precisa de torch e do dnaty: {exc}") from exc
    return torch, DynamicMLP, NeuronUtilityTracker, effective_rank, reinit_dormant


def para_dnaty(mlp: EmbeddingMLP):
    """A ``DynamicMLP`` do dNATY que computa o mesmo que a cabeça (em float32)."""
    torch, dynamic_mlp, *_ = _importa()
    tamanhos = mlp.layer_sizes
    modelo = dynamic_mlp(tamanhos[:-1], list(mlp.activations), n_classes=tamanhos[-1])
    with torch.no_grad():
        for i, (w, b) in enumerate(zip(mlp.weights, mlp.biases, strict=True)):
            linear, norma = modelo.net[3 * i], modelo.net[3 * i + 1]
            linear.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            linear.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
            norma.running_mean.zero_()
            norma.running_var.fill_(1.0 - norma.eps)
            norma.weight.fill_(1.0)
            norma.bias.zero_()
        saida = modelo.net[3 * len(mlp.weights)]
        saida.weight.copy_(torch.as_tensor(mlp.out_weight, dtype=torch.float32))
        saida.bias.copy_(torch.as_tensor(mlp.out_bias, dtype=torch.float32))
        for origem, destino, proj in mlp.skips:
            camada = None
            if proj >= 0:
                w = mlp.skip_weights[proj]
                camada = torch.nn.Linear(w.shape[1], w.shape[0])
                camada.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
                camada.bias.copy_(torch.as_tensor(mlp.skip_biases[proj], dtype=torch.float32))
            modelo.add_skip_connection(origem, destino, camada)
    return modelo.eval()


def _entrada(head: DiagnosisHead, embeddings: np.ndarray, plast: Plasticidade):
    torch = _importa()[0]
    x = np.asarray(embeddings, dtype=np.float64)
    if len(x) > plast.max_amostras:
        x = x[np.random.default_rng(plast.seed).choice(len(x), plast.max_amostras,
                                                       replace=False)]
    return torch.as_tensor(standardize(head.mlp, x), dtype=torch.float32)


def rank_efetivo(head: DiagnosisHead, embeddings: np.ndarray,
                 plast: Plasticidade = Plasticidade()) -> float:
    """O rank efetivo da última camada oculta (``effective_rank`` do dNATY)."""
    _torch, _m, _t, effective_rank, _r = _importa()
    return round(float(effective_rank(para_dnaty(head.mlp), _entrada(head, embeddings, plast))),
                 2)


def renasce(head: DiagnosisHead, embeddings: np.ndarray, plast: Plasticidade = Plasticidade()
            ) -> tuple[DiagnosisHead, dict]:
    """A cabeça com os neurônios dormentes renascidos, e o que foi feito."""
    torch, _m, neuron_utility_tracker, effective_rank, reinit_dormant = _importa()
    modelo = para_dnaty(head.mlp)
    x = _entrada(head, embeddings, plast)
    rastreador = neuron_utility_tracker(modelo, ema_beta=1.0)  # uma passada com tudo: exata
    rastreador.update(x)
    utilidades = rastreador.utilities()
    mascaras = rastreador.dormant_mask(plast.fracao)
    if plast.limiar_relativo is not None:
        mascaras = [m & (u < plast.limiar_relativo * torch.median(u))
                    for m, u in zip(mascaras, utilidades, strict=True)]
    rank_antes = float(effective_rank(modelo, x))
    renascidos = int(reinit_dormant(modelo, mascaras, seed=plast.seed))
    relatorio = {
        "motor": MOTOR_DNATY, "fracao": plast.fracao, "limiar_relativo": plast.limiar_relativo,
        "amostras": len(x), "renascidos": renascidos,
        "por_camada": [int(m.sum()) for m in mascaras],
        "utilidade_mediana": [round(float(torch.median(u)), 5) for u in utilidades],
        "rank_efetivo_antes": round(rank_antes, 2),
    }
    if renascidos == 0:
        relatorio["rank_efetivo_depois"] = relatorio["rank_efetivo_antes"]
        return head, relatorio
    relatorio["rank_efetivo_depois"] = round(float(effective_rank(modelo, x)), 2)
    novo = EmbeddingMLP.from_dnaty(modelo, head.mlp.mean, head.mlp.std)
    pesos = [np.array(w, dtype=np.float64) for w in head.mlp.weights]
    vieses = [np.array(b, dtype=np.float64) for b in head.mlp.biases]
    saida = np.array(head.mlp.out_weight, dtype=np.float64)
    atalhos = [np.array(w, dtype=np.float64) for w in head.mlp.skip_weights]
    for j, mascara in enumerate(mascaras):
        idx = np.flatnonzero(mascara.numpy())
        if not len(idx):
            continue
        pesos[j][idx] = novo.weights[j][idx]
        vieses[j][idx] = novo.biases[j][idx]
        if j + 1 < len(pesos):
            pesos[j + 1][:, idx] = 0.0
        else:
            saida[:, idx] = 0.0
        for origem, _destino, proj in head.mlp.skips:
            if origem == j + 1 and proj >= 0:
                atalhos[proj][:, idx] = 0.0
    mlp = replace(head.mlp, weights=tuple(pesos), biases=tuple(vieses), out_weight=saida,
                  skip_weights=tuple(atalhos))
    return replace(head, mlp=mlp), relatorio


def adapta(head: DiagnosisHead, x_campo: np.ndarray, y_campo: np.ndarray,
           memoria: ReplayMemory | None, x_val: np.ndarray, y_val: np.ndarray,
           cfg: AdaptConfig, old_val=None, plast: Plasticidade | None = None
           ) -> tuple[AdaptResult, dict]:
    """O :func:`adapt` de sempre, partindo da cabeça com os dormentes renascidos.

    Se nenhuma época bate a de partida, volta a cabeça **instalada** (os mesmos bytes), não a
    renascida: renascer sem aprender não é atualização.
    """
    plast = plast or Plasticidade(seed=cfg.seed)
    base = np.asarray(x_campo, dtype=np.float64)
    if memoria is not None and len(memoria):
        base = np.vstack([base, memoria.embeddings.astype(np.float64)])
    renascida, relatorio = renasce(head, base, plast)
    resultado = adapt(renascida, x_campo, y_campo, memoria, x_val, y_val, cfg, old_val)
    if resultado.epoch == 0:
        resultado = AdaptResult(head, 0, resultado.curve, resultado.config)
        relatorio["sem_ganho"] = True
    relatorio["rank_efetivo_final"] = rank_efetivo(resultado.head, base, plast)
    return resultado, relatorio
