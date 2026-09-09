"""Detecção de novidade: identificar o que o modelo nunca viu.

Um softmax é fechado — ele sempre devolve uma das classes que aprendeu, mesmo
diante de uma doença que não está no catálogo, de uma deficiência nutricional,
de cigarrinha, ou de uma foto de solo. Esse é justamente o material mais valioso
para crescer o dataset próprio, e é o que este módulo separa.

A ideia: a penúltima camada do modelo (ver ``EMBEDDING_LAYER``) transforma cada
imagem num vetor de 128 dimensões — a "assinatura visual" dela. Imagens da mesma
classe caem perto umas das outras nesse espaço. Medindo a distância de uma imagem
nova até o centro de cada classe conhecida, dá para responder algo que o softmax
não responde: *"isto se parece com alguma coisa que eu já vi?"*

A distância usada é a de Mahalanobis, que leva em conta o formato da nuvem de
pontos em vez de tratar todas as direções como iguais.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PERCENTILE = 99.0
"""Percentil das distâncias de treino que vira o limiar.

99 significa: 1% das imagens que o modelo *já viu no treino* seriam marcadas
como novidade. Em campo a taxa é naturalmente maior — este número é um orçamento
do tempo do agrônomo, não uma promessa de acurácia.
"""

DEFAULT_SHRINKAGE = 0.1
"""Regularização da matriz de covariância.

Puxa a covariância na direção de uma matriz diagonal. Sem isso, com poucas
amostras por classe a matriz fica mal-condicionada e a inversão explode.
"""


@dataclass(frozen=True)
class NoveltyReference:
    """Assinatura estatística das classes que o modelo conhece.

    Cada classe tem o **seu próprio** formato de nuvem e o seu próprio limiar.
    A versão anterior usava uma covariância compartilhada entre todas, e isso
    quebrou quando o dataset cresceu: com laboratório, campo e *in-the-wild*
    misturados, a mesma classe passou a ocupar regiões muito distantes do
    espaço, a covariância comum inchou, e uma covariância inchada encolhe a
    distância de **tudo** — inclusive do que é estranho. Na prática, uma foto de
    colmo virava ``Blight`` com 100% de confiança sem ser marcada.
    """

    class_names: tuple[str, ...]
    centroids: np.ndarray
    """``(C, D)`` — o centro de cada classe no espaço de embeddings."""

    precisions: np.ndarray
    """``(C, D, D)`` — a inversa da covariância **de cada classe**."""

    thresholds: np.ndarray
    """``(C,)`` — o limiar de cada classe, no percentil pedido.

    Um limiar por classe é o que torna as distâncias comparáveis entre si. Com
    covariâncias diferentes, a distância bruta até a classe A não significa o
    mesmo que até a classe B; dividida pelo limiar da própria classe, cada uma
    passa a responder "quantas vezes o espalhamento normal desta classe", que é
    a mesma unidade em todas.
    """

    fitted_on: int
    """Quantas imagens foram usadas na calibração."""

    @property
    def threshold(self) -> float:
        """Limiar médio, só para relatório. O cálculo usa os limiares por classe."""
        return float(np.mean(self.thresholds))

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: tuple[str, ...],
        percentile: float = DEFAULT_PERCENTILE,
        shrinkage: float = DEFAULT_SHRINKAGE,
    ) -> NoveltyReference:
        """Calibra a referência a partir dos embeddings do conjunto de treino.

        Ajusta uma covariância **por classe**, com recuo para a covariância
        compartilhada nas classes que não tiverem amostras suficientes — abaixo
        de ``D`` amostras para ``D`` dimensões a matriz da classe é ruído, e o
        encolhimento não salva.

        Raises:
            ValueError: se os dados de entrada forem inconsistentes.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        labels = np.asarray(labels).ravel()

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings deveriam ser 2D, veio {embeddings.shape}.")
        if len(embeddings) != len(labels):
            raise ValueError(f"{len(embeddings)} embeddings para {len(labels)} rótulos.")
        if len(embeddings) <= embeddings.shape[1]:
            raise ValueError(
                f"Amostras de menos: {len(embeddings)} imagens para "
                f"{embeddings.shape[1]} dimensões. Treine com mais dados."
            )

        centroids, deviations = _class_statistics(embeddings, labels, len(class_names))
        shared = _regularised_precision(deviations, shrinkage)
        precisions = _per_class_precisions(labels, deviations, shared, len(class_names), shrinkage)
        thresholds = _class_thresholds(embeddings, labels, centroids, precisions, percentile)

        return cls(
            class_names=tuple(class_names),
            centroids=centroids,
            precisions=precisions,
            thresholds=thresholds,
            fitted_on=len(embeddings),
        )

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Pontua imagens: ``> 1.0`` significa novidade.

        O valor é a menor distância normalizada até uma classe conhecida —
        distância até o centro da classe dividida pelo limiar daquela classe.
        Normalizar por classe é o que impede que uma classe de nuvem larga vire
        um atrator, atraindo para si tudo o que o modelo não conhece; e manter a
        escala em "múltiplos do limiar" deixa o número comparável entre modelos:
        1,4 quer dizer "40% além do limiar", seja no milho ou no café.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        return _min_normalised_distance(
            embeddings, self.centroids, self.precisions, self.thresholds
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            class_names=np.array(self.class_names, dtype=object),
            centroids=self.centroids,
            precisions=self.precisions,
            thresholds=self.thresholds,
            fitted_on=np.array(self.fitted_on),
        )

    @classmethod
    def load(cls, path: Path) -> NoveltyReference:
        """Lê uma referência gravada por :meth:`save`.

        Aceita também o formato anterior, de covariância e limiar únicos: os
        arquivos já gravados em ``models/`` continuam carregando, replicados
        para o formato por classe. Eles não ganham a precisão nova — para isso é
        preciso rodar ``training.fit_novelty`` de novo —, mas a API sobe.

        Raises:
            ValueError: se o arquivo estiver corrompido ou incompleto.
        """
        try:
            with np.load(path, allow_pickle=True) as data:
                names = tuple(str(name) for name in data["class_names"])
                centroids = data["centroids"]

                if "precisions" in data:
                    precisions = data["precisions"]
                    thresholds = data["thresholds"]
                else:
                    precisions = np.repeat(data["precision"][None, :, :], len(names), axis=0)
                    thresholds = np.full(len(names), float(data["threshold"]))

                return cls(
                    class_names=names,
                    centroids=centroids,
                    precisions=precisions,
                    thresholds=thresholds,
                    fitted_on=int(data["fitted_on"]),
                )
        except (KeyError, OSError, ValueError, pickle.UnpicklingError) as exc:
            # UnpicklingError não é ValueError: o numpy a levanta ao tentar ler
            # um arquivo que não é .npz, e sem ela o carregamento derrubaria a API.
            raise ValueError(f"Referência de novidade inválida em {path}: {exc}") from exc


def _class_statistics(
    embeddings: np.ndarray, labels: np.ndarray, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Centro de cada classe e os desvios de cada amostra ao seu próprio centro."""
    dimensions = embeddings.shape[1]
    centroids = np.zeros((num_classes, dimensions), dtype=np.float64)
    deviations = np.empty_like(embeddings)

    for index in range(num_classes):
        mask = labels == index
        if not mask.any():
            # Classe sem amostras (ex.: rótulo previsto no catálogo mas ausente
            # do diretório). Fica no centro dos dados para nunca ser a mais
            # próxima, em vez de virar um centroide na origem.
            centroids[index] = embeddings.mean(axis=0)
            continue
        centroids[index] = embeddings[mask].mean(axis=0)
        deviations[mask] = embeddings[mask] - centroids[index]

    return centroids, deviations


def _regularised_precision(deviations: np.ndarray, shrinkage: float) -> np.ndarray:
    """Inverte a covariância compartilhada, com encolhimento para estabilizar."""
    covariance = deviations.T @ deviations / max(len(deviations) - 1, 1)
    dimensions = covariance.shape[0]

    average_variance = np.trace(covariance) / dimensions
    covariance = (1.0 - shrinkage) * covariance + shrinkage * average_variance * np.eye(dimensions)

    try:
        return np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        # Ainda singular apesar do encolhimento — a pseudo-inversa degrada com
        # elegância em vez de derrubar o carregamento do modelo.
        return np.linalg.pinv(covariance)


def _per_class_precisions(
    labels: np.ndarray,
    deviations: np.ndarray,
    shared: np.ndarray,
    num_classes: int,
    shrinkage: float,
) -> np.ndarray:
    """Uma inversa de covariância por classe, com recuo para a compartilhada.

    O recuo acontece quando a classe tem menos amostras que dimensões: aí a
    covariância própria descreve o ruído da amostra, não o formato da classe, e
    a compartilhada — estimada sobre o conjunto inteiro — é a estimativa menos
    ruim disponível.
    """
    dimensions = deviations.shape[1]
    precisions = np.repeat(shared[None, :, :], num_classes, axis=0)

    for index in range(num_classes):
        mask = labels == index
        if int(mask.sum()) < dimensions:
            continue
        precisions[index] = _regularised_precision(deviations[mask], shrinkage)

    return precisions


def _class_thresholds(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    precisions: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """O limiar de cada classe, no percentil pedido das distâncias dela mesma.

    Uma classe sem amostras herda a mediana das demais: zero viraria divisor
    zero no ``score``, e um limiar muito alto a transformaria num atrator que
    engole tudo o que o modelo não conhece.
    """
    thresholds = np.zeros(len(centroids), dtype=np.float64)

    for index in range(len(centroids)):
        mask = labels == index
        if not mask.any():
            continue
        own = _mahalanobis_to(embeddings[mask], centroids[index], precisions[index])
        thresholds[index] = float(np.percentile(own, percentile))

    empty = thresholds == 0.0
    if empty.all():
        thresholds[:] = 1.0
    elif empty.any():
        thresholds[empty] = float(np.median(thresholds[~empty]))

    return thresholds


def _mahalanobis_to(
    embeddings: np.ndarray, centroid: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    """Distância de cada amostra até **um** centro, sob a métrica daquela classe."""
    differences = embeddings - centroid
    squared = np.einsum("nd,de,ne->n", differences, precision, differences)
    return np.sqrt(np.maximum(squared, 0.0))


def _min_normalised_distance(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    precisions: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Menor distância até uma classe, medida em múltiplos do limiar dela."""
    differences = embeddings[:, None, :] - centroids[None, :, :]
    squared = np.einsum("ncd,cde,nce->nc", differences, precisions, differences)
    distances = np.sqrt(np.maximum(squared, 0.0))
    return (distances / thresholds[None, :]).min(axis=1)
