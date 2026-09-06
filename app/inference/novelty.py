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
    """Assinatura estatística das classes que o modelo conhece."""

    class_names: tuple[str, ...]
    centroids: np.ndarray
    """``(C, D)`` — o centro de cada classe no espaço de embeddings."""

    precision: np.ndarray
    """``(D, D)`` — inversa da covariância compartilhada entre as classes."""

    threshold: float
    """Distância a partir da qual a imagem é considerada novidade."""

    fitted_on: int
    """Quantas imagens foram usadas na calibração."""

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

        Usa uma covariância **compartilhada** entre as classes (em vez de uma por
        classe): com algumas centenas de imagens por classe e 128 dimensões, uma
        matriz por classe não teria amostras suficientes para ser estável.

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
        precision = _regularised_precision(deviations, shrinkage)

        distances = _min_mahalanobis(embeddings, centroids, precision)
        threshold = float(np.percentile(distances, percentile))

        return cls(
            class_names=tuple(class_names),
            centroids=centroids,
            precision=precision,
            threshold=threshold,
            fitted_on=len(embeddings),
        )

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Pontua imagens: ``> 1.0`` significa novidade.

        O valor é a distância até a classe conhecida mais próxima, dividida pelo
        limiar calibrado. Normalizar assim deixa o número comparável entre
        modelos diferentes — 1,4 quer dizer "40% além do limiar", seja no milho
        ou no café.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        distances = _min_mahalanobis(embeddings, self.centroids, self.precision)
        return distances / self.threshold

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            class_names=np.array(self.class_names, dtype=object),
            centroids=self.centroids,
            precision=self.precision,
            threshold=np.array(self.threshold),
            fitted_on=np.array(self.fitted_on),
        )

    @classmethod
    def load(cls, path: Path) -> NoveltyReference:
        """Lê uma referência gravada por :meth:`save`.

        Raises:
            ValueError: se o arquivo estiver corrompido ou incompleto.
        """
        try:
            with np.load(path, allow_pickle=True) as data:
                return cls(
                    class_names=tuple(str(name) for name in data["class_names"]),
                    centroids=data["centroids"],
                    precision=data["precision"],
                    threshold=float(data["threshold"]),
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


def _min_mahalanobis(
    embeddings: np.ndarray, centroids: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    """Distância de Mahalanobis até o centroide mais próximo, por amostra."""
    differences = embeddings[:, None, :] - centroids[None, :, :]
    squared = np.einsum("ncd,de,nce->nc", differences, precision, differences)
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))
