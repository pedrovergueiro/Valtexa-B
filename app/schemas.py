"""Modelos Pydantic que descrevem o contrato HTTP da API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ``model_`` é um namespace protegido do Pydantic; os campos abaixo usam o
# prefixo porque "modelo" é o vocabulário do domínio.
_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


class ClassProbability(BaseModel):
    """Probabilidade atribuída a uma classe."""

    class_name: str = Field(description="Nome da classe.")
    probability: float = Field(ge=0.0, le=1.0, description="Probabilidade no intervalo [0, 1].")


class PredictionResponse(BaseModel):
    """Resultado da classificação de uma imagem."""

    model_config = _ALLOW_MODEL_PREFIX

    filename: str = Field(description="Nome do arquivo enviado.")
    model_key: str = Field(description="Modelo que respondeu, ex.: 'leaf-foliar'.")
    predicted_class: str = Field(description="Classe de maior probabilidade.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probabilidade da classe prevista.")
    abstained: bool = Field(
        default=False,
        description=(
            "Verdadeiro quando a confiança ficou abaixo de MIN_CONFIDENCE. O "
            "resultado é inconclusivo e não deve ser mostrado como diagnóstico."
        ),
    )
    novelty_score: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Quão diferente a imagem é de tudo que o modelo conhece. Acima de "
            "1.0 é novidade — pode ser doença fora do catálogo, deficiência, "
            "praga nova ou foto ruim. Nulo quando o modelo não tem referência "
            "de novidade calibrada."
        ),
    )
    needs_review: bool = Field(
        default=False,
        description=(
            "Verdadeiro quando a imagem deveria ir para um agrônomo. É o "
            "material que faz o dataset próprio crescer."
        ),
    )
    probabilities: list[ClassProbability] = Field(
        description="Probabilidade de cada classe suportada, na ordem do modelo."
    )
    inference_ms: float = Field(
        ge=0.0,
        description=(
            "Duração da passada pelo modelo, em milissegundos, sem contar a "
            "decodificação da imagem. Em lote, é o tempo total dividido pelo "
            "número de imagens, já que a passada é compartilhada."
        ),
    )


class BatchPredictionResponse(BaseModel):
    """Resultado da classificação de um lote de imagens."""

    model_config = _ALLOW_MODEL_PREFIX

    model_key: str = Field(description="Modelo que respondeu ao lote inteiro.")
    results: list[PredictionResponse] = Field(
        description="Um resultado por imagem, na ordem em que foram enviadas."
    )
    summary: dict[str, int] = Field(
        description="Contagem de previsões por classe dentro deste lote."
    )
    abstained_count: int = Field(
        ge=0, description="Quantas imagens ficaram abaixo do limiar de confiança."
    )
    needs_review_count: int = Field(
        default=0, ge=0, description="Quantas imagens foram marcadas para o agrônomo."
    )


class ModelInfo(BaseModel):
    """Descrição de um modelo do catálogo."""

    key: str = Field(description="Identificador usado no parâmetro 'model'.")
    title: str = Field(description="Nome legível do modelo.")
    platform: str = Field(description="Plataforma de captura: aerial, ground ou any.")
    summary: str = Field(description="O que este modelo responde.")
    classes: list[str] = Field(description="Classes na ordem em que o modelo as indexa.")
    img_size: int = Field(description="Lado da entrada quadrada, em pixels.")
    datasets: list[str] = Field(description="Datasets de origem. Detalhes em MODELS.md.")
    loaded: bool = Field(description="Se os pesos estão carregados e prontos para prever.")
    needs_field_data: bool = Field(
        description="Se depende de coleta própria por falta de dataset público."
    )


class HealthResponse(BaseModel):
    """Estado da API e dos modelos carregados."""

    model_config = _ALLOW_MODEL_PREFIX

    status: str = Field(description="'ok' enquanto o processo estiver de pé.")
    model_loaded: bool = Field(description="Se o modelo padrão está pronto.")
    default_model: str = Field(description="Modelo usado quando nenhum é informado.")
    loaded_models: list[str] = Field(description="Modelos prontos para prever agora.")


class ErrorResponse(BaseModel):
    """Corpo padrão das respostas de erro."""

    detail: str = Field(description="Mensagem legível descrevendo o problema.")
