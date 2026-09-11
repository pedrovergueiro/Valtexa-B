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
    severity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Fração do quadro marcada como lesão pelo modelo de segmentação. "
            "Aproxima a severidade da folha quando a folha preenche o quadro. "
            "Nula quando o modelo não mede severidade ou os pesos não estão carregados."
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


class ReviewItem(BaseModel):
    """Uma imagem esperando o diagnóstico do agrônomo."""

    model_config = _ALLOW_MODEL_PREFIX

    id: str = Field(description="Identificador do item, usado nas demais rotas de revisão.")
    original_filename: str = Field(description="Nome do arquivo como chegou em /predict.")
    captured_at: str = Field(description="Quando entrou na fila, em UTC.")
    model_key: str = Field(description="Modelo que atendeu a requisição.")
    model_guess: str = Field(
        description="A classe que o modelo respondeu. É um palpite — foi por não confiar nele "
        "que a imagem veio parar aqui."
    )
    confidence: float = Field(description="Confiança do palpite, no intervalo [0, 1].")
    novelty_score: float | None = Field(
        default=None,
        description="Quanto a imagem destoa do que o modelo viu. Acima de 1,0 é novidade.",
    )
    probabilities: dict[str, float] = Field(
        default_factory=dict, description="Distribuição completa, para o agrônomo ver as dúvidas."
    )
    reviewed_label: str | None = Field(
        default=None,
        description="A classe que o agrônomo determinou. Nulo enquanto o item espera revisão.",
    )
    reviewed_by: str | None = Field(default=None, description="Quem diagnosticou.")
    reviewed_at: str | None = Field(default=None, description="Quando foi diagnosticado, em UTC.")


class ReviewStats(BaseModel):
    """Tamanho da fila — o orçamento de tempo do agrônomo."""

    enabled: bool = Field(description="Se REVIEW_DIR está configurado. Sem ele nada é arquivado.")
    pending: int = Field(description="Itens à espera de diagnóstico.")
    labeled: int = Field(description="Itens já diagnosticados, prontos para o próximo treino.")
    oldest_pending: str | None = Field(
        default=None, description="Data do item mais antigo ainda na fila."
    )


class ReviewVerdict(BaseModel):
    """O diagnóstico que o agrônomo devolve."""

    class_name: str = Field(
        description="A classe correta. Pode ser um nome novo, que ainda não existe no catálogo — "
        "é assim que uma doença nova entra no dataset.",
        examples=["Gray_Leaf_Spot", "Cigarrinha"],
    )
    reviewed_by: str | None = Field(
        default=None, description="Quem diagnosticou. Fica no metadado, para rastreabilidade."
    )


class ErrorResponse(BaseModel):
    """Corpo padrão das respostas de erro."""

    detail: str = Field(description="Mensagem legível descrevendo o problema.")
