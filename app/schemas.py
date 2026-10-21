"""Modelos Pydantic que descrevem o contrato HTTP da API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ``model_`` é um namespace protegido do Pydantic; os campos abaixo usam o
# prefixo porque "modelo" é o vocabulário do domínio.
_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())


class ClassProbability(BaseModel):
    """Probabilidade atribuída a uma classe."""

    class_name: str = Field(description="Nome da classe.")
    probability: float = Field(ge=0.0, le=1.0, description="Probabilidade no intervalo [0, 1].")


class CaptureQuality(BaseModel):
    """Se a foto está boa o bastante para ser diagnosticada.

    A régua é o dataset de treino do modelo que respondeu: uma foto é "borrada"
    quando é mais borrada que quase tudo que ele viu ao aprender. Ver
    ``app/inference/quality.py``.
    """

    ok: bool = Field(description="Falso quando há algum problema. A ação é refazer a captura.")
    issues: list[str] = Field(
        default_factory=list,
        description="Problemas encontrados: blurry, dark, overexposed, low_contrast.",
    )
    actions: list[str] = Field(
        default_factory=list, description="O que cada problema pede de quem está capturando."
    )
    sharpness: float = Field(description="Variância do laplaciano na entrada do modelo.")
    brightness: float = Field(description="Luminância média, em [0, 255].")
    contrast: float = Field(description="Desvio-padrão da luminância, em [0, 255].")


class FramingCheck(BaseModel):
    """Se a foto é o que o modelo sabe ler: folha em close, da cultura certa."""

    in_scope: bool = Field(
        description=(
            "Falso quando a foto não é folha em close — aérea, colmo, slide, solo. "
            "O diagnóstico, nesse caso, não deve ser mostrado: o modelo responde "
            "com confiança mesmo assim."
        )
    )
    score: float = Field(ge=0.0, le=1.0, description="Probabilidade de estar no escopo.")
    scene: str = Field(description="A cena mais parecida, ex.: folha_milho, aereo, colmo, texto.")
    crop: str | None = Field(
        default=None,
        description="Cultura da folha segundo o modelo de visão: corn, coffee, banana ou citrus.",
    )
    crop_matches: bool | None = Field(
        default=None,
        description="Se a cultura bate com a do modelo que respondeu. Nulo sem cultura.",
    )
    species_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Probabilidade de a folha ser da cultura do modelo, pela verificação de "
            "espécie (training.fit_heads --species). Abaixo do limiar dela, a foto sai "
            "do escopo com a cena folha_outra. Nula sem a verificação."
        ),
    )


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
    view_agreement: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Com TTA: fração das quatro orientações da imagem que votaram na classe "
            "final. Uma folha espelhada tem a mesma doença, então abaixo de 1,0 o "
            "modelo mudou de ideia só por girar a foto. Nulo sem TTA. Não é segunda "
            "opinião independente: é o mesmo peso olhando de novo."
        ),
    )
    quality: CaptureQuality | None = Field(
        default=None,
        description=(
            "Qualidade da captura. Nula quando o modelo não tem referência de "
            "qualidade calibrada (training.fit_quality)."
        ),
    )
    framing: FramingCheck | None = Field(
        default=None,
        description=(
            "Enquadramento, pelo modelo de visão-linguagem. Nulo nos modelos que não "
            "esperam folha em close, ou sem calibração (training.fit_framing)."
        ),
    )
    prediction_set: list[str] | None = Field(
        default=None,
        description=(
            "Classes que não dá para descartar, da mais para a menos provável. Contém "
            "a classe certa com a probabilidade set_coverage, garantida por classe "
            "(predição conformal). Uma classe: resposta firme. Várias: a dúvida real "
            "do modelo. Vazio: nenhuma classe alcança o nível pedido — inconclusivo. "
            "Nulo sem calibração (training.fit_conformal)."
        ),
    )
    set_coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="A cobertura garantida do prediction_set, ex.: 0.95.",
    )
    probabilities: list[ClassProbability] = Field(
        description="Probabilidade de cada classe suportada, na ordem do modelo."
    )
    inference_ms: float = Field(
        ge=0.0,
        description=(
            "Duração da passada pelo modelo, em milissegundos, sem contar a "
            "decodificação da imagem. Em lote, é o tempo total dividido pelo "
            "número de imagens, já que a passada é compartilhada. Com TTA, inclui "
            "as quatro orientações."
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
    quality_issue_count: int = Field(
        default=0,
        ge=0,
        description="Quantas imagens deveriam ser capturadas de novo. Zero sem referência.",
    )
    out_of_scope_count: int = Field(
        default=0,
        ge=0,
        description="Quantas imagens o portão de enquadramento recusou.",
    )
    ambiguous_count: int = Field(
        default=0,
        ge=0,
        description="Quantas imagens têm mais de uma classe no prediction_set.",
    )
    empty_set_count: int = Field(
        default=0,
        ge=0,
        description="Quantas imagens têm prediction_set vazio — nenhuma classe plausível.",
    )


class ExplanationResponse(BaseModel):
    """Onde o modelo olhou: o mapa de oclusão de uma classe numa imagem."""

    prediction: PredictionResponse = Field(description="A previsão, sem TTA, da imagem inteira.")
    target_class: str = Field(description="A classe explicada. Padrão: a prevista.")
    method: str = Field(
        default="occlusion",
        description="occlusion (uma célula por vez) ou rise (metade da imagem ao acaso).",
    )
    grid: int = Field(description="Células por lado da grade.")
    base_probability: float = Field(
        ge=0.0, le=1.0, description="Probabilidade da classe na imagem inteira."
    )
    drops: list[list[float]] = Field(
        description=(
            "Queda da probabilidade ao cobrir cada célula, linha a linha de cima "
            "para baixo. Positivo: a célula é evidência a favor da classe. "
            "Negativo: coberta, a classe fica mais provável."
        )
    )
    max_drop: float = Field(
        description=(
            "A maior contribuição de uma célula. Na oclusão, perto de zero também "
            "acontece com lesões espalhadas (cada uma é redundante); use rise para "
            "esse caso. Atalho de cor se mede melhor com o teste de cor média do "
            "training.audit_attention."
        )
    )
    peak_region: list[int] = Field(
        min_length=4,
        max_length=4,
        description="Célula mais decisiva, [x0, y0, x1, y1] em pixels da imagem enviada.",
    )
    passes: int = Field(description="Passadas pelo modelo que a explicação custou.")
    overlay_png: str | None = Field(
        default=None,
        description="A foto com a evidência pintada de vermelho, PNG em base64.",
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
    runtime: str | None = Field(
        default=None, description="Quem executa o modelo carregado: keras ou onnx."
    )
    safeguards: dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "Quais salvaguardas estão ativas neste modelo: novelty, quality, conformal, "
            "framing. Uma referência recusada no carregamento aparece aqui como falso — "
            "e não só num log de erro."
        ),
    )
    display_name: str = Field(default="", description="Nome curto no seletor de modelos.")
    crop: str | None = Field(default=None, description="Cultura: banana, citrus, corn, coffee.")
    variant: str = Field(default="padrão", description="Versão: campo, comercial, padrão...")
    license: str = Field(default="", description="Licença dos dados de treino.")
    field_accuracy: float | None = Field(
        default=None, description="Acurácia no teste de campo na escala do rover."
    )
    test_accuracy: float | None = Field(default=None, description="Acurácia no teste isolado.")
    recommended: bool = Field(default=False, description="Se é a versão sugerida da cultura.")


class ContinualOut(BaseModel):
    """O aprendizado contínuo da cabeça de um modelo (``training.adapt_campo``)."""

    key: str
    head_sha256: str | None = Field(
        description="SHA-256 da cabeça em produção (o que o perfil da borda do Rover confere)."
    )
    history: list[dict[str, Any]] = Field(
        description="Rodadas, promoções, reversões, monitoramentos e esquecimentos, em ordem."
    )
    memory: dict[str, Any] | None = Field(
        description="Memória de replay: exemplos, vagas por classe e contagem por origem."
    )
    regression_sets: list[dict[str, Any]] = Field(
        description="Conjuntos congelados do portão (gate=false: só mede)."
    )
    validation_sets: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Validação do que já se sabia, que escolhe a época (nunca o portão).",
    )


class ModelChoice(BaseModel):
    """Uma versão de modelo no seletor — como escolher entre modelos de um assistente."""

    key: str = Field(description="O que mandar em 'model=' para usar esta versão.")
    display_name: str
    variant: str
    summary: str
    license: str
    field_accuracy: float | None
    test_accuracy: float | None
    recommended: bool
    loaded: bool = Field(description="Se está pronta para prever agora.")


class CropChoices(BaseModel):
    """As versões de uma cultura, a recomendada primeiro."""

    crop: str
    rover_default: str = Field(
        description="A versão que o rover usa quando manda só 'crop=' (ROVER_MODELS)."
    )
    options: list[ModelChoice]


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
    context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "De onde a imagem veio, quando não foi um upload direto. No rover: talhão, rua, "
            "lado, posição, câmera, altura, zona da planta e cena que o portão viu."
        ),
    )
    priority: int | None = Field(
        default=None, description="Com order=valor: posição na fila (1 = rotular primeiro)."
    )
    priority_reasons: list[str] = Field(
        default_factory=list,
        description="Com order=valor: por que este item ensina, ex.: 'incerteza alta'.",
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


# ---------------------------------------------------------------------------
# Leitura de terreno
# ---------------------------------------------------------------------------


class RefineRequest(BaseModel):
    """O contorno que a pessoa desenhou, em graus, e o quanto a IA pode mexer."""

    polygon: list[list[float]] = Field(
        min_length=3,
        description="Vértices [lon, lat] do traço a mão. Pode estar bem torto.",
        examples=[[[-45.866, -21.902], [-45.859, -21.901], [-45.860, -21.906]]],
    )
    tolerance: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        description=(
            "Quanto o contorno pode se afastar do traço, como fração do raio. "
            "Baixo trava perto do que foi desenhado; alto corrige mão trêmula."
        ),
    )
    vertices: int = Field(default=48, ge=8, le=360)
    tiles: str | None = Field(
        default=None,
        description=(
            "Molde XYZ da imagem. Vazio usa BASEMAP_TILES do serviço. Ver a nota "
            "de licenciamento em app/basemap.py — não há padrão silencioso."
        ),
    )


class RowsInfo(BaseModel):
    """O que a imagem disse sobre as linhas de plantio."""

    detected: bool
    azimuth_deg: float | None = None
    spacing_m: float | None = None
    strength: float = Field(description="Quanto o padrão se destaca do fundo do espectro.")
    description: str


class CoverInfo(BaseModel):
    """Repartição do terreno por cobertura, em blocos."""

    fractions: dict[str, float]
    obstacle_fraction: float = Field(description="Mata mais o que a regra não soube separar.")
    heuristic: bool = Field(
        description="Sempre verdadeiro: é regra de cor e textura, não modelo treinado."
    )


class ObjectInfo(BaseModel):
    """Uma coisa localizada dentro do talhão."""

    kind: str = Field(
        description="mata, carreador, solo_exposto ou desconhecido. Palpite, não medição."
    )
    center: list[float] = Field(
        description="Centro [lon, lat] do agrupamento.", min_length=2, max_length=2
    )
    outline: list[list[float]] = Field(
        description=(
            "Retângulo [lon, lat] que cerca o objeto. Grosseiro de propósito: a "
            "grade é de blocos, e um contorno fino sugeriria precisão que não existe."
        )
    )
    area_ha: float
    traversable: bool = Field(
        description="Se o rover passa por cima (carreador, solo) ou tem que contornar."
    )
    confidence: float = Field(description="Fração dos blocos que concordaram com a classe.")
    blocks: int


class TerrainInfo(BaseModel):
    """O talhão repartido: quanto dá para operar e o que atrapalha.

    ``resolution_ok`` falso zera tudo o mais. É deliberado: numa imagem em que a
    rua não se resolve, **nada** é lavoura para o classificador, e o resultado
    seria "75% de obstáculo" num cafezal inteiro — um número que encolheria a
    área operável pela metade sem ninguém desconfiar.
    """

    resolution_ok: bool
    description: str
    crop_fraction: float
    obstacle_fraction: float
    path_fraction: float
    row_spacing_px: float | None = None
    cover: CoverInfo | None = None
    objects: list[ObjectInfo] = Field(default_factory=list)
    heuristic: bool = True


class RefineResponse(BaseModel):
    """O contorno encaixado e o quanto se pode confiar nele."""

    polygon: list[list[float]] = Field(description="Vértices [lon, lat] encaixados.")
    confidence: float = Field(
        description="Fração das direções em que a borda apareceu com transição nítida."
    )
    hit_limit: float = Field(
        description=(
            "Fração das direções que pararam na tolerância em vez de numa borda. "
            "Alto quer dizer que o traço apertou a lavoura ou que não há borda ali."
        )
    )
    area_change: float = Field(description="Variação de área em relação ao desenho.")
    meters_per_pixel: float = Field(description="Resolução da imagem usada.")
    zoom: int
    rows: RowsInfo | None = Field(
        default=None, description="Ruas de plantio lidas dentro do contorno, quando detectáveis."
    )
    terrain: TerrainInfo | None = Field(
        default=None,
        description="Cobertura e objetos localizados dentro do contorno encaixado.",
    )
    notes: list[str] = Field(default_factory=list, description="Ressalvas sobre este resultado.")


# ---------------------------------------------------------------------------
# Rover: quadro inteiro, diagnóstico por zona da planta
# ---------------------------------------------------------------------------


class RoverTile(BaseModel):
    """Um recorte do quadro do rover, e o que o modelo achou nele."""

    box: list[int] = Field(min_length=4, max_length=4, description="[x0, y0, x1, y1] no quadro.")
    height_mm: float = Field(description="Altura estimada do centro do recorte em relação ao chão.")
    zones: list[str] = Field(
        default_factory=list, description="Zonas da planta, ex.: pseudocaule, folhas_baixas."
    )
    predicted_class: str
    confidence: float
    prediction_set: list[str] | None = None
    needs_review: bool = False
    in_scope: bool | None = Field(
        default=None,
        description=(
            "Falso quando o portão de enquadramento diz que o recorte não é folha em "
            "close — pseudocaule, tronco, fruto. A classe prevista não vale e o recorte "
            "não conta como achado. Nulo sem o portão."
        ),
    )
    scene: str | None = Field(
        default=None, description="Cena segundo o portão, ex.: folha_banana, pseudocaule, tronco."
    )


class RoverFrameResponse(BaseModel):
    """O quadro de uma câmera lateral, diagnosticado recorte a recorte."""

    model_config = _ALLOW_MODEL_PREFIX

    model_key: str
    crop: str | None = Field(description="Cultura do modelo, que define as zonas.")
    frame_size: list[int] = Field(description="[largura, altura] do quadro recebido.")
    tile_px: int = Field(
        description="Lado dos recortes, em pixels — o tamanho físico na distância."
    )
    tiles_analyzed: int
    class_counts: dict[str, int] = Field(
        description="Recortes por classe prevista — só os que o portão aceitou."
    )
    zone_summary: dict[str, dict[str, int]] = Field(
        description=(
            "Por zona: recortes vistos, quantos têm achado e quantos o portão recusou "
            "(fora_de_escopo)."
        )
    )
    plants: list[str] | None = Field(
        default=None,
        description="Com field_id: as plantas do talhão que este quadro alimentou (ver /fields).",
    )
    field_prevalence: dict[str, float] | None = Field(
        default=None,
        description=(
            "Com field_id: a prevalência de cada classe no talhão até aqui, estimada sem "
            "rótulo (EM). Nula sem o prior de treino do modelo."
        ),
    )
    sigatoka: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Banana: severidade da Sigatoka no quadro (escala de Stover-Gauhl, DSI com "
            "intervalo). Escala ainda não validada por agrônomo — ver 'validated'."
        ),
    )
    out_of_scope: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description=(
            "Por zona, os recortes recusados pelo portão, contados por cena. É a parte da "
            "planta que o modelo foliar não sabe ler — no pseudocaule, quase tudo."
        ),
    )
    queued_for_review: int = Field(
        default=0,
        description="Recortes deste quadro arquivados na fila de revisão (REVIEW_DIR).",
    )
    tiles: list[RoverTile] = Field(
        description="Os recortes com achado (ou todos, com only_findings=false)."
    )
    max_forward_batch: int = Field(description="Imagens por passada usadas na análise.")
