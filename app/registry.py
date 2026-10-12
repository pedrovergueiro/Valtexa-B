"""Catálogo dos modelos do produto.

Cada modelo responde por uma combinação de cultura e enquadramento — não por
plataforma de captura sozinha. ``leaf-foliar`` e ``coffee-foliar`` servem tanto
o drone em voo baixo quanto o rover, desde que a folha preencha o quadro;
``rover-understory`` e ``aerial-canopy`` são específicos de plataforma porque
respondem a uma pergunta que só faz sentido naquele ângulo. Ver ``MODELS.md``
para o detalhamento de cada um.

Um único ``softmax`` universal não serviria: ele seria obrigado a atribuir
probabilidade a "podridão de colmo" numa foto aérea de dossel, onde isso é
fisicamente invisível — e a doenças de café numa foto de milho.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.config import settings


class Platform(str, Enum):
    """Plataforma de captura das imagens."""

    AERIAL = "aerial"
    """Drone sobrevoando o dossel."""

    GROUND = "ground"
    """Rover terrestre entre as linhas, sob o dossel."""

    ANY = "any"
    """Independe da plataforma — vale sempre que a folha preenche o quadro."""


@dataclass(frozen=True)
class ModelSpec:
    """Definição estática de um modelo do catálogo."""

    key: str
    """Identificador usado na API e na linha de comando do treino."""

    platform: Platform
    title: str
    summary: str
    class_names: tuple[str, ...]
    weights_name: str
    """Nome do arquivo de pesos dentro de ``models/``."""

    img_size: int = 224
    datasets: tuple[str, ...] = ()
    """Datasets de origem, na ordem de prioridade. Detalhes em ``MODELS.md``."""

    needs_field_data: bool = False
    """Se ``True``, não existe dataset público suficiente e o modelo depende de
    coleta própria em campo."""

    dataset_key: str | None = None
    """Dataset de outro modelo do catálogo, quando este é outra arquitetura sobre
    os mesmos dados. ``None``: o dataset é ``data/<key>/``."""

    measures_severity: bool = False
    """Se a resposta traz a severidade do :mod:`app.severity`.

    Só para folha de milho em close: é o que as máscaras de treino cobrem. Medir
    lesão de café ou num tile aéreo com essa rede seria inventar número."""

    crop: str | None = None
    """Cultura que o modelo diagnostica: ``corn`` ou ``coffee``.

    É o que o portão de enquadramento confere: uma folha de café mandada ao
    modelo de milho recebe uma das quatro classes de milho, e nada na resposta
    denunciaria o engano."""

    expects_leaf_closeup: bool = False
    """Se o modelo só sabe ler folha preenchendo o quadro.

    É a premissa dos foliares, e é o que o :mod:`app.framing` verifica. Nos de
    rover e drone o enquadramento certo é outro, e a pergunta não se aplica."""

    arch: str | None = None
    """Base convolucional. ``None``: a padrão do treino.

    Faz parte da identidade do modelo: treinar o ``leaf-foliar-b`` com a base do
    ``leaf-foliar`` produziria uma cópia, não uma segunda opinião."""

    notes: str = ""

    display_name: str = ""
    """Nome curto no seletor de modelos, ex.: ``Banana · campo``. Vazio: o título."""

    variant: str = "padrão"
    """Qual versão da mesma cultura: ``campo``, ``comercial``, ``segunda opinião``.

    Várias versões da mesma cultura convivem no catálogo, como os modelos de um
    assistente de IA: quem chama escolhe, pelo ``model=``, a que quer usar — no
    rover ou num upload manual."""

    license: str = ""
    """Licença dos dados de treino. É o que decide se a versão pode ir para uso
    comercial: uma versão treinada com dado CC BY-NC não pode."""

    field_accuracy: float | None = None
    """Acurácia no teste de campo na escala do rover (ver MODELS.md), quando medida.

    É o número a comparar entre versões: o teste isolado sai das mesmas fontes do
    treino e engana (0,985 contra 0,428 no primeiro modelo de banana)."""

    test_accuracy: float | None = None
    """Acurácia no teste isolado, com TTA."""

    recommended: bool = False
    """A versão sugerida da cultura — a que o rover usa quando só a cultura é dada."""

    @property
    def label(self) -> str:
        return self.display_name or self.title

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def weights_path(self) -> Path:
        """Caminho dos pesos.

        O modelo padrão respeita ``MODEL_PATH`` para manter compatibilidade com
        deploys e com o ``Dockerfile`` existentes.
        """
        if self.key == settings.default_model and settings.model_path is not None:
            return settings.model_path
        return settings.models_dir / self.weights_name

    @property
    def labels_path(self) -> Path:
        """Caminho do ``labels.json``, derivado do arquivo de pesos."""
        if self.key == settings.default_model and settings.labels_path is not None:
            return settings.labels_path
        return self.weights_path.with_suffix(".labels.json")

    @property
    def novelty_path(self) -> Path:
        """Referência de novidade, gravada por ``training.fit_novelty``."""
        return self.weights_path.with_suffix(".novelty.npz")

    @property
    def quality_path(self) -> Path:
        """Referência de qualidade da captura, gravada por ``training.fit_quality``."""
        return self.weights_path.with_suffix(".quality.json")

    @property
    def conformal_path(self) -> Path:
        """Limiares da predição conformal, gravados por ``training.fit_conformal``."""
        return self.weights_path.with_suffix(".conformal.json")

    @property
    def head_path(self) -> Path:
        """Cabeça de diagnóstico sobre o embedding do portão (``training.fit_heads``)."""
        return self.weights_path.with_suffix(".head.npz")

    @property
    def species_path(self) -> Path:
        """Verificação de espécie sobre o embedding do portão (``training.fit_heads --species``)."""
        return self.weights_path.with_suffix(".species.npz")

    @property
    def fusion_path(self) -> Path:
        """Calibração da fusão por planta deste modelo (``training.fit_fusion``)."""
        return self.weights_path.with_suffix(".fusion.json")

    @property
    def edge_fusion_path(self) -> Path:
        """Calibração da fusão para recortes classificados **no rover** (``training.fit_edge
        fusao``): sem TTA e sem o portão de enquadramento, que só existem na nuvem."""
        return self.weights_path.with_suffix(".edge-fusion.json")

    @property
    def data_dir(self) -> Path:
        """Diretório do dataset organizado por classe, usado pelo treino."""
        return settings.data_dir / (self.dataset_key or self.key)


LEAF_FOLIAR = ModelSpec(
    key="leaf-foliar",
    platform=Platform.ANY,
    title="Diagnóstico foliar em close",
    summary=(
        "Classifica doenças foliares quando a folha preenche o quadro. Serve às "
        "duas plataformas: ao drone em voo baixo e ao rover no baixo dossel."
    ),
    class_names=("Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"),
    display_name="Milho · padrão",
    test_accuracy=0.9555,
    recommended=True,
    weights_name="corn_disease_model.keras",
    measures_severity=True,
    crop="corn",
    expects_leaf_closeup=True,
    datasets=(
        "corn-maize-leaf-disease",
        "plantvillage-corn",
        "plantdoc-corn",
        "cd-and-s",
        "plantseg-corn",
    ),
    notes=(
        "É o modelo que sustenta o piloto comercial: dataset público maduro, "
        "treina em minutos e é o único pronto para rodar hoje."
    ),
)

LEAF_FOLIAR_B = ModelSpec(
    key="leaf-foliar-b",
    platform=Platform.ANY,
    title="Diagnóstico foliar em close — segunda arquitetura",
    summary=(
        "Mesmas classes e mesmo dataset do leaf-foliar, com outra base "
        "convolucional (EfficientNetV2B0). Existe para ser a segunda opinião da "
        "dupla validação da plataforma."
    ),
    class_names=LEAF_FOLIAR.class_names,
    weights_name="leaf_foliar_b.keras",
    display_name="Milho · segunda opinião",
    variant="segunda opinião",
    test_accuracy=0.9601,
    measures_severity=True,
    crop="corn",
    expects_leaf_closeup=True,
    datasets=LEAF_FOLIAR.datasets,
    dataset_key=LEAF_FOLIAR.key,
    arch="efficientnetv2b0",
    notes=(
        "Arquitetura diferente não garante independência: os dois aprendem do "
        "mesmo dado e herdam os mesmos rótulos contraditórios. A independência é "
        "medida (ver MODELS.md), não presumida."
    ),
)

COFFEE_FOLIAR = ModelSpec(
    key="coffee-foliar",
    platform=Platform.ANY,
    title="Diagnóstico foliar de café",
    summary=(
        "Classifica doenças e pragas foliares do cafeeiro a partir de folha em "
        "close, para a cultura mais estratégica da região de atuação da empresa. "
        "Com a cabeça sobre o embedding do portão (coffee_foliar.head.npz, "
        "01/10/2026), acerta 96% no teste isolado e 80% em 542 fotos de folha "
        "inteira de outra fazenda brasileira — a CNN sozinha, 6,5%. A folha na "
        "planta, como o rover a vê, ainda não foi medida."
    ),
    class_names=("Healthy", "Rust", "Cercospora_Leaf_Spot", "Phoma_Leaf_Spot", "Leaf_Miner_Damage"),
    display_name="Café · padrão",
    test_accuracy=0.964,
    recommended=True,
    weights_name="coffee_foliar.keras",
    crop="coffee",
    expects_leaf_closeup=True,
    datasets=("jmuben-coffee", "rocole", "plantseg-coffee", "bracol (só a cabeça)"),
    notes=(
        "Treinável hoje com dado público — mesma ressalva do leaf-foliar: o "
        "JMuBEN foi capturado em Kirinyaga, Quênia, em baixa resolução "
        "(128x128). O diferencial real nasce de coleta própria no Sul de Minas."
    ),
)

ROVER_UNDERSTORY = ModelSpec(
    key="rover-understory",
    platform=Platform.GROUND,
    title="Diagnóstico sob o dossel (rover)",
    summary=(
        "Visão lateral do baixo dossel, colmo e base da planta. Cobre as doenças "
        "foliares semanas antes do drone e as patologias que só existem embaixo."
    ),
    class_names=(
        "Healthy",
        "Blight",
        "Common_Rust",
        "Gray_Leaf_Spot",
        "Stalk_Rot",
        "Stalk_Borer_Damage",
        "Nutrient_Deficiency",
        "Corn_Stunt",
    ),
    weights_name="rover_understory.keras",
    datasets=("plantdoc-corn", "cd-and-s", "nlb-cornell", "coleta-propria"),
    needs_field_data=True,
    notes=(
        "As quatro primeiras classes têm dataset público; Stalk_Rot, "
        "Stalk_Borer_Damage, Nutrient_Deficiency e Corn_Stunt exigem coleta "
        "própria. Comece pelas foliares e vá abrindo as demais."
    ),
)

AERIAL_CANOPY = ModelSpec(
    key="aerial-canopy",
    platform=Platform.AERIAL,
    title="Triagem de dossel (drone)",
    summary=(
        "Tiles de imagem nadir. Responde 'onde há algo errado', não 'qual "
        "doença' — é a camada de triagem que direciona o rover."
    ),
    class_names=("Healthy_Canopy", "Suspect"),
    weights_name="aerial_canopy.keras",
    datasets=("disease-nlb-drone", "nlb-cornell", "pdt-uav"),
    needs_field_data=True,
    notes=(
        "Binário de propósito: a 20 m de altura uma lesão tem poucos pixels e o "
        "quadro cobre dezenas de plantas. Otimize recall, não acurácia. "
        "`disease-nlb-drone` é o subconjunto só-UAV do `nlb-cornell` — baixe "
        "apenas um dos dois."
    ),
)

CITRUS_FOLIAR = ModelSpec(
    key="citrus-foliar",
    platform=Platform.ANY,
    title="Diagnóstico foliar de citros (laranja e limão)",
    summary=(
        "Folha de citros preenchendo o quadro — um recorte do quadro lateral do "
        "rover (ver app/rover.py) ou uma foto de mão. Laranja e limão juntos: as "
        "doenças foliares que os datasets públicos cobrem são as mesmas nos dois."
    ),
    class_names=(
        "Algal_Leaf_Spot",
        "Anthracnose",
        "Black_Spot",
        "Canker",
        "Curl_Leaf",
        "Greening",
        "Healthy",
        "Leafminer",
        "Melanose",
        "Mites",
        "Scab",
        "Sooty_Mold",
    ),
    weights_name="citrus_foliar.keras",
    display_name="Citros · campo",
    variant="campo",
    license="CC BY 4.0 + PlantSeg CC BY-NC 4.0 — não comercial",
    field_accuracy=0.996,
    test_accuracy=0.931,
    recommended=True,
    crop="citrus",
    expects_leaf_closeup=True,
    datasets=("lemon-large-bd", "bdlemonleaf", "sweet-orange-bd", "plantseg-citrus (só teste)"),
    notes=(
        "Os datasets de treino fotografam a folha destacada sobre papel; o rover a "
        "vê na árvore. O número que vale é o do teste de campo (PlantSeg), não o do "
        "teste isolado. Com a cabeça sobre o embedding do portão (01/10/2026), lê a "
        "folha igual em qualquer orientação sem TTA, e a verificação de espécie "
        "recusa folha de outra planta na foto avulsa. Nenhum dos datasets é "
        "brasileiro: leprose, CVC e gomose não estão no catálogo e caem na fila de "
        "revisão."
    ),
)

BANANA_FOLIAR = ModelSpec(
    key="banana-foliar",
    platform=Platform.ANY,
    title="Diagnóstico de folha de bananeira (folhas baixas)",
    summary=(
        "Folha de bananeira no quadro — o andar baixo do rover, onde ficam as "
        "folhas velhas: é nelas que a Sigatoka é mais severa e que o mal-do-Panamá "
        "amarela primeiro."
    ),
    class_names=("Cordana", "Fusarium_Wilt", "Healthy", "Pestalotiopsis", "Sigatoka"),
    weights_name="banana_foliar.keras",
    display_name="Banana · campo",
    variant="campo",
    license="CC BY 4.0 + PlantSeg CC BY-NC 4.0 — não comercial",
    field_accuracy=0.829,
    test_accuracy=0.977,
    recommended=True,
    crop="banana",
    expects_leaf_closeup=True,
    datasets=(
        "banana-tz-lacuna",
        "bananalsd",
        "banana-recognition-bd",
        "plantseg-banana (recortes de lesão; split oficial — treino no treino, teste no teste)",
    ),
    notes=(
        "O pseudocaule não está aqui: não há dataset público com sintoma externo "
        "dele — o 'caule' do dataset da Tanzânia é corte transversal, que o rover "
        "não vê. Recortes da zona do pseudocaule vão para revisão, e é a coleta do "
        "rover que vai formar esse dataset. Sigatoka-negra e amarela ficam juntas: "
        "nas imagens públicas os rótulos não as separam com segurança. Os recortes "
        "do PlantSeg são CC BY-NC 4.0: revisar a licença antes de uso comercial."
    ),
)

CITRUS_FOLIAR_CC = ModelSpec(
    key="citrus-foliar-cc",
    platform=Platform.ANY,
    title="Diagnóstico foliar de citros — versão comercial",
    summary=(
        "As mesmas 12 classes do citrus-foliar, treinado só com os datasets CC BY "
        "(folhas sobre papel). Pode ir para uso comercial; erra mais no campo."
    ),
    class_names=CITRUS_FOLIAR.class_names,
    weights_name="citrus_foliar_cc.keras",
    display_name="Citros · comercial",
    variant="comercial",
    license="CC BY 4.0",
    field_accuracy=0.798,
    test_accuracy=0.919,
    crop="citrus",
    expects_leaf_closeup=True,
    datasets=tuple(d for d in CITRUS_FOLIAR.datasets if not d.startswith("plantseg")),
    dataset_key=CITRUS_FOLIAR.key,
    notes=(
        "Sem a troca de fundo: medida, ela derrubou este modelo no campo de 0,695 "
        "para 0,382. A cabeça sobre o embedding do portão (só dado CC BY) subiu o "
        "campo para 0,798. Sem verificação de espécie: com dado CC BY só há citros "
        "sobre papel, e a cabeça treinada assim recusa 96 a 100% do citros de campo. "
        "Serve até a coleta própria do rover substituir o PlantSeg."
    ),
)

BANANA_FOLIAR_CC = ModelSpec(
    key="banana-foliar-cc",
    platform=Platform.ANY,
    title="Diagnóstico de folha de bananeira — versão comercial",
    summary=(
        "As mesmas 5 classes do banana-foliar, treinado só com os datasets CC BY. "
        "Pode ir para uso comercial; erra mais no campo, sobretudo em Cordana."
    ),
    class_names=BANANA_FOLIAR.class_names,
    weights_name="banana_foliar_cc.keras",
    display_name="Banana · comercial",
    variant="comercial",
    license="CC BY 4.0",
    field_accuracy=0.592,
    test_accuracy=0.985,
    crop="banana",
    expects_leaf_closeup=True,
    datasets=tuple(d for d in BANANA_FOLIAR.datasets if not d.startswith("plantseg")),
    dataset_key=BANANA_FOLIAR.key,
    notes="Serve até a coleta própria do rover substituir os recortes do PlantSeg.",
)

MODEL_SPECS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        LEAF_FOLIAR,
        LEAF_FOLIAR_B,
        COFFEE_FOLIAR,
        ROVER_UNDERSTORY,
        AERIAL_CANOPY,
        CITRUS_FOLIAR,
        CITRUS_FOLIAR_CC,
        BANANA_FOLIAR,
        BANANA_FOLIAR_CC,
    )
}

DEFAULT_PLATFORM_MODEL: dict[Platform, str] = {
    Platform.GROUND: ROVER_UNDERSTORY.key,
    Platform.AERIAL: AERIAL_CANOPY.key,
    Platform.ANY: LEAF_FOLIAR.key,
}
"""Modelo escolhido quando o cliente informa só a plataforma."""


def normalize_class_name(name: str) -> str:
    """Reduz um nome de pasta a uma chave comparável.

    Minúsculas, sem pontuação e com espaços colapsados, para que
    ``Corn_(maize)___Common_rust_`` e ``Corn Common Rust`` caiam na mesma chave.
    """
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


CANONICAL_SELF_ALIASES: dict[str, str] = {
    normalize_class_name(name): name
    for spec in (
        LEAF_FOLIAR,
        COFFEE_FOLIAR,
        ROVER_UNDERSTORY,
        AERIAL_CANOPY,
        CITRUS_FOLIAR,
        BANANA_FOLIAR,
    )
    for name in spec.class_names
}
"""Toda classe do catálogo mapeia para si mesma.

Garante que uma pasta nomeada exatamente como a classe sempre seja reconhecida —
é o que permite ao agrônomo simplesmente arrastar as imagens revisadas para
``review/labeled/<Classe>/`` e o ``prepare_dataset`` entender, sem código novo.
"""


SOURCE_CLASS_ALIASES: dict[str, str] = {
    **CANONICAL_SELF_ALIASES,
    # Corn or Maize Leaf Disease Dataset (Kaggle) — já usa os nomes canônicos.
    "blight": "Blight",
    "common rust": "Common_Rust",
    "gray leaf spot": "Gray_Leaf_Spot",
    "grey leaf spot": "Gray_Leaf_Spot",
    "healthy": "Healthy",
    # PlantVillage — "Corn_(maize)___Northern_Leaf_Blight" e afins.
    "corn maize northern leaf blight": "Blight",
    "corn maize common rust": "Common_Rust",
    "corn maize cercospora leaf spot gray leaf spot": "Gray_Leaf_Spot",
    "corn maize healthy": "Healthy",
    "northern leaf blight": "Blight",
    "cercospora leaf spot gray leaf spot": "Gray_Leaf_Spot",
    # PlantDoc — nomes em linguagem natural.
    "corn leaf blight": "Blight",
    "corn rust leaf": "Common_Rust",
    "corn gray leaf spot": "Gray_Leaf_Spot",
    # CD&S (Purdue).
    "nlb": "Blight",
    "gls": "Gray_Leaf_Spot",
    "nls": "Blight",
    # JMuBEN / JMuBEN2 (café) — nomes de pasta confirmados por inspeção direta
    # do download (kaggle:noamaanabdulazeem/jmuben-coffee-dataset). "Cerscospora"
    # é como o dataset de origem grafa a pasta — não é erro de digitação nosso.
    "cerscospora": "Cercospora_Leaf_Spot",
    "leaf rust": "Rust",
    "miner": "Leaf_Miner_Damage",
    "phoma": "Phoma_Leaf_Spot",
    # RoCoLe (kaggle:nirmalsankalana/rocole-...) — pastas "coffee___<classe>".
    # "coffee___red_spider_mite" fica de fora de propósito: é uma praga que não
    # está no catálogo do coffee-foliar, então cai em "ignoradas" corretamente.
    "coffee healthy": "Healthy",
    "coffee rust": "Rust",
    # Grafias genéricas — cobrem BRACOL (nomes corretos, sem o erro do JMuBEN)
    # e qualquer outra origem que já venha com o nome da doença sozinho.
    "rust": "Rust",
    "cercospora": "Cercospora_Leaf_Spot",
    # PlantVillage re-empacotado com a cultura no fim do nome, em vez do início
    # (kaggle:rashidthihan/plant-disease-dataset). A ordem das palavras muda, e
    # a normalização não reordena, então as duas grafias precisam constar.
    "northern leaf blight corn maize": "Blight",
    "common rust corn maize": "Common_Rust",
    "cercospora leaf spot gray leaf spot corn maize": "Gray_Leaf_Spot",
    "healthy corn maize": "Healthy",
    # Datasets indonésios de milho (kaggle:ndisan/*). "daun" é folha; "karat",
    # ferrugem; "hawar", queima/requeima; "bercak", mancha; "sehat", saudável.
    "karat daun": "Common_Rust",
    "hawar daun": "Blight",
    "bercak daun": "Gray_Leaf_Spot",
    "daun sehat": "Healthy",
    "sehat": "Healthy",
    # Colmo, para o rover-understory (kaggle:ndisan/corn-stalks). "busuk" é
    # podridão; "pelepah", bainha; "arang", carvão. São duas etiologias
    # distintas — podridão de bainha e podridão-de-carvão (Macrophomina) — e o
    # catálogo tem uma classe só de colmo, então as duas caem em Stalk_Rot.
    # Se um dia elas precisarem ser separadas, é aqui que a decisão muda.
    "busuk pelepah": "Stalk_Rot",
    "busuk arang": "Stalk_Rot",
    # Citros — Large-Scale Lemon Leaf (Mendeley 8d9fv6kpt3) e BDLemonLeaf
    # (643f5bbc2t, só a pasta "Raw Image": a "Annotated Image" tem as mesmas
    # fotos com as caixas desenhadas por cima). Ácaro-da-índia e ácaro-rajado
    # ficam juntos em Mites: o manejo é o mesmo e separados não passam de 800.
    # Ficam de fora, reportadas: "Dry Leaf", "Yellow Spot", "Citrus Pest",
    # "Bacterial Blight" e "Swallowtail Larval Herbivory (Deficiency)" — rótulos
    # vagos ou que misturam praga com deficiência.
    "algal leaf spot": "Algal_Leaf_Spot",
    "anthracnose": "Anthracnose",
    "black spot": "Black_Spot",
    "citrus canker": "Canker",
    "curl leaf": "Curl_Leaf",
    "greening": "Greening",
    "citrus greening": "Greening",
    # "Dragão amarelo" é a tradução literal de huanglongbing — o próprio greening.
    "yellow dragon": "Greening",
    "citrus leafminer": "Leafminer",
    "melanose": "Melanose",
    "citrus hindu mite": "Mites",
    "spider mites": "Mites",
    "citrus scab": "Scab",
    "lemon sooty mold": "Sooty_Mold",
    "healthy leaf": "Healthy",
    # Banana — Lacuna/Tanzânia (Dataverse 10.7910/DVN/LQUWXW), BananaLSD e Banana
    # Disease Recognition (Kaggle, só as pastas originais). Sigatoka-negra e
    # amarela juntas em Sigatoka; mal-do-Panamá é Fusarium_Wilt.
    "black sigatoka 1": "Sigatoka",
    "black sigatoka 2": "Sigatoka",
    "fusarium wilt 1": "Fusarium_Wilt",
    "fusarium wilt 2": "Fusarium_Wilt",
    "healthy 1": "Healthy",
    "healthy 2": "Healthy",
    "sigatoka": "Sigatoka",
    "cordana": "Cordana",
    "pestalotiopsis": "Pestalotiopsis",
    "banana black sigatoka disease": "Sigatoka",
    "banana yellow sigatoka disease": "Sigatoka",
    "banana healthy leaf": "Healthy",
    "banana panama disease": "Fusarium_Wilt",
}
"""Nomes de pasta das origens públicas mapeados para as classes canônicas.

As chaves estão normalizadas por :func:`normalize_class_name`. Pastas que não
aparecem aqui são reportadas pelo ``prepare_dataset`` em vez de adivinhadas —
um palpite errado aqui contamina o dataset inteiro em silêncio.
"""


def get_spec(key: str) -> ModelSpec:
    """Busca um modelo pelo identificador.

    Raises:
        KeyError: se o identificador não existir no catálogo.
    """
    try:
        return MODEL_SPECS[key]
    except KeyError:
        disponiveis = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"Modelo desconhecido: {key!r}. Disponíveis: {disponiveis}.") from None


def specs_by_crop() -> dict[str, list[ModelSpec]]:
    """As versões de cada cultura, a recomendada primeiro e depois por campo.

    É a lista que um seletor de modelos mostra — como escolher entre os modelos
    de um assistente de IA: cada versão tem nome, o número de campo e a licença.
    """
    grouped: dict[str, list[ModelSpec]] = {}
    for spec in MODEL_SPECS.values():
        if spec.crop:
            grouped.setdefault(spec.crop, []).append(spec)
    for options in grouped.values():
        options.sort(key=lambda s: (not s.recommended, -(s.field_accuracy or 0.0), s.key))
    return grouped


def default_for_crop(crop: str) -> ModelSpec:
    """A versão usada quando só a cultura é informada.

    ``ROVER_MODELS`` (JSON, ex.: ``{"banana": "banana-foliar-cc"}``) troca a escolha sem mexer
    no código — é assim que o rover passa a usar a versão comercial.

    Raises:
        KeyError: se a cultura não tem modelo, ou se ``ROVER_MODELS`` aponta para
            um modelo de outra cultura.
    """
    chosen = settings.rover_models.get(crop)
    if chosen is not None:
        spec = get_spec(chosen)
        if spec.crop != crop:
            raise KeyError(
                f"ROVER_MODELS aponta {chosen!r} para {crop!r}, mas ele é de {spec.crop!r}."
            )
        return spec
    options = specs_by_crop().get(crop)
    if not options:
        culturas = ", ".join(sorted(specs_by_crop()))
        raise KeyError(f"Nenhum modelo para a cultura {crop!r}. Culturas: {culturas}.")
    return options[0]


def spec_for_platform(platform: Platform) -> ModelSpec:
    """Modelo padrão de uma plataforma de captura."""
    return MODEL_SPECS[DEFAULT_PLATFORM_MODEL[platform]]


@dataclass
class ModelRegistry:
    """Mantém uma instância carregada por modelo do catálogo.

    Modelos sem pesos no disco simplesmente não ficam disponíveis: a API sobe do
    mesmo jeito e responde ``503`` só para eles, o que permite colocar
    ``leaf-foliar`` em produção enquanto os demais ainda estão em treino ou coleta.
    """

    specs: dict[str, ModelSpec] = field(default_factory=lambda: dict(MODEL_SPECS))

    def __post_init__(self) -> None:
        from app.model import CornDiseaseModel

        self._models = {key: CornDiseaseModel(spec) for key, spec in self.specs.items()}

    def load_all(self) -> None:
        """Tenta carregar todos os modelos do catálogo."""
        for model in self._models.values():
            model.load()

    def unload_all(self) -> None:
        for model in self._models.values():
            model.unload()

    def get(self, key: str):
        """Devolve o modelo de um identificador, validando-o antes."""
        get_spec(key)
        return self._models[key]

    def resolve(self, key: str | None = None, platform: Platform | None = None):
        """Escolhe o modelo a partir do identificador ou da plataforma.

        O identificador explícito tem precedência; a plataforma é o atalho para
        quem só sabe de onde a imagem veio.
        """
        if key:
            return self.get(key)
        if platform:
            return self.get(spec_for_platform(platform).key)
        return self.get(settings.default_model)

    @property
    def models(self) -> dict[str, object]:
        return dict(self._models)

    @property
    def loaded_keys(self) -> list[str]:
        return [key for key, model in self._models.items() if model.is_loaded]


registry = ModelRegistry()
"""Catálogo único usado pelas rotas, carregado no ``lifespan`` da aplicação."""
