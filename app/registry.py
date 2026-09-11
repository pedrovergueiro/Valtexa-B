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

    arch: str | None = None
    """Base convolucional. ``None``: a padrão do treino.

    Faz parte da identidade do modelo: treinar o ``leaf-foliar-b`` com a base do
    ``leaf-foliar`` produziria uma cópia, não uma segunda opinião."""

    notes: str = ""

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
    weights_name="corn_disease_model.keras",
    measures_severity=True,
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
    measures_severity=True,
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
        "close. Mesma arquitetura do leaf-foliar, para a cultura mais "
        "estratégica da região de atuação da empresa."
    ),
    class_names=("Healthy", "Rust", "Cercospora_Leaf_Spot", "Phoma_Leaf_Spot", "Leaf_Miner_Damage"),
    weights_name="coffee_foliar.keras",
    datasets=("jmuben-coffee", "rocole", "plantseg-coffee"),
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

MODEL_SPECS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (LEAF_FOLIAR, LEAF_FOLIAR_B, COFFEE_FOLIAR, ROVER_UNDERSTORY, AERIAL_CANOPY)
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
    for spec in (LEAF_FOLIAR, COFFEE_FOLIAR, ROVER_UNDERSTORY, AERIAL_CANOPY)
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
