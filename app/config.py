"""Configuração da aplicação.

Reúne as constantes de transporte HTTP e as variáveis de ambiente de deploy
(lidas de ``.env`` via ``pydantic-settings``). O que é específico de cada modelo
— classes, tamanho de entrada, arquivo de pesos — vive em :mod:`app.registry`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
"""Raiz do repositório — um nível acima do pacote ``app``."""

ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/jpg", "image/png", "image/webp"})
"""Filtro barato baseado no header do cliente.

É apenas uma primeira barreira: o ``Content-Type`` do multipart é informado pelo
cliente e pode ser forjado. A validação que realmente vale é a decodificação
feita pelo Pillow em :mod:`app.inference.preprocess`.
"""

UPLOAD_CHUNK_BYTES = 64 * 1024
"""Tamanho do bloco de leitura dos uploads, usado para cortar arquivos grandes
antes de carregá-los inteiros na memória."""

HTTP_422 = 422
"""422 pelo número, e não pela constante do Starlette.

As duas versões não coexistem: o Starlette renomeou
``HTTP_422_UNPROCESSABLE_ENTITY`` para ``..._CONTENT``, a versão antiga não tem o
nome novo, e na nova o nome antigo emite ``DeprecationWarning`` — que o
``filterwarnings`` deste projeto trata como erro. O número vale nas duas.
"""

DEFAULT_IMG_SIZE = 224
"""Entrada padrão da MobileNetV2, usada quando o modelo carregado não declara a
sua própria."""


class Settings(BaseSettings):
    """Parâmetros de deploy, sobrescritíveis por variáveis de ambiente."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # ``model_path``/``models_dir`` colidem com o namespace protegido
        # ``model_`` do Pydantic.
        protected_namespaces=(),
    )

    service_token: str | None = None
    """Token que o SaaS manda em ``Authorization: Bearer``. Com ele definido, toda
    rota exige o token, menos ``/health`` e a documentação. Vazio (padrão) deixa a
    API aberta — o uso local na máquina do operador; em produção o SaaS recusa
    subir sem ``INFERENCE_SERVICE_TOKEN``. Ver :mod:`app.service_auth`."""

    default_model: str = "leaf-foliar"
    """Modelo usado quando a requisição não informa qual quer."""

    rover_models: dict[str, str] = Field(default_factory=dict)
    """Versão usada por cultura quando o rover informa só a cultura, em JSON.

    Ex.: ``ROVER_MODELS='{"banana": "banana-foliar-cc"}'`` põe o rover na versão
    comercial da banana. Vazio: a versão recomendada de cada cultura no catálogo.
    """

    models_dir: Path = BASE_DIR / "models"
    data_dir: Path = BASE_DIR / "data"

    model_path: Path | None = None
    """Sobrescreve o caminho dos pesos apenas do modelo padrão.

    Existe para manter compatibilidade com deploys que já apontam ``MODEL_PATH``.
    Os demais modelos são sempre resolvidos dentro de ``models_dir``.
    """

    labels_path: Path | None = None
    """Idem, para o ``labels.json`` do modelo padrão."""

    basemap_tiles: str | None = None
    """Molde XYZ da imagem de satélite usada no encaixe de contorno.

    Vazio por padrão, e é deliberado: baixar tile de basemap para **processar**
    não é o mesmo uso que exibi-lo num mapa, e nem toda fonte permite. Aponte
    para o ortomosaico do próprio cliente ou para um provedor licenciado.
    Ver a nota no topo de ``app/basemap.py``.

    Exemplo: ``https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.jpg?access_token=...``
    """

    max_forward_batch: int = Field(default=32, gt=0)
    """Teto de imagens por passada pelo modelo, inclusive com TTA.

    Medido em 26/09/2026 numa RTX 5060 Ti de 8 GB, MobileNetV2 no backend torch:
    lote de 32 em 59 ms (545 imagens/s); lote de 128 em **7 s**, com 11,8 GB
    reservados. Passou da memória da placa, o driver do Windows pagina a VRAM na
    RAM — a GPU fica "100% ocupada" rendendo 18 imagens/s, e a máquina inteira
    trava junto. Um ``/predict/batch`` de 20 com TTA são 80 imagens: sem este
    teto, é exatamente esse caso.
    """

    max_upload_mb: int = Field(default=10, gt=0)
    max_batch_size: int = Field(default=20, gt=0)
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    log_level: str = "INFO"

    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    """Confiança mínima para o resultado ser considerado conclusivo.

    Abaixo dela a resposta vem com ``abstained: true``: um classificador de 4
    classes sempre devolve uma delas, então uma foto de solo ou de céu sairia
    como doença com alta confiança. Zero mantém o comportamento sem abstenção.
    """

    novelty_sensitivity: float = Field(default=1.0, gt=0.0)
    """Multiplica o limiar de novidade calibrado no treino.

    É o botão que regula o volume da fila de revisão sem precisar recalibrar:
    abaixo de 1,0 marca mais imagens (o agrônomo vê mais, inclusive mais falso
    alarme), acima de 1,0 marca menos. Pense nisso como o orçamento do tempo
    dele, não como um parâmetro de acurácia.
    """

    inference_tta: bool = False
    """Se ``/predict`` roda o TTA por padrão, quando a requisição não diz.

    As acurácias publicadas em ``MODELS.md`` são **com** TTA; sem ele a produção
    entrega a coluna de baixo da tabela (0,9498 contra 0,9555 no ``leaf-foliar``).
    Custa quatro passadas pelo modelo em vez de uma — por isso não é o padrão, e
    por isso também cada requisição pode pedir com ``?tta=true``.
    """

    rover_tta: bool = False
    """Se ``/predict/rover-frame`` roda o TTA nos recortes, quando a requisição não diz.

    Desligado, e isso foi medido. O TTA melhora o recorte (no campo do PlantSeg, em
    30/09/2026: banana 0,812 → 0,822; citros 0,951 → 0,969), mas **piora a planta**:
    no talhão de ponta a ponta (66 plantas de citros, 3 câmeras cada), as plantas
    doentes detectadas caíram de 29 para 24 de 33, sem queda nos alarmes falsos
    (6 de 33 nos dois). A média de quatro vistas dá probabilidades menos extremas, e
    a fusão por planta (:mod:`app.plant_fusion`) recebe menos evidência por recorte;
    ela foi ajustada com saídas de uma passada só. Também custa quase o dobro por
    quadro (2,3 s contra 1,3 s). Ligue com ``ROVER_TTA=true`` (ou ``?tta=true``) só
    depois de recalibrar a fusão para saídas com TTA.
    """

    rover_species_check: bool = False
    """Se a verificação de espécie (``<pesos>.species.npz``) vale nos recortes do rover.

    Desligado, e isso foi medido em 01/10/2026. Na foto inteira ela separa citros de
    outras plantas (barra 97% das folhas de outras espécies do PlantVillage, recusando 2
    a 3% dos recortes de lesão de citros do PlantSeg), mas o recorte do rover é outra
    coisa — pedaço de folha, borda, fundo: no talhão de ponta a ponta ela recusou **49%**
    dos recortes de citros (9.087 de 18.565), e a fusão por planta, com metade da
    evidência, passou a desconfiar de planta sadia: com a mesma cabeça de diagnóstico,
    12 de 33 plantas sadias viraram alarme com ela ligada, e nenhuma sem ela. No pomar,
    o rover já sabe a cultura da fileira; quem precisa da verificação é a foto avulsa.
    """

    rover_review_per_frame: int = Field(default=3, ge=0)
    """Quantos recortes de cada quadro do rover vão para a fila de revisão, no máximo.

    Com ``REVIEW_DIR``, a fila recebe do quadro o que mais ensina: primeiro o recorte
    estranho ao modelo, depois o de conjunto conformal ambíguo, e no máximo um que o
    portão recusou (pseudocaule, tronco — o dataset que ainda não existe). O teto
    existe porque o rover manda milhares de recortes por hora e o agrônomo rotula
    dezenas. ``0`` desliga.
    """

    inference_runtime: Literal["keras", "onnx"] = "keras"
    """Quem executa os modelos: o Keras (com o backend de ``keras_backend``) ou o
    ONNX Runtime, sobre os ``.onnx`` gravados pelo ``training.export_onnx``.

    Com ``onnx``, cada modelo sem ``.onnx`` ao lado dos pesos cai para o Keras
    sozinho. Ver :mod:`app.onnx_runtime`.
    """

    review_dir: Path | None = None
    """Onde arquivar as imagens marcadas para revisão do agrônomo.

    Vazio (padrão) desliga a fila: gravar imagem de cliente em disco é decisão
    de quem opera o serviço. Ver :mod:`app.review`.
    """

    fields_dir: Path | None = None
    """Onde guardar o estado dos talhões: fusão por planta e prevalência (ver
    :mod:`app.fields_api`). Vazio (padrão) desliga — o rover responde só por quadro.
    """

    fields_tile_log: bool = False
    """Se a fusão guarda as probabilidades de cada recorte com a planta a que foi associado
    (``<FIELDS_DIR>/plantas/registro/<talhão>.jsonl``).

    É o dado da validação de campo: com o rótulo do agrônomo por planta, o
    ``training.fit_fusion`` recalibra a fusão com o rover de verdade em vez de fotos
    avulsas. Custa disco (~25 KB por quadro), por isso vem desligado.
    """

    @field_validator(
        "model_path", "labels_path", "models_dir", "data_dir", "review_dir", "fields_dir"
    )
    @classmethod
    def _resolve_against_base_dir(cls, value: Path | None) -> Path | None:
        """Torna caminhos relativos independentes do diretório de trabalho."""
        if value is None:
            return None
        return value if value.is_absolute() else BASE_DIR / value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"Nível de log inválido: {value!r}")
        return level

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        """``CORS_ORIGINS`` como lista, aceitando o formato separado por vírgula."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allow_credentials(self) -> bool:
        """Só habilita credenciais com origens explícitas.

        ``Access-Control-Allow-Origin: *`` combinado com credenciais é inválido
        por spec, e o Starlette contorna refletindo a Origin de quem chamou — o
        que na prática libera qualquer site.
        """
        return "*" not in self.cors_origin_list


settings = Settings()
