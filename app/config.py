"""Configuração da aplicação.

Reúne as constantes de transporte HTTP e as variáveis de ambiente de deploy
(lidas de ``.env`` via ``pydantic-settings``). O que é específico de cada modelo
— classes, tamanho de entrada, arquivo de pesos — vive em :mod:`app.registry`.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

    default_model: str = "leaf-foliar"
    """Modelo usado quando a requisição não informa qual quer."""

    models_dir: Path = BASE_DIR / "models"
    data_dir: Path = BASE_DIR / "data"

    model_path: Path | None = None
    """Sobrescreve o caminho dos pesos apenas do modelo padrão.

    Existe para manter compatibilidade com deploys que já apontam ``MODEL_PATH``.
    Os demais modelos são sempre resolvidos dentro de ``models_dir``.
    """

    labels_path: Path | None = None
    """Idem, para o ``labels.json`` do modelo padrão."""

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

    review_dir: Path | None = None
    """Onde arquivar as imagens marcadas para revisão do agrônomo.

    Vazio (padrão) desliga a fila: gravar imagem de cliente em disco é decisão
    de quem opera o serviço. Ver :mod:`app.review`.
    """

    @field_validator("model_path", "labels_path", "models_dir", "data_dir", "review_dir")
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
