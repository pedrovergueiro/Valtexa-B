"""Enquadramento: a foto é o que o modelo sabe ler?

É o caso mais perigoso que o projeto mediu, e nenhum filtro existente o pega:
uma foto **aérea** de cafezal vira ``Rust`` em 94% dos quadros, com confiança
média de 0,93, e só 9% são interceptados. No espaço de embeddings do
classificador, a textura verde de copa cai perto do que ele aprendeu como
ferrugem — a novidade não estranha, e a confiança não duvida.

O problema é que o classificador só conhece folhas. Para saber que uma foto
**não é** de folha, é preciso um modelo que conheça o resto do mundo.

## A tecnologia: visão-linguagem, zero-shot

O SigLIP 2 (Google, 2025) foi treinado em bilhões de pares imagem-texto e
compara uma foto com **frases**. Perguntamos a ele se a imagem se parece mais
com "a close-up photo of a maize leaf" ou com "an aerial drone photo of a farm
field", "a slide or document with text", "a photo of a plant stem". Nenhuma
imagem nossa entrou no treino dele, e nenhuma precisou ser rotulada: as cenas
são descritas em texto.

O limiar, esse sim, vem dos nossos dados: é calibrado só com fotos **dentro**
do escopo (a validação dos modelos foliares), no percentil que decide quantas
folhas boas podem ser recusadas. As fotos fora de escopo servem só para medir —
nunca para ajustar.

## Só a torre de visão em produção

O modelo completo tem 375 M de parâmetros, 283 M deles fora da torre de visão.
Mas as
frases são sempre as mesmas: o ``fit_framing`` calcula os embeddings delas uma
vez e os grava em ``models/framing.npz``. A API carrega só a torre de visão
(92 M) — medido: ela reproduz o embedding do modelo inteiro com diferença menor
que 1e-5.

## E sem torch, na ground station

O pré-processamento do SigLIP — 224x224 bilinear, pixels levados a ``[-1, 1]`` —
é **o mesmo** do :mod:`app.inference.preprocess`. Então a torre de visão exporta
para ONNX (``python -m training.export_onnx --framing``) e, com
``INFERENCE_RUNTIME=onnx``, o portão roda pelo ONNX Runtime a partir do lote que
a API já prepara, sem ``transformers`` e sem torch.

Opcional como a severidade: sem o arquivo, ou sem como executar a torre de
visão, a API sobe do mesmo jeito e ``framing`` vem nulo.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings
from app.inference.preprocess import preprocess_batch

logger = logging.getLogger(__name__)

REFERENCE_NAME = "framing.npz"
VISION_ONNX_NAME = "framing_vision.onnx"
IMG_SIZE = 224
DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"


@dataclass(frozen=True)
class Scene:
    """Uma cena que o portão reconhece, descrita em texto."""

    key: str
    prompt: str
    in_scope: bool
    crop: str | None = None


SCENES: tuple[Scene, ...] = (
    Scene("folha_milho", "a close-up photo of a maize leaf", True, "corn"),
    Scene("folha_cafe", "a close-up photo of a coffee leaf", True, "coffee"),
    Scene("folha_banana", "a close-up photo of a banana leaf", True, "banana"),
    Scene("folha_citros", "a close-up photo of a citrus leaf", True, "citrus"),
    Scene("folha_doente", "a close-up photo of a diseased plant leaf with spots and lesions", True),
    Scene("folha", "a close-up photo of a green leaf filling the frame", True),
    Scene("aereo", "an aerial drone photo of a farm field", False),
    Scene("dossel", "a top-down view of crop rows and plant canopy", False),
    Scene("colmo", "a photo of a plant stem or stalk", False),
    Scene("planta_inteira", "a photo of a whole plant in a field", False),
    Scene("texto", "a slide or document with text", False),
    Scene("solo", "a photo of bare soil", False),
    Scene("ceu", "a photo of the sky", False),
    Scene("pessoa", "a photo of a person", False),
    Scene("maquina", "a photo of a tractor or machine", False),
    Scene("fruto", "a photo of fruits or grains", False),
    Scene("pseudocaule", "a photo of a banana plant pseudostem", False),
    Scene("tronco", "a photo of a tree trunk with bark", False),
    Scene("fruto_citros", "a close-up photo of an orange or lemon fruit", False),
    Scene("corte", "a cross-section cut through a plant stem", False),
    Scene("inseto", "a photo of an insect on a plant", False),
    Scene("folha_outra", "a photo of a leaf of a vegetable plant or fruit tree", False),
)
"""As cenas, escritas **antes** de olhar as imagens de avaliação.

As de fora do escopo nomeiam os modos de falha que o projeto já documentou
(aéreo, colmo, slide) e o que mais aparece numa câmera de campo. Reescrevê-las
olhando o resultado da avaliação transformaria a medição em ajuste.

As quatro últimas vêm das zonas do rover (:data:`app.rover.ZONES`), não de
erros medidos: na faixa baixa da bananeira a câmera vê o **pseudocaule**; na do
citros, o **tronco** e os frutos da saia da copa. Nenhum dos modelos foliares
sabe ler essas partes, então o recorte vai para revisão em vez de receber uma
doença de folha. O **corte** transversal aparece nos datasets de Fusariose
(o sintoma interno do pseudocaule), que o rover nunca vê.

As duas últimas (``inseto`` e ``folha_outra``) entraram em 01/10/2026, depois de o
ensaio de ponta a ponta mostrar que o modelo de citros aceitava, sem alarme, fotos
de inseto (21%) e folhas de outras espécies (35%). As frases foram fixadas antes da
medição, a candidata foi escolhida só na metade "dev" das espécies e dos ids de
inseto, e o que vale é a metade "test": o inseto passou a ser recusado em 71 a 78%
(antes 41 a 51%), sem custo em folha boa nem em recorte de campo. A folha de outra
espécie quase não mudou onde importa: o citros recusa 1,3% das folhas do
PlantVillage (antes 0,3%) e 1,4 ponto a mais nas do PlantSeg. A visão-linguagem não
separa "folha de citros" de "folha de outra planta" por texto. O que separa é
comparar com fotos reais da cultura (vizinhos mais próximos no mesmo espaço), e isso
só funciona com referência de **campo**: com folhas sobre papel, ou recusa 99,8% dos
recortes de campo do citros (limiar do laboratório) ou deixa passar 99,8% das folhas
de outras espécies (limiar que aceita o campo). Com recortes de campo na referência,
o citros barrou 96% dos insetos, 63% das folhas do PlantVillage e 93% das do PlantSeg
e recusou 8,8% dos recortes de campo não vistos (limiar sem ajuste fino). Não foi
adotado: os recortes de campo disponíveis são do PlantSeg (CC BY-NC) e contaminariam
o portão da versão comercial. A referência certa é a do próprio rover, vinda da fila
de revisão.

O que entrou foi por modelo, não no portão comum: a verificação de espécie
(``<pesos>.species.npz``, :mod:`app.inference.heads`), uma cabeça treinada sobre este
mesmo embedding, com a licença do próprio modelo. No ``citrus-foliar`` (não comercial)
ela recusa 97% das folhas de outras espécies do PlantVillage na foto avulsa; no rover
fica desligada (``ROVER_SPECIES_CHECK``).
"""

CROP_MISMATCH_RATIO = 4.0
"""Quantas vezes a folha de outra cultura precisa superar a da escolhida para avisar.

Com o modelo escolhido por quem chama, a cultura já é conhecida — o portão não
precisa adivinhá-la, só desconfiar quando a evidência é forte. Sem essa folga,
folha de café (lisa e escura) era "folha de citros" em 43% das vezes depois que
as cenas de citros entraram, e o aviso de cultura trocada virava ruído. O
``fit_framing`` imprime a curva deste número (folha boa avisada à toa contra
folha de outra cultura pega) para cada modelo.
"""

LEAF_LIKE_OUTSIDE = ("folha_outra",)
"""Cenas fora do escopo que ainda são folha: vencer com elas não recusa a foto na regra da
"cena vencedora é folha" (medido em 16/10/2026: ~10% das folhas de café boas têm "folha de
outra planta" como cena mais provável — recusá-las custaria 13,6% de folha legítima)."""

Encoder = Callable[[list[Image.Image]], np.ndarray]
"""Imagens -> embeddings ``(N, D)`` da torre de visão, ainda não normalizados."""


@dataclass(frozen=True)
class FramingReference:
    """As frases já embutidas e o limiar calibrado nas fotos boas."""

    model_id: str
    keys: tuple[str, ...]
    prompts: tuple[str, ...]
    in_scope: np.ndarray
    """``(P,)`` booleano: a cena é o que os modelos foliares sabem ler."""

    crops: tuple[str, ...]
    """Cultura de cada cena, ``""`` quando a cena não tem cultura."""

    text_embeddings: np.ndarray
    """``(P, D)``, normalizados."""

    logit_scale: float
    threshold: float
    """Probabilidade mínima de estar no escopo para a foto passar."""

    calibrated_on: int

    thresholds: dict[str, float] = field(default_factory=dict)
    """Limiar por cultura, calibrado só com as cenas daquela cultura.

    Quando o modelo é escolhido, o portão compara a foto só com a folha da cultura
    dele e com o que não é folha — então o limiar certo é outro. Cultura ausente
    aqui usa o ``threshold`` geral."""

    leaf_winner_crops: tuple[str, ...] = ()
    """Culturas em que, além do limiar, a cena **mais provável** tem de ser folha.

    O limiar sozinho mede quanta probabilidade vai para as cenas do escopo; com ele baixo
    (calibrado para recusar poucas folhas boas), passava foto em que a cena vencedora era
    "pessoa": no café, em vídeos de lavoura (16/10/2026), 96 quadros de gente na frente da
    planta, 13 de cereja, 11 de planta inteira — e a cabeça chamava quase todos de ferrugem.
    Com a regra no café: folha de café boa recusada 1,8% → 3,4%, BRACOL 0% → 0%, e nenhum
    desses quadros passa. Fora desta lista, o portão é o de antes."""

    def allowed_scenes(self, crop: str | None = None) -> np.ndarray:
        """``(P,)`` booleano: as cenas que entram na conta para a cultura escolhida.

        Sai só a folha das **outras** culturas. Com o café escolhido, "folha de
        citros" deixa de disputar a foto — era essa disputa que derrubava a
        cultura certa do café para 57%.
        """
        crops = np.array(self.crops)
        if crop is None:
            return np.ones(len(crops), dtype=bool)
        return (crops == "") | (crops == crop)

    def scene_probabilities(
        self, image_embeddings: np.ndarray, crop: str | None = None
    ) -> np.ndarray:
        """``(N, P)``: a distribuição de cada imagem sobre as cenas permitidas.

        As cenas de fora (folha de outra cultura, quando ``crop`` é dado) ficam com
        probabilidade zero, e as demais são renormalizadas entre si.
        """
        images = np.asarray(image_embeddings, dtype=np.float64)
        images = images / np.linalg.norm(images, axis=1, keepdims=True)
        logits = self.logit_scale * images @ self.text_embeddings.T
        logits[:, ~self.allowed_scenes(crop)] = -np.inf
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)

    def in_scope_probability(
        self, image_embeddings: np.ndarray, crop: str | None = None
    ) -> np.ndarray:
        return self.scene_probabilities(image_embeddings, crop)[:, self.in_scope].sum(axis=1)

    def threshold_for(self, crop: str | None) -> float:
        return self.thresholds.get(crop, self.threshold) if crop else self.threshold

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            model_id=np.array(self.model_id),
            keys=np.array(self.keys),
            prompts=np.array(self.prompts),
            in_scope=self.in_scope.astype(bool),
            crops=np.array(self.crops),
            text_embeddings=self.text_embeddings,
            logit_scale=np.array(self.logit_scale),
            threshold=np.array(self.threshold),
            calibrated_on=np.array(self.calibrated_on),
            threshold_crops=np.array(sorted(self.thresholds), dtype=str),
            threshold_values=np.array([self.thresholds[c] for c in sorted(self.thresholds)]),
            leaf_winner_crops=np.array(sorted(self.leaf_winner_crops), dtype=str),
        )

    @classmethod
    def load(cls, path: Path) -> FramingReference:
        """Lê a referência do ``fit_framing``.

        Raises:
            ValueError: se o arquivo estiver ilegível ou incompleto.
        """
        try:
            with np.load(path, allow_pickle=False) as raw:
                reference = cls(
                    model_id=str(raw["model_id"]),
                    keys=tuple(str(k) for k in raw["keys"]),
                    prompts=tuple(str(p) for p in raw["prompts"]),
                    in_scope=raw["in_scope"].astype(bool),
                    crops=tuple(str(c) for c in raw["crops"]),
                    text_embeddings=raw["text_embeddings"].astype(np.float64),
                    logit_scale=float(raw["logit_scale"]),
                    threshold=float(raw["threshold"]),
                    calibrated_on=int(raw["calibrated_on"]),
                    # Referências anteriores ao limiar por cultura não têm estas chaves.
                    thresholds=(
                        {
                            str(c): float(v)
                            for c, v in zip(
                                raw["threshold_crops"], raw["threshold_values"], strict=True
                            )
                        }
                        if "threshold_crops" in raw
                        else {}
                    ),
                    leaf_winner_crops=(
                        tuple(str(c) for c in raw["leaf_winner_crops"])
                        if "leaf_winner_crops" in raw
                        else ()
                    ),
                )
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"Referência de enquadramento ilegível em {path}: {exc}") from exc
        if len(reference.keys) != len(reference.text_embeddings):
            raise ValueError(f"Referência de enquadramento em {path}: cenas e textos não batem.")
        return reference


@dataclass(frozen=True)
class FramingResult:
    """O veredito sobre uma imagem."""

    in_scope: bool
    score: float
    scene: str
    crop: str | None
    crop_matches: bool | None = None
    """Só com a cultura escolhida: falso quando a folha é claramente de outra."""


def judge(
    reference: FramingReference,
    image_embeddings: np.ndarray,
    crop: str | None = None,
    mismatch_ratio: float = CROP_MISMATCH_RATIO,
) -> list[FramingResult]:
    """Transforma embeddings de imagem em vereditos de enquadramento.

    Args:
        crop: a cultura do modelo escolhido. Com ela, o escopo e a cena são
            julgados só contra a folha dessa cultura (ver
            :meth:`FramingReference.allowed_scenes`), com o limiar dela; ``crop``
            do resultado continua sendo a cultura mais parecida entre todas, e
            ``crop_matches`` só é falso quando outra cultura supera a escolhida
            por ``mismatch_ratio`` vezes.
    """
    restricted = reference.scene_probabilities(image_embeddings, crop)
    everything = reference.scene_probabilities(image_embeddings) if crop else restricted
    scores = restricted[:, reference.in_scope].sum(axis=1)
    threshold = reference.threshold_for(crop)
    crop_columns = [i for i, c in enumerate(reference.crops) if c]
    leaf_like = reference.in_scope | np.isin(np.array(reference.keys), LEAF_LIKE_OUTSIDE)
    winner_rule = crop is not None and crop in reference.leaf_winner_crops
    own = [i for i in crop_columns if reference.crops[i] == crop]
    others = [i for i in crop_columns if reference.crops[i] != crop]

    results = []
    for row, full, score in zip(restricted, everything, scores, strict=True):
        detected = None
        if crop_columns:
            detected = reference.crops[max(crop_columns, key=lambda i: full[i])]
        matches = None
        if crop and own and others:
            strongest_other = max(full[i] for i in others)
            matches = bool(strongest_other < mismatch_ratio * max(full[i] for i in own))
        passes = score >= threshold
        if winner_rule:
            passes = passes and bool(leaf_like[int(np.argmax(row))])
        results.append(
            FramingResult(
                in_scope=bool(passes),
                score=float(score),
                scene=reference.keys[int(np.argmax(row))],
                crop=detected,
                crop_matches=matches,
            )
        )
    return results


class FramingModel:
    """A torre de visão e a referência, carregadas uma vez por processo."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.models_dir / REFERENCE_NAME
        self._reference: FramingReference | None = None
        self._encoder: Encoder | None = None

    @property
    def is_loaded(self) -> bool:
        return self._reference is not None and self._encoder is not None

    def load(self) -> None:
        """Carrega referência e torre de visão, se possível. Nunca derruba a API."""
        if not self.path.exists():
            logger.info(
                "Sem referência de enquadramento em %s — 'framing' virá nulo. "
                "Gere com 'python -m training.fit_framing'.",
                self.path,
            )
            return
        try:
            reference = FramingReference.load(self.path)
            encoder = self._encoder_for(reference)
        except Exception as exc:  # sem transformers, sem rede, pesos ruins: só desliga
            logger.error("Enquadramento desligado: %s", exc)
            return
        self.attach(encoder, reference)
        logger.info(
            "Enquadramento ativo: %s, limiar %.3f (calibrado em %d fotos boas).",
            reference.model_id,
            reference.threshold,
            reference.calibrated_on,
        )

    def _encoder_for(self, reference: FramingReference) -> Encoder:
        """ONNX quando pedido e exportado; senão, a torre do ``transformers``."""
        graph = self.path.parent / VISION_ONNX_NAME
        if settings.inference_runtime == "onnx" and graph.exists():
            logger.info("Enquadramento pelo ONNX Runtime: %s", graph)
            return onnx_vision_encoder(graph)
        return vision_encoder(reference.model_id)

    def attach(self, encoder: Encoder, reference: FramingReference) -> None:
        """Anexa um codificador já pronto — é também por aqui que os testes entram."""
        self._encoder = encoder
        self._reference = reference

    def unload(self) -> None:
        self._encoder = None
        self._reference = None

    @property
    def model_id(self) -> str | None:
        """A torre de visão em uso — as cabeças só valem para embeddings dela."""
        return None if self._reference is None else self._reference.model_id

    def encode(self, images: list[Image.Image]) -> np.ndarray | None:
        """Os embeddings da torre de visão, ou ``None`` sem portão ou em falha.

        É uma passada só: o veredito (:meth:`judge`) e as cabeças de
        :mod:`app.inference.heads` leem o mesmo embedding.
        """
        if not self.is_loaded or not images:
            return None
        # Em blocos, como os classificadores: a torre de visão segura as ativações
        # do lote inteiro, ~13,5 MB por recorte. Um quadro do rover a 1 m dá ~800
        # recortes, que de uma vez só pediriam ~11 GB.
        limit = settings.max_forward_batch
        try:
            return np.concatenate(
                [self._encoder(images[i : i + limit]) for i in range(0, len(images), limit)]
            )
        except (ValueError, RuntimeError) as exc:
            # Enquadramento é um extra: se falhar, a classificação continua valendo.
            logger.error("Falha ao avaliar enquadramento: %s", exc)
            return None

    def judge(
        self, embeddings: np.ndarray | None, crop: str | None = None
    ) -> list[FramingResult | None]:
        """O veredito de embeddings já calculados por :meth:`encode`."""
        if embeddings is None or self._reference is None:
            return [None] * (0 if embeddings is None else len(embeddings))
        try:
            return judge(self._reference, embeddings, crop)
        except (ValueError, RuntimeError) as exc:
            logger.error("Falha ao avaliar enquadramento: %s", exc)
            return [None] * len(embeddings)

    def assess(
        self, images: list[Image.Image], crop: str | None = None
    ) -> list[FramingResult | None]:
        """O veredito de cada imagem, numa passada só pela torre de visão.

        Args:
            crop: a cultura do modelo escolhido — ver :func:`judge`.
        """
        embeddings = self.encode(images)
        if embeddings is None:
            return [None] * len(images)
        return self.judge(embeddings, crop)


def vision_encoder(model_id: str) -> Encoder:
    """A torre de visão do SigLIP como função ``imagens -> embeddings``.

    Importa ``transformers`` e ``torch`` só aqui: a API continua subindo sem eles.
    """
    import torch
    from transformers import SiglipVisionModel

    try:  # transformers 5: o de PIL tem nome próprio, e o padrão exige torchvision
        from transformers import SiglipImageProcessorPil as Processor
    except ImportError:
        from transformers import SiglipImageProcessor as Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # O processador de PIL faz a mesma conta (224x224 bilinear, normalização
    # 0,5) do padrão, sem trazer o torchvision como dependência.
    processor = Processor.from_pretrained(model_id)
    model = SiglipVisionModel.from_pretrained(model_id).to(device).eval()

    def encode(images: list[Image.Image]) -> np.ndarray:
        inputs = processor(images=[image.convert("RGB") for image in images], return_tensors="pt")
        with torch.no_grad():
            output: Any = model(pixel_values=inputs["pixel_values"].to(device))
        return output.pooler_output.float().cpu().numpy()

    return encode


def onnx_vision_encoder(path: Path) -> Encoder:
    """A torre de visão exportada, alimentada pelo pré-processamento da própria API."""
    from app.onnx_runtime import OnnxModel

    session = OnnxModel(path)

    def encode(images: list[Image.Image]) -> np.ndarray:
        # O redimensionamento é o do Pillow, que é o do processador do SigLIP: com o da
        # API (estilo TensorFlow, sem antialias) o embedding saía diferente ao reduzir
        # uma foto grande — cosseno de até 0,93 e a cena trocada em até 12% das imagens
        # (medido em 01/10/2026). Com o do Pillow, o cosseno é 1,00000 e a cena e o
        # veredito saem 100% iguais aos do servidor.
        batch = preprocess_batch(
            [image.convert("RGB") for image in images], IMG_SIZE, resampling="pillow"
        )
        return session.run(batch.transpose(0, 3, 1, 2))[0]

    return encode


framing_model = FramingModel()
"""Instância única, carregada no ``lifespan`` da aplicação."""
