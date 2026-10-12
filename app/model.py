"""Carregamento dos pesos e execução das previsões de um modelo do catálogo.

O TensorFlow só é importado dentro de :meth:`CornDiseaseModel.load`, o que
mantém o *startup* da API rápido e permite testar as rotas sem ter o TF
instalado no ambiente.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from app import metrics as metricas_ia
from app.config import DEFAULT_IMG_SIZE, settings
from app.inference.conformal import ConformalReference
from app.inference.explain import (
    DEFAULT_GRID,
    RISE_SAMPLES,
    occlusion_map,
    region_box,
    render_overlay,
    rise_map,
)
from app.inference.heads import DiagnosisHead, SpeciesCheck
from app.inference.novelty import NoveltyReference
from app.inference.preprocess import load_image_from_bytes, preprocess_batch
from app.inference.quality import ISSUE_ACTIONS, QualityReference, measure
from app.keras_backend import configure, no_grad, to_numpy
from app.schemas import (
    CaptureQuality,
    ClassProbability,
    ExplanationResponse,
    FramingCheck,
    PredictionResponse,
)

if TYPE_CHECKING:  # evita o ciclo de import app.model <-> app.registry
    from app.registry import ModelSpec

logger = logging.getLogger(__name__)

EMBEDDING_LAYER = "embedding"
"""Camada de onde sai a assinatura visual. Espelha ``training.model_def``, que
não é importado aqui para manter a API livre do TensorFlow."""

Upload = tuple[str, bytes]
"""Par ``(nome do arquivo, bytes da imagem)`` recebido pelas rotas."""

TTA_VIEWS = 4
"""Orientações do TTA: original, espelhada na horizontal, na vertical e nas duas.

São as mesmas quatro do ``training.metrics.evaluate_dataset`` — e tem de ser,
porque é com elas que as acurácias do ``MODELS.md`` foram medidas."""


class ModelNotLoadedError(RuntimeError):
    """Uma previsão foi pedida sem que houvesse pesos carregados."""


class UnknownClassError(ValueError):
    """A classe pedida para explicação não existe neste modelo."""


class ModelConfigurationError(RuntimeError):
    """Os pesos e os rótulos carregados são incompatíveis entre si."""


class CornDiseaseModel:
    """Encapsula um modelo Keras e a tradução de índices para nomes de classe."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model: Any = None
        self._forward: Any = None
        self._novelty: NoveltyReference | None = None
        self._quality: QualityReference | None = None
        self._conformal: ConformalReference | None = None
        self._head: DiagnosisHead | None = None
        self._species: SpeciesCheck | None = None
        self._class_names: tuple[str, ...] = tuple(spec.class_names)
        self._img_size: int = spec.img_size

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def weights_path(self) -> Path:
        return self.spec.weights_path

    @property
    def labels_path(self) -> Path:
        return self.spec.labels_path

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def img_size(self) -> int:
        return self._img_size

    def load(self) -> None:
        """Carrega os pesos e os rótulos do disco.

        Falhas são registradas em log sem derrubar o processo: a API sobe assim
        mesmo e responde ``503`` apenas para este modelo, o que permite colocar
        um modelo em produção enquanto os outros ainda estão em coleta.
        """
        if settings.inference_runtime == "onnx" and self._load_onnx():
            return

        path = self.weights_path
        if not path.exists():
            logger.warning(
                "[%s] sem pesos em %s — este modelo vai responder 503. "
                "Treine com 'python -m training.train --model %s'.",
                self.key,
                path,
                self.key,
            )
            return

        configure()
        try:
            import keras  # import tardio: custa alguns segundos no startup
        except ImportError:
            # O pacote de borda (requirements-edge.txt) não traz o Keras. Um modelo
            # sem .onnx fica fora; a API sobe com os outros.
            logger.error(
                "[%s] há pesos Keras em %s, mas o Keras não está instalado — este modelo "
                "vai responder 503. Exporte com 'python -m training.export_onnx --model %s'.",
                self.key,
                path,
                self.key,
            )
            return

        logger.info("[%s] carregando de %s ...", self.key, path)
        try:
            self.attach(keras.models.load_model(path), self._read_class_names())
        except (ModelConfigurationError, OSError, ValueError) as exc:
            self._model = None
            logger.error("[%s] falha ao carregar de %s: %s", self.key, path, exc)
            return

        logger.info(
            "[%s] pronto — entrada %dx%d, %d classes: %s",
            self.key,
            self._img_size,
            self._img_size,
            len(self._class_names),
            ", ".join(self._class_names),
        )

    def _load_onnx(self) -> bool:
        """Tenta servir pelo ONNX Runtime; ``False`` devolve a vez ao Keras."""
        from app.onnx_runtime import onnx_path

        graph = onnx_path(self.weights_path)
        if not graph.exists():
            logger.warning(
                "[%s] INFERENCE_RUNTIME=onnx, mas não há %s — usando o Keras. "
                "Exporte com 'python -m training.export_onnx --model %s'.",
                self.key,
                graph,
                self.key,
            )
            return False
        try:
            from app.onnx_runtime import OnnxModel

            session = OnnxModel(graph)
            self.attach(session, self._read_class_names())
        except (ImportError, ModelConfigurationError, OSError, RuntimeError, ValueError) as exc:
            self._model = None
            logger.error("[%s] falha ao carregar %s: %s — usando o Keras.", self.key, graph, exc)
            return False
        logger.info("[%s] pronto via ONNX Runtime (%s).", self.key, ", ".join(session.providers))
        return True

    def attach(self, keras_model: Any, class_names: Sequence[str] | None = None) -> None:
        """Registra um modelo já instanciado.

        É o ponto de entrada usado por :meth:`load` e também pelos testes, que
        injetam um dublê para exercitar as rotas sem depender do TensorFlow.

        Raises:
            ModelConfigurationError: se a quantidade de rótulos não bater com o
                número de saídas do modelo. Mapear índices para nomes errados
                produziria previsões incorretas sem nenhum sinal de erro.
        """
        names = tuple(class_names) if class_names else tuple(self.spec.class_names)
        units = _output_units(keras_model)
        if units is not None and units != len(names):
            raise ModelConfigurationError(
                f"O modelo tem {units} saídas, mas foram informados {len(names)} rótulos "
                f"({', '.join(names)}). Verifique o {self.labels_path.name} ao lado dos pesos."
            )

        size = _square_input_size(keras_model)
        if size is None:
            logger.warning(
                "[%s] não foi possível inferir o tamanho de entrada — usando %d.",
                self.key,
                self.spec.img_size or DEFAULT_IMG_SIZE,
            )

        self._model = keras_model
        self._forward = _build_forward(keras_model)
        self._class_names = names
        self._img_size = size or self.spec.img_size or DEFAULT_IMG_SIZE
        self._novelty = self._read_novelty_reference(names)
        self._quality = self._read_quality_reference()
        self._conformal = self._read_conformal_reference(names)
        self._head = self._read_head(names)
        self._species = self._read_species()

    def unload(self) -> None:
        """Descarta o modelo em memória e volta aos rótulos de fallback."""
        self._model = None
        self._forward = None
        self._novelty = None
        self._quality = None
        self._conformal = None
        self._head = None
        self._species = None
        self._class_names = tuple(self.spec.class_names)
        self._img_size = self.spec.img_size

    @property
    def detects_novelty(self) -> bool:
        """Se este modelo consegue apontar o que nunca viu."""
        return self._novelty is not None

    @property
    def runtime(self) -> str | None:
        """``onnx`` ou ``keras`` quando carregado; ``None`` sem pesos."""
        if self._model is None:
            return None
        return "onnx" if callable(getattr(self._model, "dual_forward", None)) else "keras"

    @property
    def safeguards(self) -> dict[str, bool]:
        """O que este modelo sabe dizer além da classe, agora, com o que carregou."""
        from app.framing import framing_model

        gate = self.spec.expects_leaf_closeup and framing_model.is_loaded
        return {
            "novelty": self._novelty is not None,
            "quality": self._quality is not None,
            "conformal": self._conformal is not None,
            "framing": gate,
            "species": gate and self._compatible(self._species),
            "diagnosis_head": gate and self._compatible(self._head),
        }

    @property
    def checks_quality(self) -> bool:
        """Se este modelo sabe dizer que uma foto precisa ser refeita."""
        return self._quality is not None

    def predict(
        self, filename: str, image_bytes: bytes, tta: bool | None = None
    ) -> PredictionResponse:
        """Classifica uma única imagem."""
        return self.predict_many([(filename, image_bytes)], tta=tta)[0]

    def predict_many(
        self, uploads: Sequence[Upload], tta: bool | None = None
    ) -> list[PredictionResponse]:
        """Classifica várias imagens em uma única passada pelo modelo.

        Args:
            tta: roda as quatro orientações e tira a média. ``None`` segue
                ``INFERENCE_TTA``.

        Raises:
            ModelNotLoadedError: se não houver pesos carregados.
            InvalidImageError: se alguma das imagens não puder ser decodificada.
        """
        self._require_loaded()
        if not uploads:
            return []

        images = [load_image_from_bytes(data, filename) for filename, data in uploads]
        return self.predict_images([filename for filename, _ in uploads], images, tta=tta)

    def predict_images(
        self,
        filenames: Sequence[str],
        images: Sequence[Image.Image],
        tta: bool | None = None,
        species: bool = True,
    ) -> list[PredictionResponse]:
        """Classifica imagens já decodificadas.

        É o que permite a quem precisa dos dois modos — as calibrações e as
        medições — decodificar cada foto uma vez só. Decodificar JPEG grande é o
        gargalo delas, não a rede.

        Args:
            species: aplica a verificação de espécie, se houver. O rover desliga — ver
                ``Settings.rover_species_check``.
        """
        inicio = perf_counter()
        try:
            self._require_loaded()
            if not images:
                return []
            use_tta = settings.inference_tta if tta is None else tta
            respostas = self._predict_images(list(filenames), list(images), use_tta, species)[0]
        except Exception as exc:
            # Métrica da IA (seção 35): o erro conta por modelo e tipo, e segue adiante.
            metricas_ia.registrar_erro(self.key, exc)
            raise
        metricas_ia.registrar_inferencia(self.key, (r.predicted_class for r in respostas),
                                         perf_counter() - inicio, len(respostas))
        return respostas

    def explain(
        self,
        filename: str,
        image_bytes: bytes,
        target_class: str | None = None,
        grid: int = DEFAULT_GRID,
        overlay: bool = True,
        method: str = "occlusion",
        samples: int = RISE_SAMPLES,
    ) -> ExplanationResponse:
        """Mapeia onde está a evidência de uma classe na imagem.

        ``occlusion`` cobre uma célula por vez (``grid² + 1`` passadas) e é cego
        para evidência redundante — várias lesões, qualquer uma bastando. ``rise``
        cobre metade da imagem ao acaso ``samples`` vezes e enxerga esse caso.
        Ver :mod:`app.inference.explain`.

        A previsão que acompanha a explicação é sem TTA: o mapa explica uma
        passada, e misturá-lo com a média de quatro orientações explicaria uma
        resposta que o mapa não viu.

        Raises:
            ModelNotLoadedError: se não houver pesos carregados.
            InvalidImageError: se a imagem não puder ser decodificada.
            UnknownClassError: se ``target_class`` não for uma classe do modelo.
        """
        self._require_loaded()
        image = load_image_from_bytes(image_bytes, filename)
        [prediction], batch = self._predict_images([filename], [image], tta=False)

        target = target_class or prediction.predicted_class
        if target not in self._class_names:
            raise UnknownClassError(
                f"Classe {target!r} não existe no modelo {self.key!r}. "
                f"Classes: {', '.join(self._class_names)}."
            )

        index = self._class_names.index(target)
        base = prediction.probabilities[index].probability
        if self._head is not None:
            # A resposta é a mistura com a cabeça, mas o mapa só perturba a CNN: a base
            # das quedas tem de ser a da CNN sozinha, ou as quedas sairiam de duas contas.
            base = float(self._probabilities(batch)[0][index])
        if method == "rise":
            occlusion = rise_map(
                self._probabilities,
                batch[0],
                index,
                grid=grid,
                samples=samples,
                base_probability=base,
            )
            passes = samples + 1
        else:
            occlusion = occlusion_map(
                self._probabilities, batch[0], index, grid=grid, base_probability=base
            )
            passes = grid * grid + 1
        row, col = occlusion.peak
        return ExplanationResponse(
            prediction=prediction,
            target_class=target,
            method=method,
            grid=grid,
            base_probability=occlusion.base_probability,
            drops=[[round(float(v), 4) for v in line] for line in occlusion.drops],
            max_drop=occlusion.max_drop,
            peak_region=region_box(row, col, grid, image.width, image.height),
            passes=passes,
            overlay_png=(
                base64.b64encode(render_overlay(image, occlusion)).decode("ascii")
                if overlay
                else None
            ),
        )

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError(
                f"Modelo {self.key!r} não carregado. Verifique {self.weights_path} e os logs."
            )

    def _predict_images(
        self, filenames: list[str], images: list[Image.Image], tta: bool, species: bool = True
    ) -> tuple[list[PredictionResponse], np.ndarray]:
        """Classifica imagens já decodificadas e devolve também o lote do modelo."""
        batch = preprocess_batch(images, self._img_size)

        # Chamada direta em vez de ``predict()``: para lotes pequenos, o modo
        # eager evita o overhead por chamada do laço interno do Keras. As
        # probabilidades e os embeddings saem da *mesma* passada — extrair a
        # assinatura visual não custa uma segunda inferência.
        start = perf_counter()
        if tta:
            probabilities, embeddings, agreement = self._forward_views(batch)
        else:
            probabilities, embeddings = self._run(batch)
            self._check_output_shape(probabilities, len(images))
            agreement = [None] * len(images)
        elapsed_ms = (perf_counter() - start) * 1000

        # O embedding do portão sai antes do conjunto conformal: a cabeça muda a
        # resposta, e o conjunto tem de ser o da resposta que vai sair.
        gate_embeddings = self._gate_embeddings(images)
        probabilities = self._apply_head(probabilities, gate_embeddings)

        novelty = self._score_novelty(embeddings, len(images))
        severities = self._estimate_severity(images)
        qualities = self._assess_quality(batch)
        sets = self._prediction_sets(probabilities, tta)
        framings = self._check_framing(gate_embeddings, len(images), species)

        per_image_ms = elapsed_ms / len(images)
        responses = [
            self._to_response(
                filename,
                row,
                novelty_score=score,
                inference_ms=per_image_ms,
                severity=severity,
                view_agreement=views,
                quality=quality,
                prediction_set=candidates,
                framing=framing,
            )
            for filename, row, score, severity, views, quality, candidates, framing in zip(
                filenames,
                probabilities,
                novelty,
                severities,
                agreement,
                qualities,
                sets,
                framings,
                strict=True,
            )
        ]
        return responses, batch

    def _run(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Passa o lote pelo modelo em blocos de até ``MAX_FORWARD_BATCH`` imagens.

        Um lote grande demais não falha: estoura a memória da GPU, o driver pagina
        a VRAM na RAM do sistema e tudo fica dezenas de vezes mais lento — a
        máquina inclusive. Ver ``Settings.max_forward_batch``.
        """
        limit = settings.max_forward_batch
        probabilities, embeddings = [], []
        for start in range(0, len(batch), limit):
            outputs, embedded = self._forward(np.ascontiguousarray(batch[start : start + limit]))
            probabilities.append(to_numpy(outputs))
            embeddings.append(None if embedded is None else to_numpy(embedded))
        merged = np.clip(np.concatenate(probabilities).astype(np.float64), 0.0, 1.0)
        if any(part is None for part in embeddings):
            return merged, None
        return merged, np.concatenate(embeddings)

    def _forward_views(self, batch: np.ndarray) -> tuple[np.ndarray, Any, list[float]]:
        """TTA: as quatro orientações do lote, em blocos de tamanho limitado.

        A novidade é pontuada **só na orientação original**. A referência foi
        calibrada com embeddings de imagens não espelhadas; somar ou tirar média
        das quatro mudaria a distribuição que o limiar mede, e o score deixaria de
        significar o que o ``fit_novelty`` calibrou.
        """
        size = len(batch)
        views = np.concatenate(
            [batch, batch[:, :, ::-1], batch[:, ::-1], batch[:, ::-1, ::-1]], axis=0
        )
        stacked, embeddings = self._run(views)
        self._check_output_shape(stacked, TTA_VIEWS * size)
        per_view = stacked.reshape(TTA_VIEWS, size, -1)
        probabilities = per_view.mean(axis=0)

        votes = per_view.argmax(axis=2)
        agreement = (votes == probabilities.argmax(axis=1)).mean(axis=0)
        original = None if embeddings is None else embeddings[:size]
        return probabilities, original, [float(value) for value in agreement]

    def _probabilities(self, batch: np.ndarray) -> np.ndarray:
        """Só as probabilidades de um lote — o que a explicação por oclusão consome."""
        return self._run(batch)[0]

    def _gate_embeddings(self, images: list[Image.Image]) -> np.ndarray | None:
        """O embedding do portão, para os modelos de folha em close; uma passada só.

        O veredito do portão, a cabeça de diagnóstico e a verificação de espécie
        leem todos este mesmo embedding.
        """
        if not self.spec.expects_leaf_closeup:
            return None
        from app.framing import framing_model

        return framing_model.encode(images)

    @staticmethod
    def _compatible(part: DiagnosisHead | SpeciesCheck | None) -> bool:
        """Cabeça treinada sobre a torre de visão que o portão está usando agora."""
        from app.framing import framing_model

        return part is not None and framing_model.model_id == part.encoder

    def _usable(self, part: DiagnosisHead | SpeciesCheck | None) -> bool:
        """:meth:`_compatible`, avisando no log quando a cabeça é de outra torre."""
        if part is None:
            return False
        if not self._compatible(part):
            from app.framing import framing_model

            logger.error(
                "[%s] cabeça treinada sobre %s, mas o portão usa %s — ignorando.",
                self.key,
                part.encoder,
                framing_model.model_id,
            )
            return False
        return True

    def _apply_head(
        self, probabilities: np.ndarray, gate_embeddings: np.ndarray | None
    ) -> np.ndarray:
        """A resposta com a segunda leitura (:mod:`app.inference.heads`), se houver."""
        if gate_embeddings is None or not self._usable(self._head):
            return probabilities
        try:
            return self._head.combine(probabilities, gate_embeddings)
        except ValueError as exc:
            # A cabeça é um extra: se falhar, a CNN sozinha continua respondendo.
            logger.error("[%s] falha na cabeça de diagnóstico: %s", self.key, exc)
            return probabilities

    def _check_framing(
        self, gate_embeddings: np.ndarray | None, count: int, species: bool = True
    ) -> list[FramingCheck | None]:
        """Se cada foto é folha em close da cultura certa, pelo modelo de visão-linguagem.

        Com a verificação de espécie, a folha que não é da cultura do modelo sai do
        escopo, como a aérea ou o colmo: a cena vira ``folha_outra``. O texto das
        cenas não separa "folha de citros" de "folha de outra planta"; a cabeça
        treinada com fotos das duas separa.
        """
        if not self.spec.expects_leaf_closeup or gate_embeddings is None:
            return [None] * count
        from app.framing import framing_model

        scores = None
        if species and self._usable(self._species):
            try:
                scores = self._species.scores(gate_embeddings)
            except ValueError as exc:
                logger.error("[%s] falha na verificação de espécie: %s", self.key, exc)

        checks: list[FramingCheck | None] = []
        # A cultura do modelo escolhido vai para o portão: a folha é julgada só
        # contra a folha dessa cultura, e o aviso de cultura trocada exige folga.
        verdicts = framing_model.judge(gate_embeddings, crop=self.spec.crop)
        for index, result in enumerate(verdicts):
            if result is None:
                checks.append(None)
                continue
            score = None if scores is None else float(scores[index])
            other_plant = score is not None and score < self._species.threshold
            checks.append(
                FramingCheck(
                    in_scope=result.in_scope and not other_plant,
                    score=round(result.score, 4),
                    scene="folha_outra" if other_plant and result.in_scope else result.scene,
                    crop=result.crop,
                    crop_matches=False if other_plant else result.crop_matches,
                    species_score=None if score is None else round(score, 4),
                )
            )
        return checks

    def _prediction_sets(self, probabilities: np.ndarray, tta: bool) -> list[list[str] | None]:
        """O conjunto conformal de cada imagem, no modo em que ela foi prevista."""
        if self._conformal is None:
            return [None] * len(probabilities)
        try:
            return list(self._conformal.prediction_sets(probabilities, tta))
        except ValueError as exc:
            # O conjunto é um extra: sem calibração para este modo, a previsão segue.
            logger.error("[%s] %s", self.key, exc)
            return [None] * len(probabilities)

    def fusion_prediction_sets(
        self, results: Sequence[PredictionResponse], tta: bool
    ) -> list[list[str] | None]:
        """Os conjuntos da calibração **original**, que a fusão por planta espera.

        O ``prediction_set`` da resposta é o honesto: alargado com o campo quando a
        referência foi alargada. A fusão divide a evidência de cada recorte pelo tamanho
        do conjunto e foi ajustada com os conjuntos da calibração original; entregar a
        ela os alargados derrubou a detecção da banana de 9 para 7 de 18 plantas
        doentes. Sem alargamento, são os mesmos da resposta.
        """
        if self._conformal is None or not results:
            return [result.prediction_set for result in results]
        probabilities = np.array(
            [[item.probability for item in result.probabilities] for result in results]
        )
        try:
            return list(self._conformal.prediction_sets(probabilities, tta, base=True))
        except ValueError as exc:
            logger.error("[%s] %s", self.key, exc)
            return [result.prediction_set for result in results]

    def _assess_quality(self, batch: np.ndarray) -> list[CaptureQuality | None]:
        """Qualidade da captura de cada imagem, medida no lote que o modelo recebe."""
        if self._quality is None:
            return [None] * len(batch)
        assessed = []
        for metrics in measure(batch):
            issues = self._quality.assess(metrics)
            assessed.append(
                CaptureQuality(
                    ok=not issues,
                    issues=issues,
                    actions=[ISSUE_ACTIONS[issue] for issue in issues],
                    sharpness=round(metrics.sharpness, 2),
                    brightness=round(metrics.brightness, 2),
                    contrast=round(metrics.contrast, 2),
                )
            )
        return assessed

    def _estimate_severity(self, images: list) -> list[float | None]:
        """Severidade do lote, só para os modelos que a medem."""
        if not self.spec.measures_severity:
            return [None] * len(images)
        from app.severity import severity_model

        return severity_model.estimate(images)

    def _score_novelty(self, embeddings: Any, expected_rows: int) -> list[float | None]:
        """Pontua a novidade do lote, ou devolve ``None`` quando indisponível."""
        if self._novelty is None or embeddings is None:
            return [None] * expected_rows
        try:
            scores = self._novelty.score(to_numpy(embeddings).astype(np.float64))
        except (ValueError, np.linalg.LinAlgError) as exc:
            # Novidade é um extra: se falhar, a previsão principal continua boa.
            logger.error("[%s] falha ao pontuar novidade: %s", self.key, exc)
            return [None] * expected_rows
        return [float(score) for score in scores]

    def _to_response(
        self,
        filename: str,
        probabilities: np.ndarray,
        *,
        novelty_score: float | None,
        inference_ms: float,
        severity: float | None = None,
        view_agreement: float | None = None,
        quality: CaptureQuality | None = None,
        prediction_set: list[str] | None = None,
        framing: FramingCheck | None = None,
    ) -> PredictionResponse:
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        return PredictionResponse(
            filename=filename,
            model_key=self.key,
            predicted_class=self._class_names[best_index],
            confidence=confidence,
            abstained=confidence < settings.min_confidence,
            novelty_score=novelty_score,
            needs_review=_is_novel(novelty_score),
            severity=severity,
            view_agreement=view_agreement,
            quality=quality,
            framing=framing,
            prediction_set=prediction_set,
            set_coverage=None if prediction_set is None else self._conformal.coverage,
            probabilities=[
                ClassProbability(class_name=name, probability=float(probability))
                for name, probability in zip(self._class_names, probabilities, strict=True)
            ],
            inference_ms=inference_ms,
        )

    def _read_novelty_reference(self, names: tuple[str, ...]) -> NoveltyReference | None:
        """Carrega a referência de novidade gravada pelo ``fit_novelty``."""
        path = self.spec.novelty_path
        if not path.exists():
            logger.info(
                "[%s] sem referência de novidade em %s — /predict não vai marcar "
                "imagens para revisão. Gere com 'python -m training.fit_novelty --model %s'.",
                self.key,
                path,
                self.key,
            )
            return None

        try:
            reference = NoveltyReference.load(path)
        except ValueError as exc:
            logger.error("[%s] %s", self.key, exc)
            return None

        if reference.class_names != names:
            logger.error(
                "[%s] referência de novidade calibrada para %s, mas o modelo usa %s — ignorando.",
                self.key,
                ", ".join(reference.class_names),
                ", ".join(names),
            )
            return None

        logger.info(
            "[%s] detecção de novidade ativa (calibrada em %d imagens).",
            self.key,
            reference.fitted_on,
        )
        return reference

    def _read_head(self, names: tuple[str, ...]) -> DiagnosisHead | None:
        """Carrega a cabeça de diagnóstico gravada pelo ``fit_heads``."""
        path = self.spec.head_path
        if not path.exists():
            return None
        try:
            head = DiagnosisHead.load(path)
        except ValueError as exc:
            logger.error("[%s] %s", self.key, exc)
            return None
        if head.class_names != names:
            logger.error(
                "[%s] cabeça treinada para %s, mas o modelo usa %s — ignorando.",
                self.key,
                ", ".join(head.class_names),
                ", ".join(names),
            )
            return None
        logger.info(
            "[%s] cabeça de diagnóstico ativa: MLP %s, mistura %s.",
            self.key,
            "-".join(str(n) for n in head.mlp.layer_sizes),
            head.mix,
        )
        return head

    def _read_species(self) -> SpeciesCheck | None:
        """Carrega a verificação de espécie gravada pelo ``fit_heads --species``."""
        path = self.spec.species_path
        if not path.exists():
            return None
        try:
            check = SpeciesCheck.load(path)
        except ValueError as exc:
            logger.error("[%s] %s", self.key, exc)
            return None
        if check.crop != (self.spec.crop or ""):
            logger.error(
                "[%s] verificação de espécie de %r, mas o modelo é de %r — ignorando.",
                self.key,
                check.crop,
                self.spec.crop,
            )
            return None
        logger.info("[%s] verificação de espécie ativa (limiar %.3f).", self.key, check.threshold)
        return check

    def _read_conformal_reference(self, names: tuple[str, ...]) -> ConformalReference | None:
        """Carrega os limiares gravados pelo ``fit_conformal``."""
        path = self.spec.conformal_path
        if not path.exists():
            logger.info(
                "[%s] sem calibração conformal em %s — 'prediction_set' virá nulo. "
                "Gere com 'python -m training.fit_conformal --model %s'.",
                self.key,
                path,
                self.key,
            )
            return None

        try:
            reference = ConformalReference.load(path)
        except ValueError as exc:
            logger.error("[%s] %s", self.key, exc)
            return None

        if reference.class_names != names:
            # Limiar da classe errada é garantia da classe errada.
            logger.error(
                "[%s] calibração conformal feita para %s, mas o modelo usa %s — ignorando.",
                self.key,
                ", ".join(reference.class_names),
                ", ".join(names),
            )
            return None

        logger.info(
            "[%s] predição conformal ativa (cobertura %.0f%%, %d imagens de calibração).",
            self.key,
            reference.coverage * 100,
            reference.calibrated_on,
        )
        return reference

    def _read_quality_reference(self) -> QualityReference | None:
        """Carrega a referência de qualidade gravada pelo ``fit_quality``."""
        path = self.spec.quality_path
        if not path.exists():
            logger.info(
                "[%s] sem referência de qualidade em %s — 'quality' virá nulo. "
                "Gere com 'python -m training.fit_quality --model %s'.",
                self.key,
                path,
                self.key,
            )
            return None

        try:
            reference = QualityReference.load(path)
        except ValueError as exc:
            logger.error("[%s] %s", self.key, exc)
            return None

        if reference.img_size != self._img_size:
            # Nitidez medida a 224 não se compara com nitidez a 384: o mesmo
            # desfoque ocupa outro número de pixels.
            logger.error(
                "[%s] referência de qualidade medida a %d px, mas o modelo recebe %d — "
                "ignorando. Recalibre com 'python -m training.fit_quality --model %s'.",
                self.key,
                reference.img_size,
                self._img_size,
                self.key,
            )
            return None

        logger.info(
            "[%s] verificação de qualidade ativa (calibrada em %d imagens).",
            self.key,
            reference.fitted_on,
        )
        return reference

    def _check_output_shape(self, probabilities: np.ndarray, expected_rows: int) -> None:
        expected = (expected_rows, len(self._class_names))
        if probabilities.shape != expected:
            raise ModelConfigurationError(
                f"O modelo devolveu saída de shape {probabilities.shape}, esperado {expected}."
            )

    def _read_class_names(self) -> tuple[str, ...]:
        """Lê os rótulos gravados pelo treino, com o catálogo como fallback."""
        path = self.labels_path
        fallback = tuple(self.spec.class_names)

        if not path.exists():
            logger.warning(
                "[%s] rótulos não encontrados em %s — assumindo a ordem do catálogo (%s). "
                "Confirme que ela bate com a ordem usada no treino.",
                self.key,
                path,
                ", ".join(fallback),
            )
            return fallback

        try:
            labels = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "[%s] rótulos inválidos em %s (%s) — usando o catálogo.", self.key, path, exc
            )
            return fallback

        if not _is_label_list(labels):
            logger.error(
                "[%s] %s deveria ser uma lista de strings — usando o catálogo.", self.key, path
            )
            return fallback

        return tuple(labels)


def _is_label_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) for item in value)


def _is_novel(score: float | None) -> bool:
    """O limiar já está embutido no score (1.0); a sensibilidade o desloca."""
    if score is None:
        return False
    return score > settings.novelty_sensitivity


def _build_forward(keras_model: Any) -> Any:
    """Monta o callable ``lote -> (probabilidades, embeddings | None)``.

    Quando o modelo expõe a camada de embedding, devolve um modelo Keras de duas
    saídas: assim a assinatura visual sai de graça, na mesma passada das
    probabilidades. Modelos antigos, sem a camada nomeada, seguem funcionando —
    só não pontuam novidade.
    """
    dual_forward = getattr(keras_model, "dual_forward", None)
    if callable(dual_forward):  # ONNX Runtime: o grafo já tem as duas saídas
        return dual_forward

    embed = getattr(keras_model, "embed", None)
    if callable(embed):  # dublê dos testes
        return lambda batch: (keras_model(batch, training=False), embed(batch))

    def run(model: Any, batch: Any) -> Any:
        with no_grad():  # ver keras_backend.no_grad: sem isto o torch grava autograd
            return model(batch, training=False)

    try:
        embedding_output = keras_model.get_layer(EMBEDDING_LAYER).output
        dual = keras_model.__class__(
            _single_input(keras_model), [keras_model.output, embedding_output]
        )
    except (AttributeError, ValueError, TypeError) as exc:
        logger.warning(
            "Camada %r não encontrada (%s) — detecção de novidade indisponível "
            "para este modelo. Retreine para habilitá-la.",
            EMBEDDING_LAYER,
            exc,
        )
        return lambda batch: (run(keras_model, batch), None)

    return lambda batch: tuple(run(dual, batch))


def _single_input(keras_model: Any) -> Any:
    """Devolve a entrada do modelo desembrulhada, quando há só uma.

    ``model.inputs`` é sempre uma lista. Construir o modelo de duas saídas a
    partir dela faria o Keras esperar uma lista também na chamada, e como o
    lote chega como um tensor só, cada previsão emitiria um aviso de estrutura
    incompatível — ruído em log de produção, a cada requisição.
    """
    inputs = getattr(keras_model, "inputs", None)
    if isinstance(inputs, list | tuple) and len(inputs) == 1:
        return inputs[0]
    return inputs


def _square_input_size(keras_model: Any) -> int | None:
    """Extrai o lado da entrada quadrada declarada pelo modelo, se houver."""
    shape = getattr(keras_model, "input_shape", None)
    if not isinstance(shape, tuple) or len(shape) != 4:
        return None
    height, width = shape[1], shape[2]
    if height is None or width is None or height != width:
        return None
    return int(height)


def _output_units(keras_model: Any) -> int | None:
    """Extrai o número de saídas do modelo, se declarado."""
    shape = getattr(keras_model, "output_shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[1] is None:
        return None
    return int(shape[1])
