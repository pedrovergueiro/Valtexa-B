"""Escolha do backend do Keras, e o isolamento de GPU que ela exige.

O Keras 3 é agnóstico de backend: a mesma arquitetura, o mesmo ``.keras`` e a
mesma API de treino rodam sobre TensorFlow, PyTorch ou JAX. Quem faz a conta
com a GPU é o backend; o TensorFlow permanece no projeto apenas como carregador
de dados (``tf.data``), que roda em CPU.

O padrão é **torch** porque o TensorFlow não publica kernels CUDA para as GPUs
Blackwell (compute capability 12.0, ex.: RTX 50xx). Nessas placas ele enxerga o
dispositivo e falha na primeira operação — inclusive num ``Cast`` — com
``CUDA_ERROR_INVALID_PTX``, porque o *fallback* de compilar PTX em tempo de
execução também não funciona. O torch publica ``sm_120`` desde o cu128.

Para voltar ao comportamento anterior, exporte ``KERAS_BACKEND=tensorflow``
antes de rodar qualquer coisa; nada mais no código precisa mudar.

A escolha precisa acontecer **antes** do primeiro ``import keras``, então todo
módulo que importa o Keras chama :func:`configure` primeiro.
"""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np

DEFAULT_BACKEND = "torch"

logger = logging.getLogger(__name__)


def to_numpy(values: Any) -> np.ndarray:
    """Traz a saída de um modelo Keras para ``numpy``, venha de onde vier.

    O backend torch devolve tensores que vivem na GPU, e o ``numpy`` se recusa a
    lê-los diretamente. TensorFlow, JAX e o dublê usado nos testes já devolvem
    algo que o ``np.asarray`` entende, então o caminho rápido cobre esses casos
    sem custo e sem importar o backend aqui.
    """
    if isinstance(values, np.ndarray):
        return values
    detach = getattr(values, "detach", None)
    if callable(detach):  # tensor do torch, possivelmente na GPU
        return detach().cpu().numpy()
    return np.asarray(values)


def no_grad() -> AbstractContextManager:
    """Contexto de inferência: sem grafo de gradientes.

    Chamar um modelo Keras direto — ``model(lote)``, como a API faz para fugir do
    overhead do ``predict()`` em lotes pequenos — **não** desliga o autograd no
    backend torch. Medido em 26/09/2026: a saída vinha com ``requires_grad`` e
    ``SoftmaxBackward0``, ou seja, cada requisição montava e segurava o grafo de
    gradientes de uma rede inteira, só para jogá-lo fora. Na CPU, o mesmo lote
    ficou 28% mais rápido sem isso; na GPU, a memória retida é o que enche a placa.

    Nos outros backends, e sem torch instalado, não faz nada.
    """
    if configure() != "torch":
        return nullcontext()
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.no_grad()


def configure() -> str:
    """Fixa o backend do Keras e devolve o que ficou valendo.

    Respeita um ``KERAS_BACKEND`` já definido no ambiente — é assim que se
    escolhe outro backend sem tocar no código.
    """
    return os.environ.setdefault("KERAS_BACKEND", DEFAULT_BACKEND)


def disable_tensorflow_gpu() -> None:
    """Tira a GPU do TensorFlow quando ele não é o backend de compute.

    Sem isto os dois backends disputam a mesma placa: o TensorFlow reserva
    memória e passa a executar as operações auxiliares do ``tf.data`` na GPU —
    exatamente o que quebra nas placas sem kernels compatíveis. Como aqui ele só
    decodifica e redimensiona imagens, perder a GPU não lhe custa nada.

    Silenciosa quando o TensorFlow não está instalado: a API de inferência não
    depende dele.
    """
    if configure() == "tensorflow":
        return

    try:
        import tensorflow as tf
    except ImportError:
        return

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        # Já havia contexto de GPU inicializado — só acontece se algum import
        # anterior tocou o TensorFlow antes desta chamada.
        logger.warning(
            "TensorFlow já havia inicializado a GPU; não foi possível escondê-la. "
            "Importe app.keras_backend antes de qualquer uso do TensorFlow."
        )
