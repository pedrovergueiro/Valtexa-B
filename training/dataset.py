"""Carregamento e augmentation do dataset de treino."""

from __future__ import annotations

from pathlib import Path

from app.keras_backend import configure, disable_tensorflow_gpu

configure()

import keras  # noqa: E402  (o backend precisa ser escolhido antes deste import)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from app.inference.preprocess import scale_to_unit_range  # noqa: E402
from training import DEFAULT_SEED  # noqa: E402

# O TensorFlow aqui é só o carregador de dados; quem usa a GPU é o backend.
disable_tensorflow_gpu()

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def index_by_class(data_dir: Path) -> tuple[list[str], list[int], list[str]]:
    """Lista as imagens de ``data_dir``, com o índice da classe de cada uma.

    A ordem é determinística — classes e arquivos ordenados alfabeticamente —,
    porque é ela que o sorteio do *split* embaralha depois. Sem essa garantia,
    a mesma ``seed`` daria conjuntos diferentes em máquinas diferentes, e dois
    modelos deixariam de ser comparáveis.

    Os nomes das classes saem na mesma ordem alfabética que o Keras usa, que é
    o contrato gravado no ``.labels.json`` e lido pela API.
    """
    from training.prepare_dataset import is_image

    class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    paths: list[str] = []
    labels: list[int] = []

    for index, name in enumerate(class_names):
        for image in sorted((data_dir / name).iterdir()):
            if is_image(image):
                paths.append(str(image))
                labels.append(index)

    return paths, labels, class_names


SHUFFLE_BUDGET_BYTES = 256 * 1024 * 1024
"""Teto de memória do buffer de embaralhamento do treino.

O buffer guarda imagens **já decodificadas**: a 224x224x3 em ``float32`` cada
uma pesa 602 KB, então um buffer de 4.096 imagens — número que parece modesto —
pede 2,5 GB só para si. Foi exatamente assim que o treino foi morto pelo
*OOM killer* numa máquina com 7,7 GB.

O buffer pode ser pequeno sem prejuízo porque a lista de arquivos **já foi
permutada globalmente** antes de virar dataset: o que está em disco não está
agrupado por classe, então mesmo uma janela curta produz lotes bem misturados.
O embaralhamento por época serve só para variar a composição dos lotes.
"""


def _shuffle_buffer(total: int, img_size: int) -> int:
    """Quantas imagens cabem no orçamento de memória do buffer."""
    bytes_por_imagem = img_size * img_size * 3 * 4  # float32, 3 canais
    return max(64, min(total, SHUFFLE_BUDGET_BYTES // bytes_por_imagem))


def split_indices(
    total: int, seed: int, validation_split: float, test_split: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorteia treino, validação e teste sobre a lista de arquivos.

    O sorteio acontece aqui, e não dentro do ``tf.data``: o
    ``image_dataset_from_directory`` reembaralha a cada iteração, então cortar
    o resultado com ``take``/``skip`` devolveria imagens diferentes a cada
    passada — o "conjunto de teste" mudaria sozinho e deixaria de ser isolado.
    Uma permutação semeada dá exatamente o mesmo corte em qualquer processo e em
    qualquer dia, que é o que permite comparar dois modelos de forma justa.

    Raises:
        ValueError: se alguma fração for negativa, ou se as duas juntas não
            deixarem dado para treinar.
    """
    # A checagem de negativo vem antes da soma, e não é redundante: -0,1 e 0,2
    # somam 0,1 e passariam pelo teste do intervalo, mas um ``n_val`` negativo
    # faz o fatiamento devolver treino e teste **sobrepostos** — o vazamento
    # silencioso que a separação toda existe para impedir.
    if validation_split < 0.0 or test_split < 0.0:
        raise ValueError(
            f"Frações não podem ser negativas (validation={validation_split}, test={test_split})."
        )

    holdout = validation_split + test_split
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"validation_split + test_split precisa ficar entre 0 e 1 (é {holdout}).")

    order = np.random.default_rng(seed).permutation(total)
    n_test = int(total * test_split)
    n_val = int(total * validation_split)

    return order[n_test + n_val :], order[n_test : n_test + n_val], order[:n_test]


def _decode(path, img_size: int):
    """Lê e redimensiona uma imagem exatamente como o Keras faria.

    Bilinear, sem preservar o aspect ratio e sem antialias — igual ao
    ``image_dataset_from_directory`` e, por consequência, igual ao
    :func:`app.inference.preprocess.image_to_array` que a API usa. Divergir aqui
    não geraria erro, só derrubaria a acurácia em produção.
    """
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False, dtype=tf.uint8
    )
    image = tf.image.resize(tf.cast(image, tf.float32), (img_size, img_size), method="bilinear")
    image.set_shape((img_size, img_size, 3))
    return image


def build_augmentation() -> list[keras.Layer]:
    """Augmentations leves, coerentes com fotos aéreas de folhas.

    Só transformações que preservam o diagnóstico: espelhar, girar e aproximar
    um pouco não mudam a doença presente na folha. Alterações fortes de cor
    seriam contraproducentes, já que a cor é justamente o sinal que separa
    ferrugem, mancha e folha saudável.

    Devolve as camadas soltas, e não um ``keras.Sequential``, porque elas são
    aplicadas dentro do ``tf.data``. Individualmente cada uma detecta o contexto
    do ``tf.data`` e emite operações do TensorFlow, o que funciona em qualquer
    backend; o ``Sequential`` faria a chamada passar pelo backend de compute e
    quebraria com os tensores simbólicos do ``tf.data``.
    """
    return [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1),
    ]


def load_labeled_dataset(
    data_dir: Path, img_size: int, batch_size: int
) -> tuple[tf.data.Dataset, list[str]]:
    """Carrega o diretório inteiro, sem split e **sem augmentation**.

    É o que a calibração de novidade precisa: a referência tem de descrever as
    imagens como elas realmente são, não versões giradas e distorcidas delas.
    """
    dataset = keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    class_names = list(dataset.class_names)
    prepared = dataset.map(
        lambda images, labels: (scale_to_unit_range(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return prepared.prefetch(tf.data.AUTOTUNE), class_names


def load_datasets(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    validation_split: float,
    seed: int = DEFAULT_SEED,
    test_split: float = 0.0,
    cache_dir: Path | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset | None, list[str]]:
    """Monta os datasets de treino, validação e teste a partir de pastas por classe.

    O redimensionamento é bilinear e não preserva o aspect ratio, a mesma
    semântica de :func:`app.inference.preprocess.image_to_array`. A normalização
    é literalmente a mesma função nas duas pontas. O redimensionamento, não: aqui
    é ``tf.image``, na API é o Pillow, e os núcleos de reamostragem diferem — ver
    "Paridade entre treino e inferência" no README para a folga medida.

    Args:
        test_split: fração reservada para o conjunto de teste, separada da
            validação. Zero devolve ``None`` no lugar dele.
        cache_dir: onde guardar as imagens já decodificadas. Acelera muito as
            épocas seguintes; ``None`` desliga o cache.

    Por que existe um conjunto de teste separado: a melhor época é escolhida
    pela ``val_accuracy``, então esse número já viu o conjunto de validação e
    é otimista por construção — ele mede o melhor sorteio, não o desempenho.
    O teste nunca participa de nenhuma decisão, e é o único número que dá para
    levar para um cliente.

    Returns:
        Treino, validação, teste (ou ``None``) e os nomes das classes na ordem
        em que o Keras as indexou (ordem alfabética das pastas).
    """
    paths, labels, class_names = index_by_class(data_dir)
    if not paths:
        raise ValueError(f"Nenhuma imagem encontrada em {data_dir}.")

    train_idx, val_idx, test_idx = split_indices(len(paths), seed, validation_split, test_split)

    def subset(indices) -> tf.data.Dataset:
        """Dataset decodificado e **sem lote** — quem agrupa é o ``finish``."""
        selected = [paths[i] for i in indices]
        selected_labels = [labels[i] for i in indices]
        dataset = tf.data.Dataset.from_tensor_slices((selected, selected_labels))
        return dataset.map(
            lambda path, label: (_decode(path, img_size), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = subset(train_idx)
    holdout_ds = subset(val_idx)
    test_ds = subset(test_idx) if len(test_idx) else None

    print(f"Encontradas {len(paths)} imagens em {len(class_names)} classes.")
    print(f"  treino: {len(train_idx)}   validação: {len(val_idx)}   teste: {len(test_idx)}")

    augmentation = build_augmentation()

    def augment(images, labels):
        for layer in augmentation:
            images = layer(images, training=True)
        return images, labels

    def normalize(images, labels):
        return scale_to_unit_range(images), labels

    def finish(dataset: tf.data.Dataset, name: str, *, training: bool) -> tf.data.Dataset:
        # A ordem das etapas aqui não é arbitrária:
        #
        # 1. o cache guarda a imagem **decodificada e redimensionada**, que é o
        #    gargalo (a GPU fica ociosa esperando a CPU abrir JPEG) e não muda
        #    entre épocas;
        # 2. o embaralhamento vem depois do cache e antes do lote, para sortear
        #    imagens e não blocos já formados;
        # 3. a augmentation vem por último porque precisa sortear de novo a cada
        #    passada — cacheá-la congelaria um único conjunto de imagens giradas,
        #    anulando o efeito dela.
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            dataset = dataset.cache(str(cache_dir / name))
        if training:
            dataset = dataset.shuffle(_shuffle_buffer(len(train_idx), img_size), seed=seed)

        dataset = dataset.batch(batch_size)
        if training:
            dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        # Prefetch limitado, e não AUTOTUNE: cada lote já decodificado ocupa
        # ``batch_size * img_size^2 * 3 * 4`` bytes, e o AUTOTUNE se dá o direito
        # de empilhar vários deles quando a GPU está lenta — exatamente quando a
        # memória já está apertada.
        return dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)

    return (
        finish(train_ds, "train", training=True),
        finish(holdout_ds, "val", training=False),
        finish(test_ds, "test", training=False) if test_ds is not None else None,
        class_names,
    )
