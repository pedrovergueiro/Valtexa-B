"""Remove imagens repetidas de um dataset já consolidado.

Rode **depois** do ``prepare_dataset`` e **antes** do ``train``::

    python -m training.dedupe_dataset --model leaf-foliar --dry-run

Por que isso importa: o ``train`` separa treino e validação com um sorteio sobre
a lista de arquivos. Se a mesma imagem existe duas vezes, ela tem chance de cair
dos dois lados — o modelo é avaliado em algo que decorou, e a ``val_accuracy``
sobe sem que nada tenha melhorado em campo. É um erro que não gera aviso nenhum:
só um número bonito e um piloto decepcionante.

Os datasets públicos repetem imagem o tempo todo. Três casos reais no catálogo:

* o PlantVillage publica ``color/``, ``grayscale/`` e ``segmented/`` — a mesma
  folha três vezes;
* o ``maize-diseases`` do Kaggle traz ``v1.0`` e ``v1.1``, e a v1.0 é quase toda
  subconjunto da v1.1;
* o ``corn-leaf-disease`` parece ser vídeo amostrado, com quadros a dois
  segundos um do outro — que não são idênticos, mas são a mesma planta.

Por isso a comparação é por *assinatura perceptual* (dHash), e não por hash do
arquivo: o dHash sobrevive a recompressão, redimensionamento e mudança de
formato, que é justamente como as cópias costumam aparecer.

O dHash, porém, só **indica** o par; quem confirma é a miniatura colorida (ver
:data:`MAX_THUMB_DIFFERENCE`). Sem essa confirmação, folhas diferentes
fotografadas sobre papel liso viram "a mesma imagem".
"""

from __future__ import annotations

import argparse
import functools
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.registry import MODEL_SPECS, get_spec
from training.background import MAX_BORDER_SPREAD, otsu_threshold, paper_color
from training.prepare_dataset import is_image

HASH_SIZE = 8
"""Lado da grade do dHash. 8 gera uma assinatura de 64 bits."""

DEFAULT_MAX_DISTANCE = 4
"""Distância de Hamming até onde duas imagens são consideradas a mesma.

Zero pega só cópia exata (mesma imagem recomprimida). Quatro pega também o
quadro seguinte de um vídeo e o mesmo enquadramento com leve variação de luz.
Acima de oito o filtro começa a apagar folhas genuinamente diferentes da mesma
classe, que é dado bom sendo jogado fora.
"""

THUMB_SIZE = 16
"""Lado da miniatura colorida que confirma cada par indicado pelo dHash."""

MAX_THUMB_DIFFERENCE = 8.0
"""Diferença média máxima (0-255, por pixel e canal) entre as miniaturas de duas cópias.

O dHash olha só o gradiente de uma miniatura 9x8 em tons de cinza — e toda
folha destacada sobre papel liso tem o mesmo gradiente: claro, escuro, claro.
No citros (limão e laranja fotografados sobre papel), a distância 4 com união
transitiva juntou as 12 classes num grupo só e ia apagar 68% do dataset; mesmo
a distância 0 juntava folhas diferentes de classes diferentes.

A confirmação compara a miniatura colorida **só da folha** (ver
:func:`leaf_box`): com o quadro inteiro, uma folha pequena sobre papel cinza
ocupa 5% da miniatura e o papel decide sozinho — folhas diferentes empatavam e
se encadeavam em grupos de centenas. Medido em 6.000 pares que o dHash indicou
no citros: entre classes, **nenhum** par fica até 10 (mediana 39); dentro da
classe, 9,1% ficam até 6 e 9,3% até 10 — um vão limpo entre a mesma folha
fotografada em sequência e folhas diferentes.
"""

BAND_COUNT = 4
"""Em quantos pedaços a assinatura é cortada para indexar os candidatos.

Comparar todas as imagens duas a duas seria quadrático — inviável em dezenas de
milhares. Duas assinaturas a até 4 bits de distância obrigatoriamente coincidem
em pelo menos um dos 4 pedaços de 16 bits, então basta comparar quem divide
algum pedaço. É o mesmo princípio do LSH por bandas.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove imagens repetidas de data/<modelo>/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="leaf-foliar")
    parser.add_argument("--data-dir", type=Path, default=None, help="Padrão: data/<modelo>/.")
    parser.add_argument(
        "--max-distance",
        type=int,
        default=DEFAULT_MAX_DISTANCE,
        help="Distância de Hamming máxima entre duas cópias.",
    )
    parser.add_argument(
        "--max-thumb-difference",
        type=float,
        default=MAX_THUMB_DIFFERENCE,
        help="Diferença média máxima entre as miniaturas coloridas de duas cópias.",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Aceita o par só pelo dHash, sem conferir a miniatura (comportamento antigo).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Só relata o que seria apagado.")
    return parser.parse_args()


def dhash(path: Path, size: int = HASH_SIZE) -> np.uint64 | None:
    """Assinatura perceptual de uma imagem, ou ``None`` se ela não abrir.

    O dHash compara cada pixel com o vizinho da direita numa miniatura em tons
    de cinza: o que sobrevive é o gradiente da imagem, não os valores absolutos.
    Isso o torna estável a recompressão e a redimensionamento — as duas formas
    mais comuns de a mesma foto reaparecer em outro dataset.
    """
    try:
        with Image.open(path) as image:
            small = image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
            pixels = np.asarray(small, dtype=np.int16)
    except (UnidentifiedImageError, OSError, ValueError):
        return None

    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return np.packbits(bits).view(">u8")[0]


def leaf_box(image: Image.Image, work: int = 128) -> tuple[int, int, int, int] | None:
    """Caixa do que não é papel, ou ``None`` se o fundo não for liso.

    Parecida com :func:`training.background.leaf_mask`, mas sem o tamanho
    mínimo de folha: aqui a folha minúscula no meio de uma folha A4 é
    justamente o caso que precisa da caixa.
    """
    small = np.asarray(image.resize((work, work), Image.Resampling.BILINEAR), dtype=np.float32)
    color, spread = paper_color(small)
    if spread > MAX_BORDER_SPREAD:
        return None
    distance = np.sqrt(((small - color) ** 2).sum(axis=2))
    foreground = distance > max(otsu_threshold(distance), 12.0)
    if foreground.mean() < 0.002:
        return None
    ys, xs = np.nonzero(foreground)
    # Percentis e não min/max: um cisco no papel não estica a caixa.
    x0, x1 = np.percentile(xs, [0.5, 99.5])
    y0, y1 = np.percentile(ys, [0.5, 99.5])
    sx, sy = image.width / work, image.height / work
    return int(x0 * sx), int(y0 * sy), int((x1 + 1) * sx), int((y1 + 1) * sy)


def thumbnail(path: Path, size: int = THUMB_SIZE) -> np.ndarray | None:
    """Miniatura colorida ``(size, size, 3)`` da folha, ou ``None`` se não abrir.

    Com fundo liso, a miniatura é só da caixa da folha; sem ele (foto de
    campo), do quadro inteiro.
    """
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    box = leaf_box(image)
    if box is not None:
        image = image.crop(box)
    return np.asarray(image.resize((size, size), Image.Resampling.BOX), dtype=np.float32)


def thumbnail_confirmation(max_difference: float) -> Callable[[Path, Path], bool]:
    """Confirma um par se as miniaturas coloridas diferem até ``max_difference``.

    Cada miniatura é calculada uma vez só, e só para as imagens que o dHash
    indicou — a grande maioria do dataset nunca chega a ser relida.
    """

    @functools.cache
    def cached(path: Path) -> np.ndarray | None:
        return thumbnail(path)

    def same(a: Path, b: Path) -> bool:
        first, second = cached(a), cached(b)
        if first is None or second is None:
            return False
        return float(np.abs(first - second).mean()) <= max_difference

    return same


def _bands(signature: np.uint64) -> list[int]:
    """Corta a assinatura de 64 bits nos pedaços usados como índice."""
    width = 64 // BAND_COUNT
    value = int(signature)
    return [(value >> (index * width)) & ((1 << width) - 1) for index in range(BAND_COUNT)]


def _distance(a: np.uint64, b: np.uint64) -> int:
    return int(bin(int(a) ^ int(b)).count("1"))


def collect_signatures(data_dir: Path) -> tuple[dict[Path, np.uint64], list[Path]]:
    """Calcula a assinatura de cada imagem, por classe."""
    signatures: dict[Path, np.uint64] = {}
    unreadable: list[Path] = []

    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for image in sorted(class_dir.iterdir()):
            if not is_image(image):
                continue
            signature = dhash(image)
            if signature is None:
                unreadable.append(image)
                continue
            signatures[image] = signature
        print(f"  {class_dir.name}: {sum(1 for p in signatures if p.parent == class_dir)} lidas")

    return signatures, unreadable


class _UnionFind:
    """União-busca sobre caminhos de arquivo.

    A duplicidade é transitiva: se o quadro B se parece com o A e com o C, os
    três são a mesma cena, mesmo que A e C sozinhos não se pareçam. Sem unir os
    pares em grupos, um vídeo amostrado viraria dezenas de pares soltos em vez
    de um grupo só.
    """

    def __init__(self, items) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: Path) -> Path:
        # Compressão de caminho: cada busca encurta a árvore que percorreu.
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: Path, b: Path) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def groups(self) -> list[list[Path]]:
        """Os conjuntos com mais de um membro, cada um ordenado."""
        found: dict[Path, list[Path]] = defaultdict(list)
        for item in self._parent:
            found[self.find(item)].append(item)
        return [sorted(members) for members in found.values() if len(members) > 1]


def _candidate_buckets(signatures: dict[Path, np.uint64]) -> dict[tuple[int, int], list[Path]]:
    """Indexa as imagens pelas bandas da assinatura.

    Duas assinaturas a até ``max_distance`` bits de distância coincidem
    obrigatoriamente em pelo menos uma banda, então basta comparar quem divide
    alguma — o que troca uma varredura quadrática por uma busca local.
    """
    buckets: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path, signature in signatures.items():
        for index, band in enumerate(_bands(signature)):
            buckets[(index, band)].append(path)
    return buckets


def group_duplicates(
    signatures: dict[Path, np.uint64],
    max_distance: int,
    confirm: Callable[[Path, Path], bool] | None = None,
) -> list[list[Path]]:
    """Agrupa as imagens que são a mesma coisa.

    Args:
        confirm: segunda opinião sobre cada par que o dHash aproximou; o par só
            é unido se ela concordar. ``None`` aceita o dHash sozinho.
    """
    grupos = _UnionFind(signatures)

    for candidates in _candidate_buckets(signatures).values():
        if len(candidates) < 2:
            continue
        for i, a in enumerate(candidates):
            for b in candidates[i + 1 :]:
                if (
                    grupos.find(a) != grupos.find(b)
                    and _distance(signatures[a], signatures[b]) <= max_distance
                    and (confirm is None or confirm(a, b))
                ):
                    grupos.union(a, b)

    return grupos.groups()


def main() -> None:
    args = parse_args()
    spec = get_spec(args.model)
    data_dir = args.data_dir or spec.data_dir

    if not data_dir.exists():
        raise SystemExit(f"Diretório de dados não encontrado: {data_dir}")

    print(f"\n=== Deduplicando {spec.key} ===")
    print(f"Dados: {data_dir}\n")

    signatures, unreadable = collect_signatures(data_dir)
    print(f"\n{len(signatures)} imagens com assinatura calculada.")
    if unreadable:
        print(f"{len(unreadable)} arquivos ilegíveis (serão removidos):")
        for path in unreadable[:5]:
            print(f"    {path.name}")

    confirm = None if args.no_confirm else thumbnail_confirmation(args.max_thumb_difference)
    groups = group_duplicates(signatures, args.max_distance, confirm)

    # Um grupo espalhado por classes diferentes é a mesma imagem com dois
    # rótulos: um dos dois está errado e não dá para saber qual. Manter a imagem
    # ensinaria o modelo a errar com convicção, então o grupo inteiro sai.
    conflicting = [g for g in groups if len({p.parent.name for p in g}) > 1]
    redundant = [g for g in groups if len({p.parent.name for p in g}) == 1]

    to_remove = list(unreadable)
    for group in redundant:
        to_remove.extend(group[1:])  # mantém uma cópia
    for group in conflicting:
        to_remove.extend(group)  # descarta todas

    print(
        f"\n{len(redundant)} grupos de cópias dentro da mesma classe "
        f"-> {sum(len(g) - 1 for g in redundant)} arquivos sobrando"
    )
    print(
        f"{len(conflicting)} grupos com a mesma imagem em classes diferentes "
        f"-> {sum(len(g) for g in conflicting)} arquivos descartados"
    )

    for group in conflicting[:5]:
        classes = sorted({p.parent.name for p in group})
        print(f"    conflito: {' vs '.join(classes)}  ({group[0].name})")

    remaining = len(signatures) - len([p for p in to_remove if p in signatures])
    print(
        f"\nDe {len(signatures)} imagens, sobram {remaining} "
        f"({100 * remaining / max(len(signatures), 1):.1f}%)."
    )

    if args.dry_run:
        print("\n--dry-run: nada foi apagado.")
        return

    for path in to_remove:
        path.unlink(missing_ok=True)
    print(f"\n{len(to_remove)} arquivos removidos.")

    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        print(f"    {sum(1 for i in class_dir.iterdir() if is_image(i)):>6}  {class_dir.name}")


if __name__ == "__main__":
    main()
