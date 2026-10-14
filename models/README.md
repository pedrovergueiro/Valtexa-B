# Modelos treinados do VALTEXA-B

92 arquivos, 1052 MB, guardados com **Git LFS**. Exportados em 14/10/2026 de
`/home/pedro/valtexa/models` (o ambiente de treino no WSL), só os de produção e os experimentos
documentados — backups, candidatos antigos e a quarentena ficaram de fora.

## Como baixar

```bash
git lfs install            # uma vez por máquina
git clone https://github.com/pedrovergueiro/Valtexa-B.git
cd Valtexa-B && git lfs pull --include="models/*"
sha256sum -c models/SHA256SUMS   # confere a integridade (Linux/WSL)
```

O serviço (`app/`) lê desta pasta (`MODELS_DIR`, padrão `models/`). Fichas técnicas, datasets, métricas
e ressalvas de cada modelo: [`MODELS.md`](../MODELS.md). Os números de acurácia só valem com as ressalvas
de lá — em especial, a acurácia do café em campo ainda **não foi medida** (falta coleta).

**Licença dos dados:** as variantes `_cc` foram treinadas só com dados CC BY; as demais usam datasets
com restrição de uso comercial (ver `MODELS.md`). O `framing_vision.onnx` é a torre de visão do SigLIP 2
(Google, Apache 2.0) exportada.

## Arquivos

### `coffee_foliar`

**Café** (foco do produto) — diagnóstico foliar, 5 classes. Keras + ONNX (Rover); `.head.npz` = cabeça sobre o embedding SigLIP 2; `.edge*.json` = perfil/limiares da Edge AI e fusão.

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `coffee_foliar.categories.json` | categorias | 0 KB | `31405eb90a9d4b50…` |
| `coffee_foliar.conformal-a05.json` | conformal α=0,05 | 1 KB | `4ed996da34897ec8…` |
| `coffee_foliar.conformal-a10.json` | conformal α=0,10 | 1 KB | `10d7493a1c1c7d9f…` |
| `coffee_foliar.conformal.json` | conformal (α padrão) | 1 KB | `b8beb38d1a502262…` |
| `coffee_foliar.edge-fusion.json` | fusão na borda | 5 KB | `9d90aa377c11755d…` |
| `coffee_foliar.edge.json` | perfil da Edge AI | 4 KB | `38b090de8ac6f1e3…` |
| `coffee_foliar.head.npz` | cabeça sobre o embedding | 7.7 MB | `ee9fdf43b5a0c0d9…` |
| `coffee_foliar.keras` | pesos Keras | 23.9 MB | `541d6ce67545eb00…` |
| `coffee_foliar.labels.json` | rótulos | 0 KB | `981c0ffb1adc2c0d…` |
| `coffee_foliar.novelty.npz` | detector de novidade | 324 KB | `374a2392300ea76f…` |
| `coffee_foliar.onnx` | ONNX (Edge/Rover) | 9.2 MB | `2d0f9ce87f3bd362…` |
| `coffee_foliar.quality.json` | qualidade do dataset/medidas | 0 KB | `9b10661dc8338e5f…` |

### `framing_vision`

Torre de visão do SigLIP 2 em ONNX (portão de enquadramento e embedding das cabeças), usada no Rover.

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `framing_vision.onnx` | ONNX (Edge/Rover) | 354.5 MB | `1a2360fc19f12ecd…` |

### `framing`

Embeddings de referência do portão de enquadramento (`framing.npz`).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `framing.npz` | dados | 142 KB | `311f42fbaa186c89…` |

### `banana_foliar`

Banana — diagnóstico foliar, 5 classes (rover).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `banana_foliar.categories.json` | categorias | 0 KB | `355357831db5dc38…` |
| `banana_foliar.conformal.json` | conformal (α padrão) | 1 KB | `723e94dbc09289cf…` |
| `banana_foliar.edge-fusion.json` | fusão na borda | 4 KB | `c138fd325eac3b4e…` |
| `banana_foliar.edge.json` | perfil da Edge AI | 4 KB | `5407c3701be5a4fb…` |
| `banana_foliar.fusion.json` | fusão calibrada | 4 KB | `dbdc371b530ba411…` |
| `banana_foliar.keras` | pesos Keras | 23.9 MB | `07ba0860cac0a8e1…` |
| `banana_foliar.labels.json` | rótulos | 0 KB | `e45671ea7f50f3a1…` |
| `banana_foliar.novelty.npz` | detector de novidade | 315 KB | `c1101cfd9cab1847…` |
| `banana_foliar.onnx` | ONNX (Edge/Rover) | 9.2 MB | `b673597b93b723cf…` |
| `banana_foliar.prior.json` | prior de classes | 0 KB | `31fe233a9b4c5075…` |
| `banana_foliar.quality.json` | qualidade do dataset/medidas | 0 KB | `b408dbef59be30cb…` |

### `banana_foliar_anterior`

Banana — versão anterior (referência).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `banana_foliar_anterior.keras` | pesos Keras | 23.9 MB | `1f92de7eab19965c…` |
| `banana_foliar_anterior.labels.json` | rótulos | 0 KB | `e45671ea7f50f3a1…` |

### `banana_foliar_b0`

Banana — variante B0 (experimento).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `banana_foliar_b0.keras` | pesos Keras | 70.3 MB | `6a5ed2173bb59c18…` |
| `banana_foliar_b0.labels.json` | rótulos | 0 KB | `e45671ea7f50f3a1…` |

### `banana_foliar_campo`

Banana — variante treinada com recortes de campo (experimento).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `banana_foliar_campo.keras` | pesos Keras | 23.9 MB | `07ba0860cac0a8e1…` |
| `banana_foliar_campo.labels.json` | rótulos | 0 KB | `e45671ea7f50f3a1…` |

### `banana_foliar_cc`

Banana — versão comercial (só dados CC BY).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `banana_foliar_cc.conformal.json` | conformal (α padrão) | 1 KB | `064a134ced278112…` |
| `banana_foliar_cc.keras` | pesos Keras | 23.9 MB | `1f92de7eab19965c…` |
| `banana_foliar_cc.labels.json` | rótulos | 0 KB | `e45671ea7f50f3a1…` |
| `banana_foliar_cc.novelty.npz` | detector de novidade | 354 KB | `36cee2374d455e5e…` |
| `banana_foliar_cc.onnx` | ONNX (Edge/Rover) | 9.2 MB | `2d0c35b7c9060441…` |
| `banana_foliar_cc.prior.json` | prior de classes | 0 KB | `5c2015eb15b852b4…` |
| `banana_foliar_cc.quality.json` | qualidade do dataset/medidas | 0 KB | `14f8c17059f231b2…` |

### `citrus_foliar`

Citros — diagnóstico foliar, 12 classes (rover); `.species.npz` = verificação de espécie (desligada por padrão).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `citrus_foliar.categories.json` | categorias | 0 KB | `83627a86a0d54270…` |
| `citrus_foliar.conformal.json` | conformal (α padrão) | 1 KB | `93526e8d2367523e…` |
| `citrus_foliar.edge-fusion.json` | fusão na borda | 6 KB | `0b316e4edd4f1ce2…` |
| `citrus_foliar.edge.json` | perfil da Edge AI | 6 KB | `e44b2eca54e100c3…` |
| `citrus_foliar.fusion.json` | fusão calibrada | 6 KB | `2179f3f87ecd3fb8…` |
| `citrus_foliar.head.npz` | cabeça sobre o embedding | 6.7 MB | `469f6808e29c766e…` |
| `citrus_foliar.keras` | pesos Keras | 23.9 MB | `929ec89de4e0520e…` |
| `citrus_foliar.labels.json` | rótulos | 0 KB | `a65b9919ff566964…` |
| `citrus_foliar.novelty.npz` | detector de novidade | 568 KB | `c492bc9ca0eb0996…` |
| `citrus_foliar.onnx` | ONNX (Edge/Rover) | 9.3 MB | `17e696f807ba113c…` |
| `citrus_foliar.prior.json` | prior de classes | 1 KB | `a8a72c34fed066d2…` |
| `citrus_foliar.quality.json` | qualidade do dataset/medidas | 0 KB | `0e61b79864da807c…` |
| `citrus_foliar.species.npz` | verificação de espécie | 8.1 MB | `8f2f03d8843e39ad…` |

### `citrus_foliar_b0`

Citros — variante B0 (experimento).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `citrus_foliar_b0.keras` | pesos Keras | 25.4 MB | `2e64d07e0002030e…` |
| `citrus_foliar_b0.labels.json` | rótulos | 0 KB | `a65b9919ff566964…` |

### `citrus_foliar_campo`

Citros — variante de campo (experimento).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `citrus_foliar_campo.keras` | pesos Keras | 23.9 MB | `2c53958a5a02758e…` |
| `citrus_foliar_campo.labels.json` | rótulos | 0 KB | `a65b9919ff566964…` |

### `citrus_foliar_campo_sem_fundo`

Citros — variante de campo sem fundo (experimento).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `citrus_foliar_campo_sem_fundo.keras` | pesos Keras | 23.9 MB | `929ec89de4e0520e…` |
| `citrus_foliar_campo_sem_fundo.labels.json` | rótulos | 0 KB | `a65b9919ff566964…` |

### `citrus_foliar_cc`

Citros — versão comercial (só dados CC BY).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `citrus_foliar_cc.conformal.json` | conformal (α padrão) | 2 KB | `1f8010da9c7f5232…` |
| `citrus_foliar_cc.head.npz` | cabeça sobre o embedding | 7.6 MB | `cc96a139ff51233b…` |
| `citrus_foliar_cc.keras` | pesos Keras | 23.9 MB | `8e22f1216ef3d4aa…` |
| `citrus_foliar_cc.labels.json` | rótulos | 0 KB | `a65b9919ff566964…` |
| `citrus_foliar_cc.novelty.npz` | detector de novidade | 642 KB | `705438c5cba75a33…` |
| `citrus_foliar_cc.onnx` | ONNX (Edge/Rover) | 9.3 MB | `0e3bba14377d62aa…` |
| `citrus_foliar_cc.prior.json` | prior de classes | 1 KB | `4826feb71063f577…` |
| `citrus_foliar_cc.quality.json` | qualidade do dataset/medidas | 0 KB | `4a4f8af6074f1e02…` |

### `citrus_foliar_fundo`

Citros — variante com fundo (experimento).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `citrus_foliar_fundo.keras` | pesos Keras | 23.9 MB | `5d71abed1cd34e25…` |
| `citrus_foliar_fundo.labels.json` | rótulos | 0 KB | `a65b9919ff566964…` |

### `citrus_foliar_papel`

Citros — variante folha no papel (experimento).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `citrus_foliar_papel.keras` | pesos Keras | 23.9 MB | `8e22f1216ef3d4aa…` |
| `citrus_foliar_papel.labels.json` | rótulos | 0 KB | `a65b9919ff566964…` |

### `corn_disease_model`

`leaf-foliar` — milho, 4 classes (piloto comercial).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `corn_disease_model.conformal-a05.json` | conformal α=0,05 | 0 KB | `d23c65167a28346c…` |
| `corn_disease_model.conformal-a10.json` | conformal α=0,10 | 0 KB | `0ce3b9cf418be3ff…` |
| `corn_disease_model.conformal.json` | conformal (α padrão) | 0 KB | `275319064f9485eb…` |
| `corn_disease_model.keras` | pesos Keras | 23.9 MB | `b6a9a95f8d2f383c…` |
| `corn_disease_model.labels.json` | rótulos | 0 KB | `aa3faa3f1398f195…` |
| `corn_disease_model.novelty.npz` | detector de novidade | 360 KB | `1c4bc49dbd9353f0…` |
| `corn_disease_model.onnx` | ONNX (Edge/Rover) | 9.2 MB | `f5ef977c546c0a16…` |
| `corn_disease_model.quality.json` | qualidade do dataset/medidas | 0 KB | `721b99d6421d9722…` |

### `leaf_efficientnetv2s`

Milho — arquitetura EfficientNetV2-S (experimento/legado).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `leaf_efficientnetv2s.keras` | pesos Keras | 111.4 MB | `236f562f97cdbb3d…` |
| `leaf_efficientnetv2s.labels.json` | rótulos | 0 KB | `aa3faa3f1398f195…` |

### `leaf_foliar_b`

`leaf-foliar-b` — milho, segunda opinião da dupla validação.

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `leaf_foliar_b.conformal-a05.json` | conformal α=0,05 | 0 KB | `db5d314d9e242e0c…` |
| `leaf_foliar_b.conformal-a10.json` | conformal α=0,10 | 0 KB | `ba10ee9f9da78230…` |
| `leaf_foliar_b.conformal.json` | conformal (α padrão) | 0 KB | `89386a9b582f87e5…` |
| `leaf_foliar_b.keras` | pesos Keras | 38.5 MB | `e4e54b76f1d2e8df…` |
| `leaf_foliar_b.labels.json` | rótulos | 0 KB | `aa3faa3f1398f195…` |
| `leaf_foliar_b.novelty.npz` | detector de novidade | 447 KB | `c8e811c2ef08c939…` |
| `leaf_foliar_b.onnx` | ONNX (Edge/Rover) | 23.2 MB | `820d02f3e35fce0d…` |
| `leaf_foliar_b.quality.json` | qualidade do dataset/medidas | 0 KB | `721b99d6421d9722…` |

### `leaf_mobilenetv2`

Milho — MobileNetV2 (legado).

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `leaf_mobilenetv2.keras` | pesos Keras | 11.1 MB | `fa2fa83b946f43cc…` |
| `leaf_mobilenetv2.labels.json` | rótulos | 0 KB | `aa3faa3f1398f195…` |

### `leaf_severity`

Severidade: área lesionada no quadro.

| Arquivo | O que é | Tamanho | SHA-256 |
| --- | --- | --- | --- |
| `leaf_severity.keras` | pesos Keras | 21.3 MB | `dccf8d83080f84d6…` |
| `leaf_severity.metrics.json` | métricas | 1 KB | `4771c8e656bd1892…` |
| `leaf_severity.onnx` | ONNX (Edge/Rover) | 20.7 MB | `896579e92df183fc…` |

Hash completo de cada arquivo: [`SHA256SUMS`](SHA256SUMS).
