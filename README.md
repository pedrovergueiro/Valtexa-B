# Corn & Coffee Disease Detection — Backend

Backend de inferência (FastAPI) para detecção de doenças em milho e café a
partir de imagens de drone e de rover terrestre.

A API serve **um modelo por combinação de cultura e enquadramento** — o drone
varre e tria, o rover confirma e diagnostica. O detalhamento de cada modelo, das
classes e dos datasets está em **[MODELS.md](MODELS.md)**.

| Modelo | Cultura | Plataforma | Estado | Papel |
| --- | --- | --- | --- | --- |
| `leaf-foliar` | Milho | qualquer | **treinado** · teste 0,9555 | Diagnóstico foliar em close |
| `coffee-foliar` | Café | qualquer | **treinado** · teste 0,9499 | Diagnóstico foliar, diferencial estratégico |
| `rover-understory` | Milho | rover | **em quarentena** | Diagnóstico sob o dossel, colmo e base |
| `aerial-canopy` | Milho | drone | precisa de coleta | Triagem de dossel, direciona o rover |

Acurácia sobre conjunto de teste isolado, com TTA, medida em 08/09/2026 — ver
[MODELS.md](MODELS.md) para o relatório por classe e as ressalvas de cada uma.
O `rover-understory` está em `models/_quarentena/`: ele marcou 1,0000 no teste
porque aprendeu a **sessão de fotografia**, não a doença — uma regressão
logística que lê apenas a cor média da imagem também acerta 100%.

## Estrutura

```
app/                      # API de inferência
  main.py                 # rotas: /health, /models, /classes, /predict, /predict/batch
  registry.py             # catálogo de modelos (classes, pesos, datasets)
  model.py                # carregamento dos pesos + execução das previsões
  review.py               # fila de revisão do agrônomo (ciclo de dados próprio)
  keras_backend.py        # escolha do backend do Keras e isolamento da GPU
  config.py               # variáveis de ambiente
  schemas.py              # contrato HTTP (Pydantic)
  inference/
    preprocess.py         # decodificação e pré-processamento das imagens
    novelty.py             # detecção do que o modelo nunca viu (Mahalanobis)
training/                 # treino
  train.py                # script principal (2 fases: cabeça + fine-tuning)
  evaluate.py             # relatório por classe de um modelo já treinado
  metrics.py              # precisão/recall/F1 e matriz de confusão
  fit_novelty.py          # calibra a detecção de novidade após o treino
  prepare_dataset.py      # consolida datasets (públicos ou revisados) no layout canônico
  dedupe_dataset.py       # remove imagens repetidas antes do treino
  import_maize_in_field.py  # Maize_in_Field: CSV multirrótulo -> pastas
  import_plantseg.py      # PlantSeg: classe no nome do arquivo -> pastas
  dataset.py              # carregamento e augmentation
  model_def.py            # arquiteturas (MobileNetV2, EfficientNetV2) + cabeça
tests/                    # suíte da API, do catálogo, da novidade e do pré-processamento
models/                   # pesos + .labels.json + .novelty.npz (não versionados)
data/<modelo>/            # dataset por modelo (não versionado)
review/                   # fila do agrônomo: pending/ e labeled/ (não versionado)
```

## Requisitos

**Python 3.10–3.12.** O TensorFlow ainda não publica wheels para 3.13+, então um
ambiente 3.14 não consegue instalar as dependências de treino.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
copy .env.example .env
```

Os requisitos são divididos por para que serve cada coisa:

| Arquivo | Para quê | O que traz |
| --- | --- | --- |
| `requirements.txt` | Rodar a API | FastAPI, Keras e **torch**. Sem TensorFlow. |
| `requirements-training.txt` | Treinar | O acima + TensorFlow, usado só como `tf.data`. |
| `requirements-dev.txt` | Desenvolver | O acima + pytest, httpx e ruff. |

### Backend do Keras

O projeto usa **Keras 3**, que é agnóstico de backend: a mesma arquitetura e o
mesmo arquivo `.keras` rodam sobre PyTorch, TensorFlow ou JAX. O padrão é
**torch**, definido em [`app/keras_backend.py`](app/keras_backend.py).

O motivo é concreto: o TensorFlow não publica kernels CUDA para as GPUs
**Blackwell** (RTX 50xx, compute capability 12.0). Nelas ele enxerga a placa e
falha na primeira operação — inclusive num `Cast` — com
`CUDA_ERROR_INVALID_PTX`, porque compilar o PTX em tempo de execução também não
funciona. O torch publica `sm_120` desde o cu128.

O TensorFlow continua no pipeline de treino, mas só como carregador de dados
(`tf.data`); `keras_backend.py` tira a GPU dele para que os dois não disputem a
placa. Para voltar tudo ao TensorFlow:

```bash
export KERAS_BACKEND=tensorflow
```

## Treinando

Passo a passo completo, com datasets e classes de cada modelo, em
**[MODELS.md](MODELS.md)**. O caminho curto:

```bash
# consolida os datasets públicos em data/leaf-foliar/<Classe>/
python -m training.prepare_dataset --model leaf-foliar \
    --source "C:/datasets/corn-or-maize-leaf-disease" --dry-run

# remove repetidas — pule este passo e a métrica vira ficção (ver abaixo)
python -m training.dedupe_dataset --model leaf-foliar --dry-run
python -m training.dedupe_dataset --model leaf-foliar

# treina
python -m training.train --model leaf-foliar --tta

# relatório por classe, sem retreinar
python -m training.evaluate --model leaf-foliar --tta
```

### Deduplique antes de treinar

Não é higiene opcional. Os datasets públicos repetem imagem em escala que
surpreende: o JMuBEN distribuído no Kaggle tem **2.570 imagens distintas
infladas para 59.942** — a mesma foto até 1.800 vezes, com MSE zero entre as
cópias. Treinado assim, o `coffee-foliar` marcou **0,9962** de `val_accuracy`.
O número era falso: com o sorteio aleatório do *split*, praticamente toda
imagem de validação tinha cópia idêntica no treino, e o modelo estava sendo
avaliado no que decorou.

O `dedupe_dataset` compara por assinatura perceptual (dHash), não por hash do
arquivo, porque as cópias reaparecem recomprimidas e redimensionadas. Ele trata
dois casos de forma diferente:

- **cópia dentro da mesma classe** — mantém uma;
- **mesma imagem em classes diferentes** — descarta todas, porque é conflito de
  rótulo e não há como saber qual está certo.

### Escolhendo a arquitetura

`--arch` aceita `mobilenetv2` (padrão), `efficientnetv2b0` e `efficientnetv2s`.
Mais parâmetros nem sempre significam melhor: no `leaf-foliar`, a
EfficientNetV2-S custa 9,4× o tamanho em disco, sobe 0,65 ponto de acurácia e
**piora** o macro-F1. Meça antes de trocar — ver
[o teto do dataset](MODELS.md#teto-do-dataset).

Ao final, `models/` contém os pesos da **melhor época** segundo `val_accuracy` e
o `.labels.json` com a ordem das classes — que é o que a API usa para traduzir o
índice previsto em nome.

`python -m training.train --help` lista todos os parâmetros.

Depois de treinar, calibre a detecção de novidade (uma vez por modelo):

```bash
python -m training.fit_novelty --model leaf-foliar
```

Sem esse passo o modelo funciona normalmente — só não marca o que nunca viu.
Ver [ciclo de dados próprio](#ciclo-de-dados-próprio) abaixo.

### GPU no Windows

Treine pelo **WSL2**: o `tf.data` do pipeline não tem wheel de GPU para Windows
nativo, e o torch com CUDA é bem mais simples de instalar no Linux. Instale o
torch pelo índice da arquitetura da sua placa — o wheel padrão do PyPI pode não
trazê-la:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.get_arch_list())"
```

A lista impressa precisa conter a compute capability da sua GPU (`sm_120` para
as RTX 50xx). Se não contiver, o treino cai para CPU ou quebra em tempo de
execução.

Referência de velocidade, numa RTX 5060 Ti (8 GB) com `--batch-size 64`:
`leaf-foliar` (4.566 imagens) fecha uma época em ~17 s, contra ~6 min em CPU.

## Rodando a API

```bash
uvicorn app.main:app --reload
```

Docs interativas: http://localhost:8000/docs

A API sobe mesmo sem pesos: `/health` responde `200` e cada modelo sem pesos
responde `503` individualmente. Dá para colocar o `leaf-foliar` em produção
enquanto os outros dois ainda estão em coleta.

### Endpoints

| Método | Rota | Descrição |
| --- | --- | --- |
| `GET` | `/health` | Estado da API e modelos prontos. |
| `GET` | `/models` | Catálogo completo: classes, plataforma, datasets, se está carregado. |
| `GET` | `/classes` | Classes de um modelo. |
| `POST` | `/predict` | `multipart/form-data`, campo `file`. Uma imagem. |
| `POST` | `/predict/batch` | `multipart/form-data`, campo `files`. Várias de uma vez. |
| `GET` | `/review/stats` | Tamanho da fila de revisão. |
| `GET` | `/review` | Imagens aguardando diagnóstico, com o palpite do modelo. |
| `GET` | `/review/{id}/image` | A foto de um item da fila. |
| `POST` | `/review/{id}` | Registra o diagnóstico do agrônomo. |

As rotas aceitam `?model=<chave>` ou `?platform=aerial|ground`. Sem nenhum dos
dois, usam o modelo padrão (`DEFAULT_MODEL`).

```bash
curl -F "file=@folha.jpg" http://localhost:8000/predict
curl -F "file=@colmo.jpg" "http://localhost:8000/predict?platform=ground"
```

```json
{
  "filename": "folha.jpg",
  "model_key": "leaf-foliar",
  "predicted_class": "Common_Rust",
  "confidence": 0.973,
  "abstained": false,
  "novelty_score": 0.42,
  "needs_review": false,
  "probabilities": [
    { "class_name": "Blight", "probability": 0.011 },
    { "class_name": "Common_Rust", "probability": 0.973 },
    { "class_name": "Gray_Leaf_Spot", "probability": 0.009 },
    { "class_name": "Healthy", "probability": 0.007 }
  ],
  "inference_ms": 42.8
}
```

`novelty_score` acima de `1.0` significa que a imagem não se parece com nada
que o modelo viu no treino — inclusive quando ele responde com confiança alta
mesmo assim. `needs_review` vira `true` nesse caso; ver
[ciclo de dados próprio](#ciclo-de-dados-próprio).

O `/predict/batch` processa o lote inteiro numa única passada e acrescenta um
`summary` com a contagem por classe e um `needs_review_count` — útil para fechar
o resultado de um voo e saber quanto sobra para o agrônomo olhar. A validação é
tudo-ou-nada: se uma imagem for ilegível, a requisição falha com `400`
identificando o arquivo.

## Configuração

Todas opcionais, via `.env` (ver `.env.example`).

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `DEFAULT_MODEL` | `leaf-foliar` | Modelo usado quando nenhum é informado. |
| `MODELS_DIR` | `models` | Onde ficam os pesos. |
| `DATA_DIR` | `data` | Raiz dos datasets de treino. |
| `MODEL_PATH` | — | Sobrescreve os pesos **só do modelo padrão**. |
| `MIN_CONFIDENCE` | `0.0` | Abaixo disso a resposta vem como `abstained`. **Recomendado `0.7` em produção** — ver abaixo. |
| `NOVELTY_SENSITIVITY` | `1.0` | Regula o volume da fila de revisão; menor = mais imagens marcadas. |
| `REVIEW_DIR` | — | Onde arquivar as imagens marcadas. Vazio = fila desligada (só marca `needs_review` na resposta). |
| `MAX_UPLOAD_MB` | `10` | Tamanho máximo por imagem. |
| `MAX_BATCH_SIZE` | `20` | Imagens por requisição em `/predict/batch`. |
| `CORS_ORIGINS` | `localhost:5173,localhost:3000` | Origens do front, separadas por vírgula. |
| `LOG_LEVEL` | `INFO` | Nível de log. |

Com `CORS_ORIGINS=*` as credenciais de CORS são desativadas automaticamente — a
combinação `*` + credenciais é inválida por spec e, na prática, liberaria
qualquer site.

### Os dois filtros de segurança, e o que cada um pega

`MIN_CONFIDENCE` e `NOVELTY_SENSITIVITY` interceptam coisas **diferentes**, e o
projeto sai com os dois desligados por padrão. Medido no `leaf-foliar` com sete
entradas fora de escopo:

| Entrada | Confiança | Novidade | Pego por |
| --- | ---: | ---: | --- |
| Folha de café | 0,527 | 0,822 | confiança |
| Colmo | **1,000** | **1,005** | **novidade** |
| Cor chapada clara | 0,435 | 0,960 | confiança |
| Ruído | 0,930 | **1,102** | novidade |

A novidade pega o que é visualmente estranho **mesmo com confiança alta** — a
foto de colmo virava `Blight` com 100% de certeza. O `MIN_CONFIDENCE` pega o que
o modelo já responde com dúvida. Com `MIN_CONFIDENCE=0.7` e
`NOVELTY_SENSITIVITY=1.0`, **6 das 7** entradas são interceptadas, e as imagens
legítimas — que saem entre 0,999 e 1,000 de confiança — passam sem custo.

## Ciclo de dados próprio

Um softmax é fechado: diante de cigarrinha, deficiência nutricional ou qualquer
coisa fora do catálogo, ele responde uma das classes que conhece — às vezes com
confiança alta e errado. A API detecta isso comparando o embedding da imagem
(assinatura visual de 128 dimensões, extraída na mesma passada da inferência)
contra o que o modelo viu no treino. O que não se parece com nada conhecido
vira `needs_review: true` e, se `REVIEW_DIR` estiver configurado, é arquivado
com o palpite do modelo ao lado.

```
/predict → needs_review? → review/pending/<data>/<id>.jpg + <id>.json
                                          ↓  agrônomo diagnostica e move
                          review/labeled/<Classe>/<id>.jpg
                                          ↓
              python -m training.prepare_dataset --source review/labeled
                                          ↓
                          data/<modelo>/<Classe>/  →  train  →  fit_novelty
```

A ação do agrônomo é arrastar o arquivo para a pasta com o nome da classe — todo
nome do catálogo já é reconhecido pelo `prepare_dataset`. Detalhe completo,
incluindo como regular `NOVELTY_SENSITIVITY`, em
**[MODELS.md](MODELS.md#novidade)**.

### Pela API, sem tocar em disco

O ciclo acima funciona por linha de comando, mas o agrônomo não precisa de
acesso ao servidor. As quatro rotas de `/review` cobrem o fluxo inteiro:

```bash
curl localhost:8000/review/stats                  # quanto está esperando
curl localhost:8000/review                        # a lista, com o palpite do modelo
curl localhost:8000/review/<id>/image --output f.jpg   # a foto
curl -X POST localhost:8000/review/<id> \
     -H 'Content-Type: application/json' \
     -d '{"class_name": "Cigarrinha", "reviewed_by": "agrônoma Ana"}'
```

**O rótulo não precisa existir no catálogo.** Um nome novo cria uma classe nova
em `review/labeled/<Classe>/`, e é exatamente assim que uma doença que o modelo
não conhecia entra no dataset próprio. O `prepare_dataset` consome esse
diretório como qualquer outro, e a imagem entra no próximo treino.

Como o rótulo e o id viram caminho de arquivo, os dois são validados por formato
fechado: o id precisa casar com o `uuid4` gerado na entrada, e a classe aceita
apenas letras, números e `_`. Um `../` em qualquer um deles é recusado antes de
tocar o disco.

A fila só grava com `REVIEW_DIR` definido — sem ele, as rotas respondem `503`.
Guardar imagem de cliente em disco é decisão de quem opera, não default nosso.

As rotas são de **consulta**: o cliente pergunta o que há na fila. Não existe
push — nenhum webhook, nenhum alerta sai daqui sozinho.

### O que acontece com imagem que não deveria chegar

Teste feito em 08/09/2026 com 210 quadros de vídeos de lavoura do YouTube. Não é
captura da plataforma, e quase nada tem o enquadramento que os modelos
esperam — que é exatamente o motivo de servir como sanidade.

| Entrada | O modelo respondeu | Interceptado |
| --- | --- | ---: |
| Folha de café com **deficiência de magnésio** | Phoma 40%, Healthy 40%, Rust 20% | **75%** |
| **Slides de PowerPoint** sobre doenças do milho | Common_Rust 57%, Gray_Leaf_Spot 39% | 41% |
| **Vista aérea de drone** de lavoura de café | Rust em 94% dos quadros, confiança média 0,93 | **9%** |

O primeiro caso é a engrenagem funcionando: deficiência nutricional não está no
catálogo, o modelo não tinha resposta certa disponível, e três de cada quatro
quadros foram para a fila do agrônomo. É assim que `Deficiencia_Magnesio` vira
uma classe nova.

**O terceiro caso é o perigoso, e nenhuma configuração resolve.** Uma foto aérea
de dossel — onde nenhuma folha individual é sequer visível — vira `Rust` com 93%
de confiança, e a detecção de novidade não a marca: no espaço de embeddings, uma
textura verde de copa cai perto do que o modelo aprendeu como ferrugem.

A consequência prática é uma regra de operação, não um parâmetro:

> **Não envie imagem aérea para `leaf-foliar` ou `coffee-foliar`.** Eles são
> modelos de *enquadramento*, e exigem a folha preenchendo o quadro. Triagem de
> dossel é o papel do `aerial-canopy`, que ainda precisa de coleta — enquanto
> ele não existir, essa validação tem de ser feita por quem chama a API.

## Paridade entre treino e inferência

O modo de falha mais perigoso do projeto é um pré-processamento diferente entre
treino e produção: não gera erro, só derruba a acurácia em silêncio.

**A normalização é literalmente o mesmo código.** Os dois lados chamam
`scale_to_unit_range` de [`app/inference/preprocess.py`](app/inference/preprocess.py)
para levar os pixels a `[-1, 1]`. Aqui não há como divergir.

**O redimensionamento tem a mesma semântica, mas duas implementações.** Os dois
lados são bilineares e não preservam aspect ratio — o treino usa
`tf.image.resize`, a inferência usa o Pillow. A decodificação do treino foi
verificada contra a do Keras e é **idêntica ao pixel** (diferença máxima
`0.000000`), mas os núcleos de reamostragem do Pillow e do TensorFlow não são os
mesmos: medida sobre imagens reais, a diferença máxima por pixel chega a **0,83
numa escala de 2,0**, concentrada nas bordas.

Não é divergência de escala nem de canal — é ruído de reamostragem, e nenhum
ensaio até agora atribuiu perda de acurácia a ele. Mas é uma folga conhecida, e
está registrada aqui em vez de ser afirmada como paridade perfeita. Fechá-la de
verdade exigiria decodificar com a mesma biblioteca nas duas pontas.

`tests/test_preprocess.py` cobre o contrato observável — faixa de saída, formato
e comportamento do aspect ratio — e é pulado quando o TensorFlow não está
instalado.

## Testes e lint

```bash
pytest
ruff check .
ruff format .
```

A suíte roda **sem TensorFlow**: as rotas, o catálogo e a detecção de novidade
são exercitados com um dublê do modelo Keras e embeddings sintéticos. O único
teste que exige o TF é o de paridade do pré-processamento, que é pulado
automaticamente quando ele não está instalado.

## Docker

```bash
docker build -t valtexa-disease-backend .
docker run -p 8000:8000 -v "$PWD/models:/app/models" valtexa-disease-backend
```

Os pesos não entram na imagem — são binários grandes e versionados à parte, então
`models/` é montado como volume em tempo de execução.

## Roadmap

- [ ] Treinar `leaf-foliar` e `coffee-foliar`, calibrar `fit_novelty`, validar cada
      um num conjunto de teste próprio.
- [ ] Piloto comercial com drone terceirizado (voo baixo + `leaf-foliar`/`coffee-foliar`).
- [ ] Primeiro ciclo real do fluxo de revisão: ligar `REVIEW_DIR` em produção e
      medir quanto do volume de voo vira dataset novo por semana.
- [ ] Coleta própria de café no Sul de Minas — ver prioridade em [MODELS.md](MODELS.md).
- [ ] Coleta de campo para as classes exclusivas do rover — ver [MODELS.md](MODELS.md).
- [ ] Camada de fusão: risco do drone → tasking do rover → diagnóstico confirmado.
- [ ] Autenticação entre front e backend (API key / JWT).
- [ ] Integrar o front (`VALTEXA FRONT`), hoje uma landing page sem chamadas à API.
- [ ] Deploy (cloud, servidor próprio ou edge na ground station).
