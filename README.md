# VALTEXA — Detecção de doenças de plantas (API e rover)

Backend de inferência (FastAPI) para detecção de doenças em **citros, banana**,
milho e café a partir de imagens de drone e do rover terrestre. O mesmo código
serve a API e roda na Jetson do rover.

A API serve **um modelo por combinação de cultura e enquadramento** — o drone
varre e tria, o rover confirma e diagnostica. O detalhamento de cada modelo, das
classes e dos datasets está em **[MODELS.md](MODELS.md)**.

| Modelo | Cultura | Plataforma | Estado | Papel |
| --- | --- | --- | --- | --- |
| `banana-foliar` | Banana | rover / qualquer | **treinado** · campo 0,829 · teste 0,977 | Folhas baixas da bananeira |
| `citrus-foliar` | Citros (laranja e limão) | rover / qualquer | **treinado** · campo 0,969 · teste 0,884 | Copa baixa do citros |
| `leaf-foliar` | Milho | qualquer | **treinado** · teste 0,9555 | Diagnóstico foliar em close |
| `coffee-foliar` | Café | qualquer | **treinado** · teste 0,9499 | Diagnóstico foliar, diferencial estratégico |
| `rover-understory` | Milho | rover | **em quarentena** | Diagnóstico sob o dossel, colmo e base |
| `aerial-canopy` | Milho | drone | precisa de coleta | Triagem de dossel, direciona o rover |

"Teste" é a acurácia no conjunto de teste isolado, com TTA — ver
[MODELS.md](MODELS.md) para o relatório por classe e as ressalvas de cada uma.
**"Campo" é o número a citar para banana e citros**: fotos de campo de outra
fonte (PlantSeg), que nenhum treino viu, cortadas na escala dos recortes do
rover — ver [o teste de campo](#o-teste-de-campo-na-escala-do-rover). O teste
isolado sai das mesmas fontes do treino e engana: o primeiro modelo de banana
marcou 0,985 nele e 0,428 no campo.
O `rover-understory` está em `models/_quarentena/`: ele marcou 1,0000 no teste
porque aprendeu a **sessão de fotografia**, não a doença — uma regressão
logística que lê apenas a cor média da imagem também acerta 100%.

**Sistema inteiro** (como esta IA conversa com a plataforma e o Rover, componentes,
decisões, estado): `VALTEXA-SAAS/docs/ARCHITECTURE.md`, `COMPONENTS.md`, `DECISIONS.md` e
`ESTADO_ATUAL.md`. **Testes**: rodam no WSL (no Windows o App Control bloqueia a DLL do
ONNX) com `pytest` e `ruff check .` (ruff **0.6.9**, fixado em `requirements-dev.txt`);
cada teste começa com uma pasta de modelos vazia (`tests/conftest.py`), então a suíte não
depende dos pesos de `models/`.

## Estrutura

```
app/                      # API de inferência
  main.py                 # rotas: /health, /models, /classes, /predict, /predict/batch, /predict/explain
  registry.py             # catálogo de modelos (classes, pesos, datasets)
  model.py                # carregamento dos pesos + execução das previsões
  review.py               # fila de revisão do agrônomo (ciclo de dados próprio)
  keras_backend.py        # escolha do backend do Keras, isolamento da GPU e no_grad
  framing.py              # portão de enquadramento: SigLIP 2, zero-shot
  rover.py                # geometria do rover: quadro -> recortes de 150 mm, altura e zona
  rover_api.py            # rota /predict/rover-frame
  plant_fusion.py         # fusão multivista por planta, incidência, evolução entre passadas, GeoJSON
  fields_api.py           # rotas /fields: plantas, mapa, passadas, incidência e prevalência
  active_learning.py      # ordem da fila de revisão pelo que mais ensina
  sigatoka_severity.py    # escala de Stover-Gauhl, índice de severidade (DSI), kappa
  onnx_runtime.py         # servir pelo ONNX Runtime, sem Keras nem torch
  config.py               # variáveis de ambiente
  schemas.py              # contrato HTTP (Pydantic)
  inference/
    preprocess.py         # decodificação e pré-processamento das imagens
    novelty.py             # detecção do que o modelo nunca viu (Mahalanobis)
    quality.py            # a foto está boa o bastante? (nitidez, exposição, contraste)
    explain.py            # onde o modelo olhou (mapa de oclusão)
    conformal.py          # conjunto de classes com cobertura garantida
    prior_shift.py        # prevalência por talhão (EM de Saerens, BBSE) e monitor de deriva
    heads.py              # segunda leitura: cabeças dNATY sobre o embedding do portão (NumPy)
training/                 # treino
  train.py                # script principal (2 fases: cabeça + fine-tuning)
  evaluate.py             # relatório por classe de um modelo já treinado
  metrics.py              # precisão/recall/F1 e matriz de confusão
  fit_novelty.py          # calibra a detecção de novidade após o treino
  fit_quality.py          # calibra a verificação de qualidade da captura (sem pesos)
  audit_attention.py      # audita, por classe, se o modelo decide pela lesão ou pela foto
  fit_conformal.py        # calibra a predição conformal na validação, mede no teste
  fit_framing.py          # calibra o portão de enquadramento com fotos boas
  fit_heads.py            # treina as cabeças do embedding do portão com o dNATY
  export_onnx.py          # exporta para ONNX, recusando se a paridade não se sustenta
  prepare_dataset.py      # consolida datasets (públicos ou revisados) no layout canônico
  dedupe_dataset.py       # remove imagens repetidas antes do treino
  download_public.py      # baixa datasets do Mendeley Data e do Dataverse
  background.py           # troca de fundo: folha sobre papel -> folha sobre pomar
  lesion_tiles.py         # recortes na escala do rover sobre as lesões do PlantSeg
  demo_photos.py          # fotos da demonstração do rover no SaaS (simulator.pomar)
  fit_prior.py            # prevalência das classes no treino, ponto de partida do EM
  fit_fusion.py           # calibra a fusão por planta e mede/recalibra com o rótulo de campo
  import_maize_in_field.py  # Maize_in_Field: CSV multirrótulo -> pastas
  import_plantseg.py      # PlantSeg: classe no nome do arquivo -> pastas
  import_bracol.py        # BRACOL: rótulo no CSV -> pastas (lê o zip truncado do Mendeley)
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

Para banana e citros há dois passos a mais — o teste de campo e os recortes de
campo no treino, que foram o que fez os dois modelos funcionarem fora do dataset
(ver [o teste de campo](#o-teste-de-campo-na-escala-do-rover)):

```bash
# recortes na escala do rover, cortados sobre as lesões marcadas no PlantSeg,
# separados pelo split oficial: train/ entra no treino, test/ é o teste de campo
python -m training.lesion_tiles --source ~/valtexa-raw/plantseg/plantseg \
    --output ~/valtexa-raw/_plantseg-recortes

# mede o modelo numa pasta de outra fonte, com só parte das classes
python -m training.evaluate --model banana-foliar --tta \
    --external ~/valtexa-raw/_plantseg-recortes/test/banana
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

**O dHash sozinho apagava folhas diferentes.** Toda folha destacada sobre papel
liso tem o mesmo gradiente — claro, escuro, claro. No citros, com a distância 4,
a união transitiva juntou as 12 classes num grupo só e ia apagar **68% do
dataset**. Agora o dHash só indica o par; quem confirma é a miniatura colorida
**da folha** (a caixa do que não é papel), que precisa diferir no máximo 8 níveis.
Medido em 6.000 pares indicados no citros: entre classes, nenhum par fica até 10
(mediana 39); dentro da classe, há um vão limpo entre a mesma folha fotografada
em sequência e folhas diferentes. O citros saiu com 11.764 fotos distintas de
18.839 — os datasets fotografam a mesma folha até 14 vezes seguidas. Na banana, a
confirmação devolveu 1.600 fotos que o método antigo descartava como "conflito".

<a id="imagens-quase-pretas-no-dataset"></a>
### Máscaras de segmentação não são fotos

Encontrado em 26/09/2026, ao calibrar a qualidade da captura: 1,6% das "fotos"
de treino do milho tinham brilho médio abaixo de 20 numa escala de 255. Eram
**máscaras de segmentação do PlantSeg** — fundo preto e polígonos cinza com o
índice da classe. O PlantSeg publica a máscara de cada foto com o mesmo nome, em
`.png`, e o importador trouxe as duas:

| Dataset | Máscaras como fotos | Classes | No teste isolado |
| --- | ---: | --- | ---: |
| `leaf-foliar` | **388** de 19.787 | `Common_Rust` 163, `Blight` 127, `Gray_Leaf_Spot` 98 | 55 |
| `coffee-foliar` | **150** de 3.462 | todas `Rust` | — |

No milho, o modelo aprendeu a diagnosticar imagens pretas: acerta 80% das
máscaras do teste. A acurácia publicada não foi inflada por elas — sem as
máscaras, o `leaf-foliar` com TTA sobe de 95,49% para 95,78% —, mas um modelo
que responde com confiança sobre o que não é folha é exatamente o defeito que as
salvaguardas deste projeto tentam conter. No café o estrago é maior: 1 em cada 10
exemplos de ferrugem é uma imagem sem folha, e o modelo aprendeu que sem textura
quer dizer `Rust` (ver [o atalho do café](#o-atalho-do-café-sem-textura-quer-dizer-rust)).

O `prepare_dataset` e o `import_plantseg` agora recusam mapas de rótulo — imagem
cinza com até 16 tons distintos, que foto nenhuma tem. Nos dados reais o
detector pega as 388 do milho e nenhuma das 19.399 fotos. As calibrações
(`fit_quality`, `fit_conformal`, `fit_framing`) também as ignoram.

Os arquivos **não** foram removidos de `data/`: tirá-los muda a permutação do
corte, e os pesos atuais passariam a ter imagens de treino no "teste isolado". A
limpeza vai junto com o próximo retreino — reimporte o PlantSeg e retreine.

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

Calibre também a qualidade da captura. Não usa os pesos nem o Keras — roda no
mesmo ambiente da API, e mede antes de gravar quanto marca de foto boa e quanto
pega de foto estragada:

```bash
python -m training.fit_quality --model leaf-foliar
```

E, antes de pôr pesos novos em produção, audite **onde** o modelo olha:

```bash
python -m training.audit_attention --model leaf-foliar --per-class 25 --out reports/atencao
```

A acurácia não denuncia um modelo que aprendeu o atalho errado — o atalho
acerta. A auditoria denuncia. Ver [onde o modelo olhou](#onde-o-modelo-olhou).

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
| `GET` | `/models/choices` | As versões de cada cultura, para escolher qual usar — ver [escolhendo o modelo](#escolhendo-o-modelo). |
| `GET` | `/classes` | Classes de um modelo. |
| `POST` | `/predict` | `multipart/form-data`, campo `file`. Uma imagem. |
| `POST` | `/predict/batch` | `multipart/form-data`, campo `files`. Várias de uma vez. |
| `POST` | `/predict/explain` | Uma imagem. Mapa de onde o modelo olhou (`method=occlusion` ou `rise`), com a foto pintada em PNG. |
| `POST` | `/predict/rover-frame` | Quadro inteiro de uma câmera lateral do rover. Recorta em escala física e diz o que achou em cada zona da planta — ver [rover](#rover-do-quadro-à-zona-da-planta). |
| `GET` | `/fields` | Talhões com histórico do rover — ver [talhão](#talhão-da-foto-à-planta-e-da-planta-ao-mapa). |
| `GET` | `/fields/{talhão}/plants` | Diagnóstico fundido de cada planta. |
| `GET` | `/fields/{talhão}/map` | Mapa do talhão em GeoJSON. |
| `GET` | `/fields/{talhão}/plants/{planta}/evolution` | Histórico da planta entre passadas, com AUDPC. |
| `GET` | `/fields/{talhão}/plants/{planta}/lesoes` | As lesões no referencial da planta (ao longo, altura, dentro da copa, inclinação), reencontradas entre passadas: `nova`, `persistente`, `nao_revista`. Precisa da geometria 3D do Rover. |
| `POST` | `/fields/{talhão}/passes/{passada}/close` | Fecha a passada: o que mudou desde a anterior. |
| `POST` | `/fields/{talhão}/frames?model=` | Quadro **já classificado no Rover** (probabilidades de cada recorte) entra na fusão, com a calibração da borda (`<pesos>.edge-fusion.json`); reenvio do mesmo `frame_id` não soma. |
| `GET` | `/fields/{talhão}/incidence` | Fração de **plantas** doentes, com intervalo de 95% — ver [incidência](#incidência-do-talhão). |
| `GET` | `/fields/{talhão}/prevalence` | Proporção de cada classe nos recortes, estimada sem rótulo, e a incidência por planta. |
| `GET`/`POST` | `/fields/{talhão}/clima` | A série climática do talhão (estação, Open-Meteo, climatologia) — ver [gêmeo digital](#gêmeo-digital-do-talhão). |
| `GET` | `/fields/{talhão}/gemeo` | O grafo do talhão e os modelos epidemiológicos ajustados. |
| `POST` | `/fields/{talhão}/gemeo/ajuste?doenca=` | Ajusta o modelo epidemiológico às passadas fechadas e treina a rede residual (com portão). Com `dispersao=true&azimute_rua_graus=`, escolhe também a parte do contágio que vai pelo vento (`app/gemeo/dispersao.py`; desligada por padrão). |
| `GET` | `/fields/{talhão}/gemeo/previsao?doenca=` | Risco por planta no horizonte (inclui o latente), janelas de infecção, focos de agora e previstos. |
| `POST` | `/fields/{talhão}/gemeo/prescricao` | Prescrição sobre o risco previsto, com a data de aplicação e os cenários. |
| `POST` | `/fields/{talhão}/gemeo/aplicacao-localizada` | Prescrição variável preditiva: mapa de risco por trecho de rua, taxa por trecho dentro da bula, ciclo de PWM por bico dentro da máquina, verificações determinísticas e o `sha256` do mapa (o SaaS prende a aprovação a ele). Proposta. |
| `GET` | `/fields/{talhão}/gemeo/prioridades?doenca=&canal=` | Onde vale olhar: trechos de rua (Rover) ou blocos (drone) pelo valor da informação — ver [o gêmeo no enxame](MODELS.md#enxame). |
| `POST` | `/fields/{talhão}/gemeo/observacoes` | Evidência por planta de outra fonte (drone, agrônomo), com o canal dela. O Rover não entra por aqui. |
| `GET` | `/review/stats` | Tamanho da fila de revisão. |
| `GET` | `/review` | Imagens aguardando diagnóstico, com o palpite do modelo. `?order=valor` põe primeiro o que mais ensina. |
| `GET` | `/review/{id}/image` | A foto de um item da fila. |
| `POST` | `/review/{id}` | Registra o diagnóstico do agrônomo. |

As rotas aceitam `?model=<chave>` ou `?platform=aerial|ground`. Sem nenhum dos
dois, usam o modelo padrão (`DEFAULT_MODEL`). `/predict` e `/predict/batch`
aceitam também `?tta=true` — ver [TTA em produção](#tta-em-produção).

### Escolhendo o modelo

Cada cultura tem mais de uma versão, como os modelos de um assistente de IA — e
quem chama escolhe qual usar, a cada requisição, no upload manual ou no rover:

| Versão (`model=`) | Nome | Campo | Licença dos dados | Quando usar |
| --- | --- | ---: | --- | --- |
| `banana-foliar` | Banana · campo | 0,829 | inclui PlantSeg (não comercial) | melhor precisão; piloto e pesquisa |
| `banana-foliar-cc` | Banana · comercial | 0,592 | CC BY 4.0 | uso comercial |
| `citrus-foliar` | Citros · campo | 0,969 | inclui PlantSeg (não comercial) | melhor precisão; piloto e pesquisa |
| `citrus-foliar-cc` | Citros · comercial | 0,695 | CC BY 4.0 | uso comercial |
| `leaf-foliar` / `leaf-foliar-b` | Milho · padrão / segunda opinião | — | ver MODELS.md | dupla validação |
| `coffee-foliar` | Café · padrão | — | ver MODELS.md | |

`GET /models/choices` devolve essa lista agrupada por cultura — a recomendada
primeiro, cada versão com o número de campo, a licença e se está carregada —,
pronta para um seletor no app ou no painel do rover:

```bash
curl http://localhost:8000/models/choices
# upload manual, escolhendo a versão
curl -F "file=@folha.jpg" "http://localhost:8000/predict?model=citrus-foliar-cc"
# rover: a versão explícita, ou só a cultura (usa ROVER_MODELS)
curl -F "file=@quadro.jpg" "http://localhost:8000/predict/rover-frame?model=banana-foliar&camera_height_mm=1000&distance_mm=270"
curl -F "file=@quadro.jpg" "http://localhost:8000/predict/rover-frame?crop=banana&camera_height_mm=1000&distance_mm=270"
```

Com só `crop=`, o rover usa a versão de `ROVER_MODELS` (JSON, ex.:
`{"banana": "banana-foliar-cc"}`) ou, sem ela, a recomendada da cultura — trocar
o rover para a versão comercial é uma variável de ambiente, sem mexer em código.

**O modelo escolhido também decide o portão.** Com a cultura conhecida, o
[portão de enquadramento](#enquadramento) compara a foto só com a folha daquela
cultura e com o que não é folha, e só avisa cultura trocada
(`crop_matches: false`) quando a folha de outra cultura supera a escolhida por 4
vezes. Era a disputa entre culturas que fazia o aviso disparar em 43% das folhas
boas de café quando as cenas de citros entraram; com a folga, 13,8%, e ele ainda
pega 97% das folhas de outra cultura mandadas ao modelo de café (banana: 0,9% de
aviso à toa e 90% pegas; citros: 0,1% e 94%).

O limiar continua o geral. Um limiar por cultura (percentil das fotos boas de
cada uma) foi medido nos recortes de campo e rejeitado: no citros, calibrado em
folha sobre papel, recusou 58 recortes de folha boa de campo; na banana,
calibrado com fotos de planta inteira, deixou passar o pseudocaule (barrou 4
recortes de pseudocaule, corte e tronco, contra 35 com o limiar geral).
`fit_framing --per-crop-thresholds` ainda o grava, para quem tiver uma validação
que pareça o que o rover vê.

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
  "severity": 0.18,
  "view_agreement": 1.0,
  "quality": {
    "ok": true, "issues": [], "actions": [],
    "sharpness": 453.5, "brightness": 57.0, "contrast": 65.8
  },
  "framing": {
    "in_scope": true, "score": 0.998, "scene": "folha_milho",
    "crop": "corn", "crop_matches": true
  },
  "prediction_set": ["Common_Rust"],
  "set_coverage": 0.95,
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

`severity` é a fração do quadro que o modelo de segmentação marcou como lesão
(`models/leaf_severity.keras`, treinado por `training.train_severity`). É área
lesionada **no quadro**: aproxima a severidade da folha quando ela preenche a
imagem, que é a premissa do `leaf-foliar`. Só os modelos de folha de milho a
medem; nos demais, e sem os pesos, vem `null`. Métricas em
[`MODELS.md`](MODELS.md#severidade).

Os campos depois de `severity` são as salvaguardas, cada uma explicada abaixo:
`quality` diz se a foto está boa o bastante e o que fazer quando não está
([qualidade da captura](#qualidade-da-captura)); `view_agreement` só vem com TTA
([TTA em produção](#tta-em-produção)); `framing` diz se a foto é folha em close
da cultura certa ([enquadramento](#enquadramento)); `prediction_set` é o
conjunto de classes que não dá para descartar, com cobertura garantida
([predição conformal](#predição-conformal)). Cada uma vem nula quando o modelo
não tem a calibração dela — e `GET /models` diz, por modelo, quais estão ativas.

O `/predict/batch` processa o lote inteiro numa única passada e acrescenta um
`summary` com a contagem por classe e as contagens `needs_review_count`,
`quality_issue_count`, `out_of_scope_count`, `ambiguous_count` e `empty_set_count` — útil para fechar o resultado de um voo e saber quanto
sobra para o agrônomo olhar e quanto precisa ser fotografado de novo. A validação é
tudo-ou-nada: se uma imagem for ilegível, a requisição falha com `400`
identificando o arquivo.

### Memória da GPU: dois defeitos medidos e corrigidos

Os dois apareceram ao rodar as medições deste README numa RTX 5060 Ti de 8 GB —
o segundo chegou a travar a máquina inteira.

**A inferência gravava o grafo de gradientes.** A API chama o modelo direto —
`model(lote)` — para fugir do overhead do `predict()` em lotes pequenos. No
backend torch, essa chamada **não** desliga o autograd: a saída vinha com
`requires_grad` e `SoftmaxBackward0`, ou seja, cada requisição montava e segurava
o grafo de gradientes da rede inteira para jogá-lo fora. Toda passada agora roda
dentro de `no_grad` ([`app/keras_backend.py`](app/keras_backend.py)); na CPU, o
mesmo lote ficou 28% mais rápido.

**Um lote grande não falha — trava.** MobileNetV2 no backend torch, sem autograd:

| Imagens por passada | Tempo | Vazão | Memória reservada |
| ---: | ---: | ---: | ---: |
| 32 | 59 ms | **545 img/s** | — |
| 128 | **7.023 ms** | 18 img/s | 11,8 GB numa placa de 8 |

São ~90 MB por imagem numa rede de 2,3 M de parâmetros: no backend torch, o
modelo funcional do Keras segura as ativações intermediárias de todas as camadas
até o fim da chamada. Pelo mesmo motivo, o `fit_novelty` com o lote padrão de 64
não coube em 4 GB — rode-o com `--batch-size 16`.

Passou da memória da placa, o driver do Windows pagina a VRAM na RAM do sistema:
a GPU aparece "100% ocupada" rendendo 30 vezes menos, e o computador inteiro
engasga junto. Um `/predict/batch` de 20 imagens com TTA são 80 imagens numa
passada — exatamente esse caso. `MAX_FORWARD_BATCH` (padrão 32) fatia toda
passada, com e sem TTA, e um teste garante que fatiar não muda nenhuma resposta.

## Configuração

Todas opcionais, via `.env` (ver `.env.example`).

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `DEFAULT_MODEL` | `leaf-foliar` | Modelo usado quando nenhum é informado. |
| `MODELS_DIR` | `models` | Onde ficam os pesos. |
| `DATA_DIR` | `data` | Raiz dos datasets de treino. |
| `MODEL_PATH` | — | Sobrescreve os pesos **só do modelo padrão**. |
| `INFERENCE_TTA` | `false` | TTA por padrão em `/predict`. As acurácias publicadas são com TTA; custa 4×. |
| `ROVER_TTA` | `false` | TTA nos recortes do `/predict/rover-frame` (`?tta=` por requisição). Melhora o recorte (~1 a 2 pontos no campo) mas piora a planta, ver o rover abaixo. |
| `ROVER_SPECIES_CHECK` | `false` | Verificação de espécie (`<pesos>.species.npz`) nos recortes do rover. Desligada: no talhão ela recusou 49% dos recortes de citros e piorou a planta — ver a [segunda leitura](#segunda-leitura-cabeças-sobre-o-embedding-do-portão). |
| `ROVER_REVIEW_PER_FRAME` | `3` | Recortes de cada quadro que vão para a fila de revisão (precisa de `REVIEW_DIR`). `0` desliga. |
| `INFERENCE_RUNTIME` | `keras` | `onnx` serve pelos `.onnx` do `export_onnx`, sem Keras nem torch. |
| `MAX_FORWARD_BATCH` | `32` | Teto de imagens por passada. Acima da VRAM, o driver pagina e a máquina trava. |
| `MIN_CONFIDENCE` | `0.0` | Abaixo disso a resposta vem como `abstained`. **Recomendado `0.7` em produção** — ver abaixo. |
| `NOVELTY_SENSITIVITY` | `1.0` | Regula o volume da fila de revisão; menor = mais imagens marcadas. |
| `REVIEW_DIR` | — | Onde arquivar as imagens marcadas. Vazio = fila desligada (só marca `needs_review` na resposta). |
| `MAX_UPLOAD_MB` | `10` | Tamanho máximo por imagem. |
| `MAX_BATCH_SIZE` | `20` | Imagens por requisição em `/predict/batch`. |
| `CORS_ORIGINS` | `localhost:5173,localhost:3000` | Origens do front, separadas por vírgula. |
| `LOG_LEVEL` | `INFO` | Nível de log. |
| `CLIMA_OPEN_METEO` | `false` | Deixa o gêmeo buscar histórico e previsão de tempo no Open-Meteo (`POST /fields/{id}/clima` com `buscar_open_meteo`). Desligado, a API não chama nada de fora: o tempo vem da estação. |

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

## Qualidade da captura

A novidade diz que uma imagem é estranha, mas não diz **por quê** — e uma foto
borrada, escura ou estourada pede outra ação: pede outra foto, não um agrônomo.
O rover pode refazer a captura na hora; o agrônomo não diagnostica o que não
enxerga.

Cada previsão traz `quality`, com os problemas encontrados (`blurry`, `dark`,
`overexposed`, `low_contrast`) e o que fazer com cada um. A régua não é um
número de tutorial: é o **próprio dataset de treino do modelo**. Uma foto é
"borrada" quando é mais borrada que 99,5% das fotos com que ele aprendeu — fora
dessa faixa, a resposta dele é extrapolação, seja qual for a confiança. E tudo é
medido **na entrada do modelo** (224×224): desfoque que some ao reduzir uma foto
de 12 MP não importa para ele, e não é marcado.

Medido em 26/09/2026 no dataset consolidado de cada modelo — 19.787 imagens no
milho, 3.462 no café —, calibrando em 80% e medindo nos 20% que ficaram de fora.
As referências estão gravadas ao lado dos pesos (`*.quality.json`):

| | `leaf-foliar` | `coffee-foliar` |
| --- | ---: | ---: |
| alarme falso em foto boa | **2,0%** | 3,2% |
| desfoque de 1,5 px | 70,7% | 14,5% |
| desfoque de 2 px | **98,3%** | 18,2% |
| brilho ×0,3 | 52,2% | 18,9% |
| brilho ×2,2 | **96,8%** | 93,1% |
| contraste ×0,2 | 91,3% | 28,0% |

Três leituras que importam:

- **No milho, desfoque e estouro são pegos quase sempre; escuridão, só metade.**
  O limite inferior de brilho ficou em 3,7 numa escala de 255: 0,5% das fotos de
  treino do milho são praticamente pretas. Para o modelo, foto escura é normal —
  ele aprendeu com elas. É um defeito do dataset, não da verificação, e está
  detalhado em [imagens quase pretas](#imagens-quase-pretas-no-dataset).
- **O custo cai sobre a classe mais difícil**: `Gray_Leaf_Spot` tem 4,4% de
  alarme falso, contra 1,5% a 2,0% nas outras. A régua é uma só para todas as
  classes, porque a verificação acontece antes de saber qual é.
- **No café, a verificação quase não pega nada — e isso é um achado, não um
  defeito.** O JMuBEN é 128×128, ampliado para 224: o dataset de treino inteiro já
  é borrado e mal exposto na entrada do modelo. Ele aprendeu com fotos ruins,
  então uma foto ruim não está fora do que ele conhece. A verificação responde
  "isto está fora do que o modelo aprendeu?", e para o café a resposta honesta é
  que o modelo aprendeu pouco sobre qualidade de captura. É mais um motivo para a
  coleta própria.

A cor chapada — o caso que escapava tanto da confiança quanto da novidade — tem
contraste zero e é sempre marcada, de qualquer cor.

`quality` não mexe em `abstained` nem em `needs_review`: são perguntas
diferentes. A recomendação para quem chama é não mostrar diagnóstico com
`quality.ok` falso, e pedir outra foto.

## TTA em produção

As acurácias deste README e do [MODELS.md](MODELS.md) são **com TTA**: cada
imagem prevista em quatro orientações e a média das quatro. Até aqui, a API não
fazia TTA — a produção entregava a linha de baixo:

| | Sem TTA | Com TTA |
| --- | ---: | ---: |
| `leaf-foliar` | 0,9498 | **0,9555** |
| `leaf-foliar-b` | 0,9535 | **0,9601** |
| `coffee-foliar` | 0,9441 | **0,9499** |

Agora `?tta=true` (ou `INFERENCE_TTA=true`) aplica exatamente as quatro
orientações do `training.evaluate`, empilhadas num lote só: é uma passada pelo
modelo, com quatro vezes as imagens. Custa 4× em processamento, e por isso não é
o padrão.

A novidade continua pontuada **só na orientação original**. A referência foi
calibrada com imagens não espelhadas; tirar a média dos quatro embeddings mudaria
a distribuição que o limiar mede, e uma imagem estranha poderia deixar de ir para
o agrônomo só porque o TTA foi ligado. Há teste para isso.

De quebra, vem `view_agreement`: a fração das quatro orientações que votou na
classe final. Uma folha espelhada tem a mesma doença, então abaixo de 1,0 o
modelo mudou de ideia só porque a foto girou. **Não é segunda opinião
independente** — é o mesmo peso olhando de novo, e na convenção da plataforma
conta como `dependent:`.

Medido em 30/09/2026, no teste isolado com TTA: quando as quatro vistas concordam o
modelo acerta de 95,2% (citros) a 99,1% (banana); quando discordam, de 60,5% a 70,4%.
A discordância sozinha marca de 4,5% (banana) a 21,2% (citros) das imagens e pega **58%
dos erros no milho, 63% na banana, 69% no citros e 31% no café** — o café é o fraco:
com o dataset de 128×128 ampliado, girar a foto quase não muda a resposta.

<a id="onde-o-modelo-olhou"></a>
## Onde o modelo olhou

`POST /predict/explain` cobre a imagem célula por célula com cinza, pergunta de
novo ao modelo, e mede quanto a confiança cai em cada uma. O resultado é o mapa
de **quanto cada região sustenta a resposta**, e a foto com esse mapa pintado de
vermelho:

```bash
curl -F "file=@folha.jpg" "localhost:8000/predict/explain?grid=7"
curl -F "file=@folha.jpg" "localhost:8000/predict/explain?target_class=Gray_Leaf_Spot"
curl -F "file=@folha.jpg" "localhost:8000/predict/explain?method=rise"
```

O segundo responde "por que **não** cercosporiose?" — útil quando o agrônomo
discorda do modelo.

Três escolhas de desenho, todas pelo mesmo motivo — não sugerir mais do que o
método sabe:

- **Oclusão, não Grad-CAM.** Grad-CAM exige gradiente, e gradiente no Keras 3 é
  código de backend. A oclusão só precisa de previsões: funciona em torch, em
  TensorFlow, com qualquer arquitetura do catálogo. E o número tem unidade:
  "cobrir esta região derruba a confiança de 0,97 para 0,31".
- **Intensidade absoluta.** Vermelho pleno é uma região que, coberta, zera a
  confiança. Normalizar pelo máximo de cada imagem faria um mapa sem nenhuma
  região decisiva parecer tão quente quanto um com a lesão marcada.
- **Blocos, sem suavizar.** Interpolar a grade sugeriria uma resolução que a
  oclusão não tem.

Custa `grid² + 1` passadas — 50 na grade padrão. É ferramenta de console e de
auditoria, não de triagem de voo, e nada do que passa por ela entra na fila de
revisão.

### Quando a oclusão fica cega, e o RISE

Medido em 26/09/2026: na auditoria do `leaf-foliar`, cobrir **qualquer** célula
sozinha mexeu menos de 5 pontos na confiança em 96% das folhas de
`Common_Rust` — queda mediana de 0,000. Não era atalho: era a ferrugem
espalhando pústulas pela folha inteira. Tirada uma, as outras sustentam a
resposta, e a oclusão de uma célula por vez não vê nada. É o padrão de quase
toda doença foliar.

`?method=rise` (Petsiuk et al., 2018) cobre **metade** da imagem ao acaso,
centenas de vezes (`samples`, padrão 256). Com metade das lesões escondidas, as
que sobram deixam de bastar, e a contribuição de cada região aparece na média —
na mesma unidade: quanto a confiança é maior, em média, com a célula visível do
que coberta. Nas mesmas 100 folhas, "sem região" caiu de 44% para 4% em
`Blight`, de 52% para 4% em `Gray_Leaf_Spot`, de 76% para 8% em `Healthy`. Só a
ferrugem comum, a mais difusa, ficou alta (76%).

<a id="a-auditoria-que-teria-pegado-o-rover-understory"></a>
### A auditoria que teria pegado o `rover-understory`

```bash
python -m training.audit_attention --model leaf-foliar --per-class 25 --out reports/atencao
```

Para cada classe, o `audit_attention` mede duas coisas:

- **sem região** — em quantas imagens nenhuma parte da foto sustenta a resposta
  sozinha (RISE, por padrão). As sobreposições vão para `--out`;
- **só a cor** — a imagem é trocada por um quadro chapado com a **cor média
  dela**, sem lesão, forma ou textura; conta quantas vezes o modelo mantém a
  resposta com confiança ≥ 0,9. Não há doença num quadro chapado: um valor alto
  é atalho, em qualquer classe.

Medido em 26/09/2026, 25 imagens por classe:

| Modelo | Classe com "só a cor" alto | Leitura |
| --- | --- | --- |
| `rover-understory` (quarentena) | `Healthy`: **100%** | **o caso de controle, denunciado** |
| `coffee-foliar` | `Rust`: **92%** | atalho — ver abaixo |
| `leaf-foliar` | nenhuma: 0% em todas | sem atalho de cor |

O teste da cor é o que discrimina. O mapa sozinho não bastaria: "sem região" alto
aparece tanto no atalho quanto na doença difusa.

### O atalho do café: "sem textura" quer dizer `Rust`

Uma varredura de 936 cores chapadas — verdes, amarelos, marrons, cinzas — deu
**936 respostas `Rust` com confiança ≥ 0,9**. O mesmo para o quadro chapado de
cada fonte e cada classe do dataset. Para o `coffee-foliar`, imagem sem textura
é ferrugem. É o que explica o caso documentado mais adiante: o cafezal visto de
drone, onde nenhuma folha se resolve, vira `Rust` em 94% dos quadros.

A causa está nos dados: **150 das 1.540 imagens de `Rust` são máscaras de
segmentação do PlantSeg** — fundo preto com polígonos cinza, não fotos. Ver
[máscaras no dataset](#imagens-quase-pretas-no-dataset).

<a id="predição-conformal"></a>
## Predição conformal: um conjunto com garantia

A confiança de um softmax não é uma probabilidade calibrada: o modelo diz 0,93 e
erra. Quando `Blight` e `Gray_Leaf_Spot` se confundem — e se confundem, são o par
dos 88 grupos de rótulo contraditório —, forçar uma resposta só produz um erro
confiante.

`prediction_set` troca a pergunta. Em vez de "qual é a doença?", responde "quais
doenças **não dá para descartar**?", com uma garantia que não depende de o modelo
estar bem calibrado:

    P(classe verdadeira ∈ conjunto) ≥ 1 − α, para cada classe

Os limiares são calibrados **por classe** (conformal de Mondrian): sem isso a
garantia valeria só na média, e a classe difícil pagaria pela fácil.

```bash
python -m training.fit_conformal --model leaf-foliar --test-split 0.15   # α = 0,10
```

Calibra na validação e mede no teste isolado, que nem o treino nem a calibração
viram. Grava `models/<pesos>.conformal.json`, com limiares separados para as
previsões com e sem TTA — as duas têm distribuições diferentes.

| Conjunto | Leitura | O que fazer |
| --- | --- | --- |
| uma classe | resposta firme, com a garantia | seguir |
| duas ou mais | a dúvida real do modelo, dita com precisão | agrônomo decide **entre essas** |
| vazio | nenhuma classe alcança o nível pedido | inconclusivo |

Medido em 26/09/2026, com TTA, no teste isolado — o pré-processamento já idêntico
ao do treino e as máscaras do PlantSeg fora (ver
[máscaras no dataset](#imagens-quase-pretas-no-dataset)):

| α = 0,10 | `leaf-foliar` | `leaf-foliar-b` | `coffee-foliar` |
| --- | ---: | ---: | ---: |
| cobertura (pedida: 90%) | 88,5% | 89,6% | **91,3%** |
| pior classe | `Gray_Leaf_Spot` 84,1% | `Common_Rust` 86,4% | `Rust` 89,4% |
| resposta única | 89,8% | 91,6% | 94,9% |
| **acerto quando é resposta única** | **98,5%** | **97,7%** | **96,2%** |
| conjunto vazio (inconclusivo) | 10,2% | 8,4% | 5,1% |
| erros do top-1 que viram conjunto ≠ 1 | **67,5%** | 46,7% | 30,8% |

**Por que α = 0,10, e não 0,05.** Num modelo que já acerta 95%, α = 0,05 quase só
devolve resposta única: na primeira rodada sinalizava 37% dos erros do
`leaf-foliar`. E diante de uma doença **nunca vista** — o par de 3 classes
treinado sem `Gray_Leaf_Spot`, diante de 400 imagens dela —, α = 0,05 interceptou
5% e α = 0,10 interceptou **33,5%**, sozinho; somado ao `MIN_CONFIDENCE`, 44,8%.
Para comparação, a novidade pegava 3,5% desse caso.

**A garantia ficou abaixo do pedido no milho**, e isso é informação, não ruído: a
calibração usa a validação, que já serviu para escolher a época do treino — o
modelo foi escolhido por ir bem nela, e os limiares saem otimistas. No café
(91,3%) a garantia se sustentou. A correção é separar, no próximo retreino, uma
fatia que sirva **só** para calibrar.

**A garantia tem uma condição**: vale enquanto as fotos de produção vierem da
mesma distribuição da calibração. Calibrado em dataset público, garante cobertura
em dataset público. Em campo, remeça com imagens da frota — com a vantagem de que
aqui a ressalva tem nome e fórmula.

**A ressalva se confirmou na banana.** No teste de campo do PlantSeg, o conjunto
conformal da banana cobria 56–59% (pedido: 90%) e o da versão comercial, 23–27%. O
`fit_conformal --field <recortes de campo>` alarga os limiares: divide os recortes
**por foto** ao meio, calibra numa metade, mede na outra e fica com o **maior** dos
dois limiares em cada classe presente no campo (classe sem exemplo no campo mantém
o de antes). Medido em 01/10/2026, na metade dos recortes que a calibração não viu:

| | Cobertura no campo, sem / com TTA | Resposta única no campo | Cobertura no teste isolado |
| --- | --- | ---: | ---: |
| `banana-foliar` antes | 58,7% / 56,8% | 73% / 70% | 89,4% / 89,8% |
| `banana-foliar` depois (154 recortes calibram) | **87,7% / 86,5%** | 33% / 72% | 93,7% / 93,3% |
| `banana-foliar-cc` antes | 25,6% / 22,4% | 47% / 36% | 89,1% / 88,8% |
| `banana-foliar-cc` depois (631 recortes calibram) | **90,1% / 92,9%** | 11% / 8% | 96,7% / 96,2% |
| `citrus-foliar-cc` antes | 71,6% / 74,1% | 75% / 77% | 89,6% / 89,4% |
| `citrus-foliar-cc` depois (1.008 recortes calibram) | **91,5% / 90,1%** | 52% / 58% | 93,7% / 92,6% |

Conferido de ponta a ponta pela API nos recortes de teste inteiros, que nenhuma das
duas versões comerciais viu na calibração: banana 27,2% → **93,5%** e citros 73,2% →
**90,8%** de cobertura no campo. O preço é conjunto com duas ou mais classes onde o
modelo duvida, que é a resposta certa: na banana comercial, que acerta 59% no campo, só
4% das respostas de campo ficam com uma classe (antes 37%). E o **conjunto vazio**, que
era um alarme de entrada estranha ("nenhuma classe alcança o nível pedido"), fica mais
raro: de 9,3% para 4,5% no teste isolado da banana, e nas folhas do PlantVillage mandadas
ao modelo de banana a fração que passa sem nenhum alarme subiu de 21,7% para 25,7%. A versão recomendada ficou 2 a 3 pontos abaixo do pedido porque só 154 recortes
calibram. Três ressalvas: os recortes de campo só têm lesões (a classe `Healthy`
continua calibrada em dataset público); na versão recomendada a metade que calibra
sai do conjunto de teste de campo, que deixa de ser independente para o conformal (a
comerciais calibram com os recortes de **treino** do PlantSeg, que elas nunca viram); e
o `citrus-foliar` não precisou, porque já cobria 95–97% no campo.

**A fusão por planta não recebe o conjunto honesto.** A fusão divide a evidência de cada
recorte pelo tamanho do conjunto, e essa divisão foi ajustada com os conjuntos da
calibração original. Só alargar derrubou a detecção da banana no talhão de ponta a
ponta de **9 para 7 de 18** plantas doentes (a posterior de Sigatoka das plantas doentes
caiu ~3×), sem mudar um único recorte aceito pelo portão; trocando o portão e o
conformal separadamente (2×2), o portão não faz diferença e o conformal explica tudo. Por
isso o JSON guarda os dois: o `prediction_set` da resposta é o honesto (alargado), e o
que vai para a fusão é o da calibração original (`base_thresholds`,
`CornDiseaseModel.fusion_prediction_sets`). Com isso a banana volta a 9 de 18, o citros
fica em 29 de 33 e a prescrição sai igual à de antes de alargar.

<a id="enquadramento"></a>
## Enquadramento: a foto é o que o modelo sabe ler?

É o caso mais perigoso que o projeto mediu, e nenhum filtro existente o pegava:
uma foto **aérea** de cafezal vira `Rust` em 94% dos quadros, com confiança média
de 0,93, e só 9% eram interceptados. O classificador só conhece folhas; para
saber que uma foto **não é** de folha é preciso um modelo que conheça o resto do
mundo.

`framing` usa o **SigLIP 2** (Google, 2025), um modelo de visão-linguagem treinado
em bilhões de pares imagem-texto, em modo *zero-shot*: a foto é comparada com
frases — "a close-up photo of a maize leaf", "an aerial drone photo of a farm
field", "a slide or document with text", "a photo of a plant stem" — e nenhuma
imagem nossa precisou ser rotulada. As frases foram escritas **antes** de olhar
as imagens de avaliação, a partir dos modos de falha já documentados.

O limiar, esse sim, vem dos nossos dados: calibrado **só com fotos boas** (a
validação dos modelos foliares), no percentil que decide quantas folhas boas
podem ser recusadas. As fotos fora de escopo servem só para medir.

```bash
python -m training.fit_framing --negatives <pastas fora de escopo> --positives <pastas que devem passar>
```

Medido em 26/09/2026, com o limiar no percentil 1 das fotos boas:

| | Resultado |
| --- | ---: |
| folha boa recusada — milho / café (teste) | 0,5% / 1,7% |
| cultura certa — milho / café | 98,6% / 90,8% |
| **drone sobre cafezal** (antes: 91% passava) | **79 de 79 recusadas** |
| folha de café com deficiência de magnésio (deve passar) | 95% passa |
| colmo (rover, 1.582 fotos) | 54,7% recusadas |
| slides de aula com fotos de folha | 7,2% recusados |

O caso mais perigoso do projeto está resolvido. Colmo e slide, não: o colmo é
reconhecido como `colmo` em 98% das fotos, mas o limiar ficou baixo demais —
0,032 — porque a calibração tinha 85 máscaras pretas do PlantSeg contadas como
"folhas boas" (ver [máscaras no dataset](#imagens-quase-pretas-no-dataset)). O
`fit_framing` agora as exclui e imprime a curva de limiares; a recalibração ficou
pendente. Slide com foto de folha continua sendo lacuna: para o modelo de visão,
aquilo **é** uma folha.


De quebra, `framing.crop` diz a cultura da folha, e `crop_matches` avisa quando
uma folha de café foi mandada ao modelo de milho — que responderia uma das quatro
doenças de milho sem nada na resposta denunciar o engano.

**As cenas das zonas baixas do rover.** Em 27/09/2026 entraram seis cenas, tiradas
das [zonas do rover](#rover-do-quadro-à-zona-da-planta) e não de erros medidos:
`folha_banana` e `folha_citros` (no escopo, com a cultura) e `pseudocaule`,
`tronco`, `fruto_citros` e `corte` (fora). É o que impede o recorte do pé da
bananeira ou do tronco do citros de receber uma doença de folha.

Recalibrado em 27/09/2026 com as cenas novas, no percentil 1 das 9.012 fotos boas
dos quatro modelos foliares (limiar 0,066):

| | Folha boa recusada (teste) | Cultura certa |
| --- | ---: | ---: |
| banana | 1,7% | 96,0% |
| citros | 0,1% | 99,0% |
| milho | 0,0% | 92,3% (antes 98,6%) |
| café | 1,4% | **56,9%** (antes 90,8%) |

O preço das cenas novas está na última linha: folha de café, lisa e escura, é
vista muitas vezes como folha de citros quando o portão tem de adivinhar a
cultura. A correção foi não pedir que ele adivinhe: com o modelo escolhido, a
foto só disputa com a folha daquela cultura — ver [escolhendo o
modelo](#escolhendo-o-modelo). O diagnóstico nunca mudou por isso: `crop` e
`crop_matches` são só aviso.

Nos recortes de campo do PlantSeg, o portão fez o que devia: tirou da banana 14
recortes de pseudocaule, 8 de corte transversal e 6 de tronco (quase todos da
Fusariose), e do citros 18 de fruto — sem recusar nenhum recorte de Cordana.

**Inseto e folha de outra espécie (01/10/2026).** O ensaio de ponta a ponta mostrou
que o modelo de citros aceitava, sem alarme, 21% das fotos de inseto e 35% das
folhas de outras espécies (PlantVillage). Entraram duas cenas fora do escopo,
`inseto` e `folha_outra`: as frases foram escritas antes da medição, a escolha entre
10 combinações usou só metade das espécies e dos ids de inseto, e o que vale é a
outra metade. O inseto passou a ser recusado em **71 a 78%** (antes 41 a 51%), sem
custo em folha boa nem em recorte de campo. A folha de outra espécie **quase não
mudou onde importa**: o citros recusa 1,3% das folhas do PlantVillage (antes 0,3%).
A visão-linguagem não separa "folha de citros" de "folha de outra planta" por texto.
Comparar com fotos reais da cultura separa, mas só com referência de **campo**: com
folhas sobre papel, ou recusa 99,8% dos recortes do rover ou deixa passar 99,8% das
outras espécies; com recortes de campo na referência barrou 96% dos insetos, 63% das
folhas do PlantVillage e 93% das do PlantSeg, recusando 8,8% dos recortes de campo
não vistos. Não foi adotado, porque os recortes de campo que temos são CC BY-NC: a
referência certa é a do próprio rover, e é a fila de revisão que a produz.

**A verificação de espécie (01/10/2026)** resolve isso no `citrus-foliar`, cuja licença
já é não comercial: uma cabeça binária sobre o mesmo embedding ([segunda
leitura](#segunda-leitura-cabeças-sobre-o-embedding-do-portão)), treinada com folhas de
citros (papel + recortes de treino do PlantSeg) contra folhas de outras plantas
(PlantDoc, CC BY 4.0, 13 espécies em campo; banana; café), com o limiar fixado antes no
ponto que recusa 2% das folhas de citros da validação. Nos conjuntos que ela não viu:

| Recusado pelo portão | Hoje | Com a espécie |
| --- | ---: | ---: |
| recortes de citros no campo (PlantSeg, teste) | 3,8% | 6,1% |
| folhas de outras espécies — PlantVillage | 0,9% | **96,9%** |
| folhas de outras espécies — PlantSeg | 22,4% | **99,5%** |
| PlantDoc (fotos de teste) | 11,9% | **100%** |
| insetos (IP102) | 71,3% | **95,7%** |
| **folhas de laranja do PlantVillage** (citros!) | 0% | **48%** |

A última linha é o custo: folha de laranja fotografada em laboratório, sobre fundo cinza,
é uma fonte de citros que o treino nunca viu, e metade é tomada por outra planta. No
rover essa foto não existe; num upload manual, existe.

**No rover ela fica desligada** (`ROVER_SPECIES_CHECK=false`). O recorte do rover não é
o recorte de lesão do PlantSeg em que ela foi medida: é pedaço de folha, borda, fundo.
No talhão de ponta a ponta, ligada, ela recusou **49%** dos recortes de citros (9.087 de
18.565) como `folha_outra`, e a fusão por planta perdeu metade da evidência: com a mesma
cabeça de diagnóstico, 12 das 33 plantas sadias viraram alarme com ela no rover, e
nenhuma sem ela. No pomar, o rover já sabe a cultura da fileira; quem precisa da
verificação é a foto avulsa.

**O `citrus-foliar-cc` (comercial) continua sem verificação de espécie**, porque o dado
com licença comercial não permite. As únicas folhas de citros CC BY são fotos de
laboratório, sobre papel. Treinada com elas, a cabeça aprendeu "papel = citros" e
recusou **100%** dos recortes de citros do campo. Uma segunda tentativa tirou o papel
da conta — o miolo da folha no lugar da foto inteira, e folhas de café do BRACOL (CC BY
4.0) sobre fundo branco como negativo, para o papel deixar de indicar citros —, e ainda
recusou **96 a 99%** dos recortes de campo. Não é o fundo: nada no dado licenciado
mostra citros na árvore. A referência que falta é a do próprio rover, e a fila de
revisão a está juntando.

**Em produção vai só a torre de visão.** O modelo completo tem 375 M de
parâmetros; as frases são sempre as mesmas, então o `fit_framing` grava os
embeddings delas em `models/framing.npz` e a API carrega só a torre de visão
(92 M), que reproduz o embedding do modelo inteiro com diferença menor que 1e-5.

## Segunda leitura: cabeças sobre o embedding do portão

O portão já passa **toda** foto e todo recorte do rover pela torre de visão do SigLIP 2.
Desde 01/10/2026, cada modelo foliar pode ter duas redes pequenas sobre esse mesmo
embedding ([`app/inference/heads.py`](app/inference/heads.py)) — opcionais e por arquivo
de pesos, como a calibração conformal. A torre já rodou; a rede custa uma multiplicação
de matrizes.

| Arquivo | Pergunta | O que muda na resposta |
| --- | --- | --- |
| `<pesos>.head.npz` | qual das classes do modelo? | `probabilities` vira a média geométrica ponderada da CNN com a cabeça (pesos no arquivo) |
| `<pesos>.species.npz` | a folha é da cultura do modelo? | abaixo do limiar, `framing.in_scope` vira falso, com `scene: folha_outra` e `species_score` |

A arquitetura de cada rede sai do **dNATY** (`dnaty.compress`): busca evolutiva sobre
largura, profundidade, ativação e saltos do MLP, escolhida pela validação, com ajuste fino
no fim. O treino usa as quatro orientações do TTA como aumento. A rede vai exportada em
matrizes e roda em NumPy: a API não importa torch nem dNATY para usá-la, e a borda (portão
em ONNX) também a roda. Só o treino depende do dNATY — ver
[`requirements-training.txt`](requirements-training.txt) para a instalação e a licença.

```bash
python -m training.import_bracol --zip ~/valtexa-raw/bracol/yy2k5y8mxg-v1.zip \
    --output data/coffee-bracol
python -m training.fit_heads --model coffee-foliar --extra data/coffee-bracol \
    --mix cnn=0,head=1 --holdout data/coffee-bracol-holdout
# a resposta mudou: recalibre, alargando com as folhas do BRACOL que a cabeça não treinou
python -m training.fit_conformal --model coffee-foliar --field data/coffee-bracol-holdout
```

**O que ela resolveu (medido em 01/10/2026, pela conta da API):**

| | Só a CNN | Com a segunda leitura |
| --- | ---: | ---: |
| café — 542 fotos de outra fazenda (Carneiro et al. 2021, só medida) | 6,5% | **80,1%** |
| café — teste isolado / teste do BRACOL (nunca treinado) | 0,950 / 0,313 | 0,964 / 0,914 |
| citros — folha de cabeça para baixo (espelho vertical), sem TTA | 0,819 | **0,937** |
| citros — 511 recortes de campo, sem TTA (como o rover) | 0,951 | **1,000** |
| citros comercial — espelho vertical / campo com TTA | 0,806 / 0,693 | 0,926 / 0,798 |
| citros — folhas de outras espécies recusadas na foto avulsa (PlantVillage) | 0,9% | **96,9%** |

**E na planta, pelo rover.** O talhão de ponta a ponta (33 plantas de citros doentes e
33 sadias, 3 câmeras cada, pela API real, sem TTA) é o que decide, porque a fusão por
planta ([`app/plant_fusion.py`](app/plant_fusion.py)) foi ajustada com saídas da CNN:

| Talhão de citros | Só a CNN | Cabeça + espécie no rover | **Cabeça** (espécie fora do rover) |
| --- | ---: | ---: | ---: |
| plantas doentes detectadas (doente ou suspeita) | 29 | 27 | 27 |
| doença principal certa | 30 | 32 | 32 |
| alarme falso em planta sadia | 6 | 12 | **0** |
| plantas tratadas pela prescrição (sadias entre elas) | 50 (17) | 36 (4) | **34 (2)** |
| calda | 10,0 L | 7,2 L | **6,8 L** |

Com a cabeça, nenhuma planta sadia vira alarme e a calda cai 32%, mas duas plantas
doentes a menos são detectadas (o cancro cai de 25 para 20 de 25 como doente ou
suspeita; o greening sobe de 4 para 7 de 8), e a prescrição trata 32 das 33 doentes em
vez das 33. É uma troca de sensibilidade por precisão, melhor no balanço (acurácia
balanceada 0,85 → 0,91), com uma amostra pequena: dois votos. A fusão continua ajustada
para a CNN — recalibrá-la para a resposta nova é o mesmo item aberto do TTA no rover. A
banana não tem cabeça e saiu idêntica (9 de 18, 1 alarme falso, mesma prescrição). O
tempo por quadro foi de 1,3 para 1,6 s nesta rodada, mas a cabeça custa 3,6 ms por 128
recortes e a banana, que não mudou, ficou 23% mais lenta na mesma rodada: foi a máquina.

**Como foi escolhido.** As regras foram fixadas antes de medir: só o split de treino
treina; a validação escolhe a arquitetura e o peso da mistura; o teste isolado, o campo
e as fotos externas só medem. Em cada problema a cabeça do dNATY foi comparada com uma
sonda linear e com o MLP de partida sem busca, sobre a mesma entrada. Na validação o dNATY
empata ou ganha com 31 a 39% menos FLOPs que o MLP de partida; **fora da fonte do treino,
uma cabeça mais simples às vezes generaliza melhor** (café externo: linear 90%, MLP sem
busca 82%, dNATY 78%; citros comercial no campo: MLP sem busca 0,92, dNATY 0,80). A regra
foi mantida: escolher pelo conjunto externo o tornaria um conjunto de ajuste, e ele é o
único teste independente que o café tem.

**O que muda em volta:**

- **Recalibre a predição conformal** depois de instalar uma cabeça: a resposta é outra. O
  `fit_conformal` carrega o portão sozinho quando há cabeça, e recusa calibrar sem ele.
- Sem portão (sem `framing.npz` ou sem a torre), a CNN responde sozinha e a cabeça fica
  de fora — e a calibração feita com ela deixa de valer. `/models` mostra
  `safeguards.diagnosis_head` e `safeguards.species`, verdadeiros só quando a cabeça é da
  mesma torre de visão que o portão está usando.
- O mapa de oclusão/RISE continua explicando a **CNN**; com cabeça, a base das quedas é a
  probabilidade da CNN sozinha, para a conta fechar.
- `view_agreement` (TTA) continua medindo a CNN. No café, que usa só a cabeça, o TTA não
  muda mais a resposta.
- A novidade continua no embedding da CNN.

## Rover: do quadro à zona da planta

O rover (Rev E, hardware em `VALTEXA-SAAS/hardware/rover-v3`) não manda folha
recortada: manda o quadro de 12 MP (4056×3040) de uma das seis OAK-D Pro W
laterais, montadas a **450, 1.000 e 1.540 mm** do chão, com campo de visão de
127° na horizontal e 110° na vertical, a ~270 mm das plantas. Os modelos
foliares aprenderam com fotos em que a folha ocupa o quadro. As duas coisas não
se encontram sem uma etapa no meio — e é ela que [`app/rover.py`](app/rover.py)
faz:

1. **Escala física.** A distância até a planta (a mediana da profundidade da
   OAK-D) dá o tamanho do pixel no alvo: a 270 mm, 0,27 mm por pixel. O quadro
   é fatiado em recortes de **150 mm** — um pedaço de limbo na escala das fotos
   de treino —, com 25% de sobreposição. Quanto mais longe a planta, menos
   pixels por recorte; abaixo de 112 px a requisição é recusada, porque o
   recorte já não tem resolução para lesão.
2. **Altura de cada recorte.** Pela altura da câmera e pelo modelo pinhole, cada
   recorte sabe a que altura do chão está — e, com isso, **em que zona da
   planta**:

   | Cultura | Zona | Altura | Por quê |
   | --- | --- | --- | --- |
   | banana | `pseudocaule` | 0–1.200 mm | rachadura na base e folhas pendendo: a Fusariose aparece aqui |
   | banana | `folhas_baixas` | 900–2.500 mm | folhas velhas, onde a Sigatoka é mais severa |
   | citros | `tronco_e_saia` | 0–500 mm | tronco, gomose, frutos da saia |
   | citros | `copa` | 350–2.500 mm | folhas da parte baixa da copa |

3. **Portão por recorte.** Cada recorte passa pelo [enquadramento](#enquadramento):
   o que não é folha em close — pseudocaule, tronco, fruto, corte — **não recebe
   diagnóstico**. Sem isso, um pseudocaule sadio sairia como "Fusariose" com
   confiança alta, porque o modelo foliar sempre responde uma doença de folha.
   Esses recortes são contados à parte, por zona e por cena.

```bash
curl -F "file=@quadro.jpg"   "http://localhost:8000/predict/rover-frame?model=banana-foliar&camera_height_mm=1000&distance_mm=270"
```

```json
{
  "model_key": "banana-foliar",
  "crop": "banana",
  "frame_size": [4056, 3040],
  "tile_px": 562,
  "tiles_analyzed": 70,
  "class_counts": {"Healthy": 36, "Sigatoka": 4},
  "zone_summary": {
    "pseudocaule":   {"recortes": 50, "com_achado": 0, "fora_de_escopo": 30},
    "folhas_baixas": {"recortes": 40, "com_achado": 4, "fora_de_escopo": 0}
  },
  "out_of_scope": {"pseudocaule": {"pseudocaule": 28, "solo": 2}},
  "tiles": [
    {"box": [1266, 0, 1828, 562], "height_mm": 1314.0, "zones": ["folhas_baixas"],
     "predicted_class": "Sigatoka", "confidence": 0.94, "prediction_set": ["Sigatoka"],
     "needs_review": false, "in_scope": true, "scene": "folha_banana"}
  ]
}
```

A geometria é a real (câmera do meio, a 270 mm: 70 recortes de 562 px, 50 na
zona do pseudocaule e 40 na das folhas baixas); as classes são ilustrativas. `tiles` traz só os recortes com achado — doença, conjunto
conformal ambíguo ou imagem estranha ao modelo; `only_findings=false` devolve
todos.

**TTA no rover: opcional, desligado.** `ROVER_TTA=true` (ou `?tta=true`) prevê cada
recorte nas quatro orientações. Melhora o recorte: no teste de campo de 30/09/2026 a
banana vai de 0,812 para 0,822 e o citros de 0,951 para 0,969. Mas **piora a planta**:
no talhão de ponta a ponta de 01/10/2026 (66 plantas de citros, 3 câmeras cada), as
plantas doentes detectadas caíram de 29 para 24 de 33, com os mesmos 6 alarmes falsos
em 33 (e o quadro leva 2,3 s em vez de 1,3 s). A média de quatro vistas dá
probabilidades menos extremas, e a fusão por planta foi ajustada com saídas de uma
passada só, então recebe menos evidência por recorte. Para ligar o TTA no rover é
preciso recalibrar a fusão antes. Em compensação, **com** TTA a resposta é idêntica
para a folha em pé e de cabeça para baixo (ver a orientação no
[MODELS.md](MODELS.md)); sem ele, o citros perde ~6 pontos com a folha invertida.

**Memória do computador.** A passada do portão é fatiada em blocos de
`MAX_FORWARD_BATCH`, e as chamadas ao classificador em fatias de 128 recortes: a torre
de visão segura as ativações do lote inteiro (~13,5 MB por recorte) e o TTA empilha
quatro vistas, de modo que um quadro a 1 m (~800 recortes) pedia ~11 GB de uma vez. Com
as fatias, um quadro de 828 recortes na borda, em CPU, fecha em 3,0 GB com respostas
idênticas às sem fatia.

**Do quadro para a fila de revisão.** Com `REVIEW_DIR` ligado, até
`ROVER_REVIEW_PER_FRAME` (3) recortes de cada quadro são arquivados com de onde
vieram (talhão, rua, lado, posição, câmera, altura, zona e cena): primeiro o
estranho ao modelo, depois o de conjunto conformal ambíguo, e **no máximo um** dos
que o portão recusou — no pseudocaule quase todo recorte é recusado, e eles
inundariam a fila. Antes disso a fila só recebia imagens do `/predict`: o rover
mandava milhares de recortes e nenhum virava dataset próprio, que é justamente o
que deve substituir os recortes do PlantSeg (licença não comercial). A resposta traz
`queued_for_review`.

**O que o rover não lê hoje: o pseudocaule.** Não existe dataset público de
doença de pseudocaule na escala do rover — o que há de "caule" nos datasets de
Fusariose é o **corte transversal** com a descoloração interna, que a câmera
nunca vê. Os recortes da zona `pseudocaule` que não mostram folha vão para
`out_of_scope`; com a [fila de revisão](#ciclo-de-dados-próprio) ligada no rover,
são eles que viram o dataset de um futuro modelo de pseudocaule.

O mesmo código roda na Jetson Orin NX do rover com `INFERENCE_RUNTIME=onnx`
(ver [ONNX Runtime](#servindo-sem-torch-onnx-runtime)): a API e o robô não têm
duas implementações do mesmo cálculo.

## Talhão: da foto à planta, e da planta ao mapa

Um app de celular classifica uma foto. O rover vê a **mesma planta** em seis
câmeras e vários quadros seguidos, passa de novo na semana seguinte, e sabe onde
está. Com `FIELDS_DIR` configurado e a pose do rover na requisição, cada quadro
do `/predict/rover-frame` alimenta o estado do talhão:

```bash
curl -F "file=@quadro.jpg" "http://localhost:8000/predict/rover-frame?model=banana-foliar\
&camera_height_mm=1000&distance_mm=270\
&field_id=T07&row_id=L12&side=left&x_m=38.4&pass_id=2026-10-02&lat=-22.25&lon=-45.70&heading_deg=90"
```

**Fusão por planta** ([`plant_fusion`](app/plant_fusion.py)). Cada recorte é
posto ao longo da linha pela coluna do pixel e pela distância, e associado à
planta mais próxima (espaçamento de 2,5 m na banana e 4 m no citros). Por
planta e por doença, a evidência se acumula em log-odds, a partir de uma
prevalência a priori, com a razão de verossimilhança de cada recorte medida em
plantas rotuladas — ver [calibração da fusão](#calibração-da-fusão). Recortes que
se sobrepõem e quadros seguidos **não** contam como vistas independentes: um
desconto de correlação impede que vinte recortes do mesmo quadro virem certeza.
Recorte recusado pelo portão não entra. O resultado (`GET /fields/{talhão}/plants`)
traz, por planta:

- a posterior de cada doença e o status: `sadia`, `suspeita`, `doente` ou `revisar`;
- a incidência (a fração dos recortes da planta com doença);
- quantas câmeras e quadros a viram;
- em que zona a doença apareceu;
- a calibração com que a passada foi julgada (`calibration_id`; nulo = sem calibração).

**Evolução e mapa.** Ao fechar a passada (`POST .../passes/{passada}/close`),
cada planta é comparada com a passada anterior: nova infecção, piora ou
melhora. A evolução guarda a AUDPC (área sob a curva de progresso da doença).
Mande `captured_at` com o instante da foto: o rover que ficou sem enlace
sincroniza horas depois, e sem ele a planta seria datada pela chegada — a AUDPC
sairia errada. `x_m` e `heading_deg` com `nan` são recusados (422): toda
comparação com NaN é falsa, e a planta nunca casaria com a vizinha.
`GET .../map` devolve um GeoJSON com um ponto por planta, que abre direto no
QGIS ou num mapa web. Com GPS as coordenadas são WGS84; sem GPS, são metros ao
longo da linha, marcados como coordenadas locais.

<a id="incidência-do-talhão"></a>
**Incidência do talhão** (`GET /fields/{talhão}/incidence`). A fração de
**plantas** doentes, que é a pergunta do agrônomo. A fração aparente — plantas
`suspeita` ou `doente` — mistura dois erros: a doente que passa e a sadia
sinalizada. Com as duas taxas por planta medidas na calibração da fusão, a
correção de Rogan-Gladen desfaz a mistura, e o intervalo de 95% leva a incerteza
das três proporções (a do talhão e as duas taxas, que vêm de amostras finitas;
cada uma vira uma Beta com prior de Jeffreys). Plantas `revisar` ficam fora da
conta, contadas à parte em `status_counts`. Sem calibração, sai só a aparente
(`method: "aparente"`). A rota `prevalence` traz a mesma resposta em
`plant_incidence`. Em talhões simulados, sorteados de plantas que a calibração não
viu (ver a tabela em [calibração da fusão](#calibração-da-fusão)): na banana, erro médio de 1 a 4 pontos de
incidência (a fração aparente erra 3 a 6) e o intervalo cobriu a verdade em 83 a 100%
dos talhões; nos citros, erro de até 1,5 ponto e cobertura de 100% — mas com sadias de
papel, que quase nunca dão alarme. Até a validação de campo medir o alarme falso numa
árvore de verdade, a incidência dos citros é indicação.

**Prevalência dos recortes** ([`prior_shift`](app/inference/prior_shift.py)).
Os modelos aprenderam com a prevalência dos datasets públicos: 38% de Sigatoka
no treino da banana. Num talhão com 3% de Sigatoka, esse prior superestima a
doença em toda folha duvidosa. O talhão estima a própria prevalência **sem
rótulo**, pelo EM de Saerens, Latinne & Decaestecker (2002), a partir dos
recortes que o portão aceitou. O esquecimento é exponencial (meia-vida de 2.000
imagens), para acompanhar a estação. `GET .../prevalence` mostra lado a lado o
prior do treino e o do talhão. O ponto de partida é o `<modelo>.prior.json`,
gravado por `python -m training.fit_prior`.

**Não é a fração de plantas.** É a proporção de cada classe entre os recortes, e o
modelo põe probabilidade de doença em pedaço de folha sadia: num talhão simulado de
bananeiras todas sadias (02/10/2026), o EM dava 60% dos recortes como doentes (a
contagem simples, 52%); num de citros sadios, quase todos. Ela serve para corrigir o
prior do modelo; quantas plantas estão doentes é a [incidência](#incidência-do-talhão).

**O EM pode piorar a estimativa, e a resposta diz quando.** Ele assume que o modelo
está calibrado no talhão e amplifica a dúvida: nos recortes de campo do PlantSeg
(30/09/2026), o erro da prevalência da banana foi 0,16 com o EM e 0,07 só contando
os palpites; no citros, 0,03 e 0,04. Em vez de trocar de estimador, a rota devolve
`predicted_shares` (a contagem simples), `divergence` (distância de variação total
entre as duas) e `reliable`: falso quando passa de 0,05, com uma `note`. Nos 8
casos medidos, a regra separou os 6 em que o EM errava mais de 0,13 (discordância de
0,086 a 0,199) dos 2 em que acertava (até 0,022). É uma regra tirada de poucos
casos, não uma garantia. O módulo também traz BBSE (Lipton
et al., 2018) e um monitor de deriva (KS, PSI, qui-quadrado, MMD), que ainda
esperam a matriz de confusão da validação ser gravada no treino para entrar na
API.

**Severidade da Sigatoka** ([`sigatoka_severity`](app/sigatoka_severity.py)).
Nos quadros de banana, a resposta traz `sigatoka`: grau por folha (faixa de
altura) na escala de Stover modificada por Gauhl e o índice de severidade (DSI),
com intervalo de 95% por bootstrap. **A escala ainda não foi validada por
agrônomo** (`scale_validated: false`): o módulo traz o kappa ponderado e o MAE
do DSI justamente para essa validação, contra notas de campo.

Esses quatro módulos (fusão, prevalência, aprendizado ativo e severidade) foram
construídos por agentes na nuvem como módulos independentes, com testes. Aqui
foram revisados, conferidos com os testes e o lint do projeto e ligados às rotas.

<a id="calibração-da-fusão"></a>
### Calibração da fusão

Até 02/10/2026 a fusão usava sensibilidade e especificidade por recorte escritas à
mão, e os status saíam de posteriores de 0,5 e 0,9 sem medida: no talhão de ponta a
ponta, 9 de 18 bananeiras doentes foram achadas. O motivo, medido: o modelo põe
probabilidade de doença em pedaço de folha sadia — na banana, 45% em média somando
as quatro doenças; nos citros, 48% de Greening em recortes de plantas sem Greening —,
então a média das probabilidades quase não separa as plantas. O que separa é a cauda
perto de 1. O [`fit_fusion`](training/fit_fusion.py) mede em plantas rotuladas o que a
fusão precisa e grava `<pesos>.fusion.json`:

- **a razão de verossimilhança de cada recorte**, por faixa da probabilidade da
  doença: o histograma em recortes de plantas com a doença contra o de plantas sem ela;
- **os limiares dos status pelo alarme falso em plantas sadias**: `suspeita` ou
  `doente` em no máximo 5% delas, `doente` em no máximo 2%. Não dependem da
  prevalência do talhão. Com piso no que o status quer dizer: `suspeita` só com
  evidência a favor da doença (razão de verossimilhança calibrada ≥ 1), `doente` só
  com posterior de pelo menos 50% no prior de 5% — o número que o painel mostra. Sem
  o piso, a meta de 5% nas sadias de papel dos citros pôs o limiar de `suspeita`
  abaixo de zero, e a demonstração do SaaS marcou como suspeita laranjeira sadia com
  0,1% de posterior;
- **uma escala de Platt**, para a posterior ser probabilidade de verdade — a
  prescrição pesa custo contra ela;
- **a sensibilidade e o alarme falso por planta**, com o tamanho efetivo da amostra
  que os mediu, que corrigem a [incidência](#incidência-do-talhão).

Uma planta da calibração são três fotos, uma por altura de câmera do mastro, pelo
mesmo caminho do `/predict/rover-frame` (recortes de 150 mm a 270 mm, portão,
conjunto da fusão, sem TTA). Os números são **fora da amostra**: validação cruzada
aninhada em 10 partes, em que a escolha da correlação e do corte da LLR, os limiares e
o Platt saem só das partes de treino.

| | Doentes detectadas | Com a doença certa | Sadias com alarme (como `doente`) |
| --- | ---: | ---: | ---: |
| banana, sem calibração | 66/125 (53%) | 20 | 13/283 (0) |
| **banana, calibrada** | **115/125 (92%)** | 61 | 15/283 (0) |
| citros, sem calibração | 27/33 | 27 | 3/83 (0) |
| **citros, calibrada** | **33/33** | 32 | 2/83 (0) |

Na banana, as doentes e as sadias vêm do teste isolado do acervo (fotos de campo) e
mais 18 doentes do teste do PlantSeg; a detecção é a mesma nas duas fontes (98/107 e
17/18), sinal de que a calibração não aprendeu a diferença entre acervos. Por doença:
Cordana 18/20, Fusariose 41/44, Pestalotiopsis 10/10, Sigatoka 46/51. Nos citros, o
acervo é de folha destacada **sobre papel**: com as doentes de papel na calibração, a
detecção nas fotos de campo do PlantSeg caiu de 33 para 25 de 33, e elas ficaram de fora
(`--data-dir-classes Healthy`). Medida sem reajuste nas 315 plantas de papel, a
calibração dos citros acha Canker em 33 de 40 e Greening em 29 de 40: reage à doença, não
à fonte.

No talhão de ponta a ponta pela API (as mesmas fotos, montadas em outras plantas), a
banana achou 17 de 18 doentes (eram 9) e os citros 33 de 33 (eram 27). Das sadias, 3 de
18 bananeiras viraram `suspeita` (era 1) e nenhuma laranjeira; nenhuma sadia saiu
`doente`. Em 2.000 trios sadios sorteados das mesmas fotos, o alarme é 5,9% na banana —
as 3 de 18 do e2e são um sorteio azarado — e 0,2% nos citros. A incidência estimada foi
60% (IC95 41–78%) na banana e 49% (36–62%) nos citros, para 50% de plantas doentes nos
dois. Na demonstração do SaaS (`simulator.pomar` com a IA de verdade), a situação da
planta saiu certa em 18 de 18 no bananal e 15 de 15 no pomar.

**O que ainda não está medido — e por isso a validação de campo:**

- as plantas da calibração são fotos de folha agrupadas de três em três, não o rover
  no pomar;
- as sadias dos citros são folhas sobre papel: **o alarme falso dos citros numa árvore
  de verdade não foi medido**, nem com a fusão calibrada nem sem ela;
- acertar a doença é o ponto fraco da banana (61 das 115 detectadas): Sigatoka, Cordana
  e Fusariose se confundem por planta;
- nove das onze doenças dos citros não têm planta de campo rotulada; as tabelas delas
  herdam a distribuição combinada das que têm.

Cada passada guarda a calibração com que foi ingerida (`calibration_id` em cada
planta): instalar uma nova não mistura evidências de modelos diferentes e vale da
próxima passada em diante. Sem `<pesos>.fusion.json`, a fusão usa os valores de partida
e avisa no log.

```bash
# os recortes do rover, planta a planta (GPU)
python -m training.fit_fusion collect --model banana-foliar \
    --data-dir data/banana-foliar-campo \
    --plantseg ~/valtexa-raw/plantseg/plantseg/images/test --out medicoes/banana.jsonl
python -m training.fit_fusion collect --model citrus-foliar \
    --data-dir data/citrus-foliar-campo --data-dir-classes Healthy \
    --plantseg ~/valtexa-raw/plantseg/plantseg/images/test --out medicoes/citros.jsonl
# a calibração (CPU, segundos), gravada ao lado dos pesos
python -m training.fit_fusion fit --model banana-foliar --frames medicoes/banana.jsonl
python -m training.fit_fusion fit --model citrus-foliar --frames medicoes/citros.jsonl
```

#### Validação de campo

1. No servidor do piloto, `FIELDS_DIR` e `FIELDS_TILE_LOG=true`: cada recorte fica
   gravado com a planta a que foi associado, em `FIELDS_DIR/plantas/registro/`.
2. Escolha **fileiras inteiras** e diagnostique **todas** as plantas delas, não só as
   sinalizadas: sem as sadias e as doentes que o rover deixou passar, não dá para medir
   sensibilidade nem alarme falso. Grave um CSV `field_id,plant_id,label` (com `pass_id`
   se o rótulo vale para uma passada só), com os ids do mapa e as classes do modelo
   (`Healthy` para sadia).
3. Meça a calibração instalada nessas plantas, **sem reajuste** — é a validação externa:
   ```bash
   python -m training.fit_fusion evaluate --model banana-foliar \
       --frames fields/plantas/registro/T07.jsonl --labels rotulos.csv --fields-dir fields
   ```
4. Com dezenas de plantas doentes rotuladas, recalibre com o rover de verdade (`fit`,
   com os mesmos argumentos), sozinho ou junto das fotos do `collect`.


<a id="o-teste-de-campo-na-escala-do-rover"></a>
### O teste de campo na escala do rover

O rover classifica recortes de 150 mm de folha, não fotos de planta inteira. O
teste que corresponde a isso sai do [`lesion_tiles`](training/lesion_tiles.py):
as fotos de campo do PlantSeg (split oficial de teste, nunca vistas no treino)
viram recortes quadrados sobre as lesões marcadas nas máscaras. Conferido antes
de medir: nenhuma dessas fotos está nos dados de treino. O tamanho real da
amostra é o de **fotos** (52 de banana, 95 de citros), não o de recortes — e o
citros só tem duas das doze classes no teste (cancro e greening).

| Modelo | Treino | Teste isolado | **Campo** |
| --- | --- | ---: | ---: |
| banana, 1º treino, fotos inteiras do PlantSeg | 3 datasets públicos | 0,985 | 0,428 |
| banana, 1º treino, recortes | 3 datasets públicos | 0,985 | 0,592 |
| **banana** | + recortes de campo do PlantSeg-treino | 0,977 | **0,829** |
| citros | só as fotos sobre papel | 0,874 | 0,695 |
| citros | papel + troca de fundo | 0,847 | 0,382 |
| citros | papel + troca de fundo + recortes de campo | 0,848 | 0,959 |
| **citros** | papel + recortes de campo, **sem** troca de fundo | **0,884** | **0,969** |

O que os números ensinaram:

1. **O teste isolado mente para cima.** Ele sai das mesmas câmeras e do mesmo
   fundo do treino. Só o de campo diz como o modelo se sai numa lavoura.
2. **A escala importa.** O mesmo modelo de banana foi de 0,428 para 0,592 só por
   ser medido em recortes em vez de fotos de bananal inteiro.
3. **Recortes de campo no treino são o que faz funcionar.** Banana 0,592 → 0,829;
   citros 0,695 → 0,969. A Cordana da banana, que o primeiro modelo quase nunca
   reconhecia (8% dos recortes), passou a 62,5%.
4. **A troca de fundo piorou.** Colar a folha do papel num pomar filmado pareceu
   óbvio; medido, o citros só-papel caiu de 0,695 para 0,382 no campo com ela —
   passou a chamar cancro e greening de fumagina. Com os recortes de campo, tirar
   a troca de fundo ainda melhorou os dois testes (0,848 → 0,884 no isolado,
   0,959 → 0,969 no campo): o modelo instalado é o sem troca de fundo.

**Licença.** O PlantSeg é CC BY-NC 4.0 (não comercial). Os modelos de banana e
citros com melhor resultado de campo usam recortes dele no treino; para uso
comercial, eles precisam ser substituídos por fotos de campo próprias — que é o
que a [fila de revisão](#ciclo-de-dados-próprio) do rover produz. As versões
treinadas só com dados CC BY ficam em `models/` com o sufixo `_anterior` (banana)
e `_papel` (citros).

<a id="servindo-sem-torch-onnx-runtime"></a>
## Gêmeo digital do talhão

O resto do sistema responde *o que a planta tem hoje*. O gêmeo (`app/gemeo/`) responde *onde
a doença vai estar*, com o tempo da fazenda, e o que fazer antes de ela chegar:

| módulo | o que faz |
| --- | --- |
| `grafo` | As plantas da fusão viram nós; arestas ligam cada pé às vizinhas na rua (até 12 m) e nas duas ruas de cada lado. Uma camada por passada fechada, com a posterior por planta **sem** o carregamento temporal da fusão (`PlantFusionStore.pass_observations`). |
| `clima` | O tempo dia a dia: estação (registros horários), Open-Meteo (com `CLIMA_OPEN_METEO`), climatologia além da previsão. Sem sensor de molhamento, hora com UR ≥ 90% ou chuva conta como folha molhada. |
| `epidemia` | SEIR por planta (probabilidade de cada estado), passo diário: temperatura (Yan & Hunt), molhamento, chuva, latência que encurta no calor, núcleo de contágio anisotrópico (o mesmo da prescrição). Presets qualitativos para ferrugem, cercosporiose, phoma e bicho-mineiro. A passada entra por **Bayes exato por planta**, com a razão de verossimilhança da fusão. |
| `ajuste` | MAP de β, alcances, fundo, alarme falso e sensibilidade sobre a previsão um passo à frente (Nelder-Mead + Levenberg-Marquardt, sem SciPy), incerteza de Laplace. |
| `rede` | Rede residual espaço-temporal (uma camada de mensagem pelo grafo) sobre o logit da física. Só é usada se ganhar da física na passada que não viu. |
| `previsao` | Conjunto (parâmetros de Laplace × tempo perturbado conforme a fonte): risco por planta com faixa, janelas de infecção, focos de agora e do horizonte. |
| `prescricao_risco` | A prescrição de sempre sobre o risco previsto, com os alcances ajustados, a data (véspera da janela de infecção) e os cenários no gêmeo. |
| `prioridade` | Onde vale olhar: o valor da informação da decisão tratar / não tratar, por trecho de rua ou bloco, para o enxame (`VALTEXA-SAAS/docs/ENXAME.md`) alocar. |

Uso, num talhão com passadas fechadas:

```bash
curl -X POST localhost:8000/fields/T01/clima -H 'content-type: application/json' \
     -d '{"registros_horarios": [{"time": "2026-03-01T00:00:00-03:00", "temperatura_c": 19.5, "ur": 93, "chuva_mm": 0}]}'
curl -X POST 'localhost:8000/fields/T01/gemeo/ajuste?doenca=Rust'
curl 'localhost:8000/fields/T01/gemeo/previsao?doenca=Rust&horizonte_dias=21'
```

**O que está medido e o que não está.** Tudo foi medido num talhão **sintético** onde a
verdade é conhecida (`python -m training.mede_gemeo`): epidemia estocástica planta a planta,
cobertura parcial, fusão calibrada. Não há medição em lavoura real ainda — os parâmetros
biológicos dos presets são faixas da literatura, a revisar com o agrônomo, e o piloto é que
vai dizer quanto o gêmeo acerta no campo. Os números e os limites estão em
[`MODELS.md`](MODELS.md#gemeo).

## Servindo sem torch: ONNX Runtime

O `.keras` precisa do Keras e de um backend inteiro — o torch com CUDA passa de
2 GB instalado — só para multiplicar matrizes na hora de servir. O
`export_onnx` grava o mesmo modelo em ONNX, e o ONNX Runtime o executa com uma
dependência de dezenas de MB, em CPU, GPU ou num computador de bordo na ground
station:

```bash
python -m training.export_onnx --model leaf-foliar
python -m training.export_onnx --severity
python -m training.export_onnx --framing
INFERENCE_RUNTIME=onnx uvicorn app.main:app      # com requirements-edge.txt
```

Antes de gravar, o exportador roda as mesmas imagens reais pelos dois caminhos e
**apaga o arquivo** se a diferença passar da tolerância: um grafo que responde
diferente não quebra nada, só troca diagnósticos em silêncio.

Medido em 26/09/2026, contra o Keras em fp32 (com o TF32 da GPU desligado — ele
sozinho gerava diferenças de até 0,11 numa probabilidade), em 256 imagens reais:

| Grafo | Tamanho | Maior diferença | Classe / severidade | CPU, 1 imagem |
| --- | ---: | ---: | --- | ---: |
| `leaf-foliar-b` | 24,4 MB | 7,8e-6 | 100% iguais | 12,4 ms |
| severidade | 21,7 MB | 1,0e-4 por pixel | Δ severidade 0,0000 | 23,9 ms |
| torre do enquadramento | 372 MB | 1,1e-5 | 100% dos vereditos | — |
| `leaf-foliar` / `coffee-foliar` | 9,7 MB | 1,1e-4 / 1,2e-4 | 100% iguais | — |

Os dois últimos foram recusados pela tolerância de então, que comparava o
embedding em valor absoluto; agora ele é comparado em relação à própria escala, e
a reexportação ficou pendente.


O grafo dos classificadores sai com **duas saídas**, probabilidades e embedding,
então a novidade continua vindo da mesma passada. E a torre de visão do
enquadramento também exporta: o pré-processamento do SigLIP (224×224 bilinear,
pixels em `[-1, 1]`) é **o mesmo** do servidor, então com `INFERENCE_RUNTIME=onnx` o
portão roda sem `transformers`. O redimensionamento é o do Pillow, não o da API (estilo
TensorFlow): com o da API o embedding saía diferente ao reduzir uma foto grande, e a
cena trocava em até 12% das imagens; com o do Pillow o cosseno com o servidor é
1,00000 e cena e veredito saem 100% iguais (medido em 01/10/2026).

## Como usar as salvaguardas

Cada uma responde a uma pergunta diferente, e nenhuma substitui a outra:

| Campo | Pergunta | Ação quando dispara |
| --- | --- | --- |
| `framing.in_scope` falso | a foto é folha em close? | não mostrar diagnóstico; recapturar |
| `framing.crop_matches` falso | é da cultura deste modelo? | mandar ao modelo da outra cultura |
| `quality.ok` falso | a foto está boa? | recapturar |
| `prediction_set` com ≠ 1 classe | dá para afirmar uma só? | agrônomo, entre as candidatas |
| `view_agreement` < 1 | o modelo muda de ideia ao girar a foto? | agrônomo |
| `needs_review` | a imagem é estranha para o modelo? | fila de revisão |
| `abstained` | a confiança passou do mínimo? | inconclusivo |

Na plataforma (`VALTEXA-SAAS`), `framing.in_scope` é o que deveria alimentar o
`model_input_ok` da observação — hoje ele é inferido só pela plataforma de
captura.

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

### O que o agrônomo rotula primeiro

O rover manda milhares de recortes por dia, e o agrônomo rotula dezenas. Por
data, a fila gasta o tempo dele com folha que o modelo já acerta com folga.
`GET /review?order=valor` ordena pelo que mais ensina ao próximo treino
([`active_learning`](app/active_learning.py)): incerteza (margem, entropia,
conjunto conformal), novidade e classe rara entre as já rotuladas, com cotas por
classe. Cada item vem com `priority_reasons` ("incerteza alta: margem 0,04",
"classe rara: Cordana").

Medido no simulador do módulo (gaussianas desbalanceadas com uma classe rara,
regressão logística, 5 sementes): para chegar a 0,777 de acurácia, o sorteio
aleatório precisou de 100 rótulos, e a ordem por incerteza, de 70 — **30% menos
fotos rotuladas para o mesmo modelo**. O módulo também tem BADGE (Ash et al.,
2020) e core-set, que usam o embedding da penúltima camada; enquanto a fila não
guarda embeddings, a ordem é por incerteza, que no simulador foi a melhor.

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

## Aprendizado contínuo no campo

A cabeça de diagnóstico (a leitura sobre o embedding do SigLIP 2 — no café, a resposta
inteira) aprende com o que o agrônomo valida no campo, **sem esquecer o que já sabia**. A
torre e a CNN ficam congeladas; o treino é NumPy, na CPU, em segundos por rodada.
O porquê de cada peça está em [`app/inference/continual.py`](app/inference/continual.py).

```
validação do agrônomo (plataforma) → training.import_campo → recortes confirmados
        ↓
training.adapt_campo adapta    replay (memória por origem) + destilação + âncora L2-SP
        ↓                      época escolhida na validação que a cabeça nunca treinou
portão de regressão            conjuntos congelados, comparação pareada (bootstrap)
        ↓ APROVADA
training.adapt_campo promove   instala, arquiva a anterior, carimba o perfil do Rover
        ↓
training.adapt_campo monitora  a cada lote novo do agrônomo; reverte se piorou
```

```bash
python -m training.adapt_campo memoria --model coffee-foliar --extra data/coffee-bracol \
    --externo ~/valtexa-raw/coffee-leaves-disease            # uma vez por modelo
python -m training.adapt_campo adapta --model coffee-foliar --campo data/coffee-campo \
    --escopo fazenda-boa-vista                               # sai 0 (APROVADA) ou 2
python -m training.adapt_campo promove --model coffee-foliar --rodada <rodada>     --aprovador "Eng. Agr. Fulana (CREA 0000)"               # grava a procedência
python -m training.adapt_campo monitora --model coffee-foliar --campo data/coffee-campo-novo
python -m training.adapt_campo reverte --model coffee-foliar
python -m training.adapt_campo esquece --model coffee-foliar --fonte campo_T01   # LGPD
python -m training.adapt_campo historico --model coffee-foliar
curl localhost:8000/models/coffee-foliar/continuo            # o mesmo histórico, pela API
```

**O portão.** A candidata só substitui a cabeça em produção se: no agregado dos conjuntos
congelados, o limite inferior do IC de 95% da diferença de acurácia balanceada fica acima
de −2 pontos (não-inferioridade: o ônus da prova é dela); nenhum conjunto tem piora
demonstrada; nenhuma classe perde mais de 5 pontos nem tem piora demonstrada; o alarme falso
em folha sadia não sobe mais de 2 pontos — no café, o erro que mais custa; e ela **ganha**
no teste do campo. O teste de cada rodada aprovada vira conjunto congelado: a rodada seguinte
não pode esquecê-lo. O laudo (`models/_continuo/<pesos>/rodadas/<rodada>/laudo.md`) mostra
conjunto a conjunto, com os intervalos.

**Medido em 15/10/2026** (embeddings reais; cabeça de partida treinada só no
`coffee-foliar`; o BRACOL fazendo o papel de uma fazenda nova, chegando em 3 rodadas de
~197 folhas; teste final de 336 folhas do BRACOL que ninguém viu; 3 sementes; média ±
desvio do estado final):

| | café, teste isolado | alarme falso em sadia (café) | BRACOL, teste final | alarme falso em sadia (BRACOL) | externo (só mede) |
| --- | ---: | ---: | ---: | ---: | ---: |
| cabeça de partida | 96,8% | 2,6% | 30,0% | 100% | 16,5% |
| ajuste ingênuo, promovido à força | **81,3%** ± 3,0 | 3,2% | 85,0% | 0% | 86,1% |
| **adapt_campo (portão decide)** | **96,7%** ± 0,3 | **1,2%** ± 0,4 | **84,7%** ± 1,5 | **0%** | **84,1%** ± 3,5 |

* O ajuste ingênuo aprende a fazenda nova e **esquece o café** (Cercospora perde até 57
  pontos); o portão recusou 9 de 9 rodadas ingênuas, com piora demonstrada.
* Com rótulos ruins (40% trocados — validação malfeita), o portão recusou 3 de 3.
* O portão é rigoroso: aprovou 4 de 9 rodadas boas. Das 5 recusadas, em 2 nenhuma época
  melhorava o campo sem piorar o que já se sabia — a cabeça volta intacta ("sem ganho"); nas
  outras 3 o custo era pequeno e real (ferrugem com piora demonstrada de 3,8 e 4,6 pontos;
  alarme falso em sadia +2,5 pontos). Para comparação, a cabeça **treinada de uma vez** com
  café + BRACOL — a que está em produção — sobe o alarme falso no café de 2,6% para 5,2%:
  aprender o BRACOL tem esse custo por natureza do dado, e o aprendizado contínuo com portão
  terminou em 1,2%.

O que **não** foi medido: recortes do Rover num cafezal (não existem ainda — o BRACOL são
folhas inteiras fotografadas); a primeira rodada de verdade é com a validação de campo
(ROVER_RUNTIME §17, `import_campo`). Reproduzir: `python -m training.mede_continuo`.

**Depois de promover**: a cabeça nova muda as probabilidades. O `promove` carimba o perfil
da borda instalado (`.edge.json`, `head.head_sha256`): se a evidência nas folhas sadias se
move mais que meia faixa entre quase-evento e evento, o perfil fica preso à cabeça anterior
e **o Rover recusa a nova** até o `fit_edge` recalibrar; senão, o perfil passa a valer para
a nova. O conformal da API (`fit_conformal`) precisa ser recalibrado sempre — o `promove`
lista as pendências.

**Que tipo de atualização é.** Cada mudança mexe numa coisa e invalida coisas diferentes
([`app/inference/atualizacao.py`](app/inference/atualizacao.py)); o `adapta` classifica a
candidata contra a instalada **pela forma**, não pelo comando que a gerou, e registra no
laudo o que ela exige e invalida:

| tipo | o que muda | o ciclo contínuo pode? | invalida |
| --- | --- | :---: | --- |
| `cabeca` | os pesos da cabeça, mesma forma | sim | conformal; perfil da borda se a deriva passar da folga |
| `estrutura` | neurônios, camadas, atalhos, ativações | não (treino novo) | conformal, perfil da borda sempre, logits da memória |
| `embedding` | a torre de visão | não | tudo: cabeças, memória, regressão, régua, conformal, perfil, motor TensorRT |
| `prior` | a prevalência esperada no talhão | sim (`prior_shift`, gêmeo) | — |
| `regra_de_decisao` | limiares e tabelas de evidência | não (`fit_edge`, `fit_fusion`) | comparações de alarme com a regra antiga |

**Procedência.** O `promove` exige `--aprovador` e grava `<pesos>.head.provenance.json` ao
lado da cabeça: o tipo, a cabeça de partida e a nova (SHA-256), a rodada e o escopo, o hash
do laudo do portão e do dado de campo, quem liberou e, se forçada, o motivo. O hash dela vai
no perfil da borda (`measured.continuo.procedencia_sha256`) e o Rover recusa cabeça adaptada
sem procedência válida (cópia byte a byte de `atualizacao.py` em
`valtexa_rover/edge_ai/vendor/`). O `reverte` leva a procedência junto: a cabeça de origem
volta sem nenhuma.

**Motor com a plasticidade do dNATY** (`--motor dnaty-plasticidade`,
[`training/motor_dnaty.py`](training/motor_dnaty.py); precisa de torch e dnaty): renasce os
neurônios dormentes da cabeça antes do ajuste. Medido em 18/10/2026 (MODELS.md): a cabeça do
café não tem dormentes nem perde rank efetivo em 3 rodadas, o motor sai idêntico ao padrão, e
renascer à força a fração pura do dNATY piora. Fica desligado.

**Consentimento**: cada exemplo na memória e nos conjuntos guarda a origem e o arquivo; o
`esquece` tira o que o cliente retirou. A cabeça já treinada com o dado continua com o que
aprendeu — para tirar a influência, `reverte` para antes da rodada e adapte de novo sem ele.

## Paridade entre treino e inferência

O modo de falha mais perigoso do projeto é um pré-processamento diferente entre
treino e produção: não gera erro, só derruba a acurácia em silêncio.

**A normalização é literalmente o mesmo código.** Os dois lados chamam
`scale_to_unit_range` de [`app/inference/preprocess.py`](app/inference/preprocess.py)
para levar os pixels a `[-1, 1]`. Aqui não há como divergir.

**O resto agora também é o mesmo cálculo.** Até 26/09/2026, a inferência
girava a foto pelo EXIF e redimensionava com o bilinear do Pillow; o treino não
gira (`tf.io.decode_image` ignora o EXIF) e redimensiona com o `tf.image.resize`.
Estava registrado aqui como "folga conhecida, a que nenhum ensaio atribuiu perda".
O ensaio foi feito — mesmas imagens de teste, mesmo modelo, três caminhos:

| Com TTA | Pipeline do treino | Pillow, sem EXIF | API de antes (Pillow + EXIF) |
| --- | ---: | ---: | ---: |
| `leaf-foliar` | 95,55% | 95,96% | 95,49% |
| `leaf-foliar-b` | **96,01%** | 95,60% | 95,35% |
| `coffee-foliar` | **94,99%** | 92,49% | 92,29% |

A folga custava **2,7 pontos no café**. Duas causas, separáveis:

- **a reamostragem.** O café é 128×128 ampliado para 224, e o Pillow amplia de
  outro jeito — é daí que vem quase toda a perda dele;
- **o EXIF.** Um terço das fotos de milho (999 de 2.968 no teste) traz rotação no
  EXIF. Girada, a foto é espremida para o quadrado **na outra direção** — o
  redimensionamento não preserva a proporção —, e o modelo nunca viu esse
  achatamento. Custava de 0,4 a 0,5 ponto.

A API agora decodifica sem girar e redimensiona com
[`resize_like_tensorflow`](app/inference/preprocess.py): o bilinear do TensorFlow
(centros de pixel, sem antialias) reimplementado em NumPy, sem depender dele.
`tests/test_preprocess.py` o compara com o `tf.image.resize` em ampliação,
redução e proporções estranhas, quando o TensorFlow está instalado.

Medido depois da correção, pelo caminho da própria API, no mesmo teste isolado:

| Com TTA | Treino (publicado) | API de antes | **API agora** |
| --- | ---: | ---: | ---: |
| `leaf-foliar` | 95,55% | 95,49% | **95,6%** |
| `leaf-foliar-b` | 96,01% | 95,35% | **95,9%** |
| `coffee-foliar` | 94,99% | 92,29% | **95,0%** |

A produção passou a entregar o número que o treino mediu. Sobra 0,1 ponto no
`leaf-foliar-b`, sem causa isolada; a candidata é a decodificação de JPEG
(Pillow × TensorFlow), que o teste de paridade do redimensionamento não cobre.

Exceção deliberada: a **severidade** continua com o Pillow, porque foi com ele
que o `train_severity` redimensionou imagem e máscara. `load_image_from_bytes`
ainda gira pelo EXIF com `apply_exif=True`, para quem vai mostrar a foto a uma
pessoa.

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

- [ ] **Coleta de campo própria de banana e citros com o rover**, para substituir
      os recortes do PlantSeg (licença não comercial) e cobrir as dez classes de
      citros que o teste de campo ainda não mede.
- [ ] **Validação de campo da fusão e da incidência**: `FIELDS_TILE_LOG=true` no
      piloto, fileiras inteiras rotuladas pelo agrônomo, `fit_fusion evaluate` antes de
      recalibrar. Primeiro o alarme falso dos citros em árvore de verdade, que ninguém
      mediu — ver [calibração da fusão](#calibração-da-fusão).
- [ ] Dataset de pseudocaule a partir dos recortes que o portão manda para
      `out_of_scope` na zona `pseudocaule` — hoje não há dado público.
- [ ] Gravar a matriz de confusão da validação no treino, para ligar o BBSE e o
      monitor de deriva da prevalência por talhão.
- [ ] Guardar o embedding de cada item da fila, para o BADGE e o core-set.
- [ ] Validar a escala de Sigatoka com notas de agrônomo em campo (kappa e MAE do
      DSI já estão no módulo).

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
- [ ] Rodar `audit_attention` no `leaf-foliar` e no `rover-understory` em
      quarentena (o caso de controle) e registrar o resultado.
- [x] Medir quanto `view_agreement < 1` pega de erro no teste isolado (30/09/2026, acima).
- [ ] Recalibrar `fit_quality` sobre o dataset consolidado de cada modelo.
