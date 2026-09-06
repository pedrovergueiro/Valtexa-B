# Corn & Coffee Disease Detection — Backend

Backend de inferência (FastAPI) para detecção de doenças em milho e café a
partir de imagens de drone e de rover terrestre.

A API serve **um modelo por combinação de cultura e enquadramento** — o drone
varre e tria, o rover confirma e diagnostica. O detalhamento de cada modelo, das
classes e dos datasets está em **[MODELS.md](MODELS.md)**.

| Modelo | Cultura | Plataforma | Papel |
| --- | --- | --- | --- |
| `leaf-foliar` | Milho | qualquer | Diagnóstico foliar em close — **treinável hoje** |
| `coffee-foliar` | Café | qualquer | Diagnóstico foliar em close — **treinável hoje**, diferencial estratégico |
| `rover-understory` | Milho | rover | Diagnóstico sob o dossel, colmo e base |
| `aerial-canopy` | Milho | drone | Triagem de dossel, direciona o rover |

## Estrutura

```
app/                      # API de inferência
  main.py                 # rotas: /health, /models, /classes, /predict, /predict/batch
  registry.py             # catálogo de modelos (classes, pesos, datasets)
  model.py                # carregamento dos pesos + execução das previsões
  config.py               # variáveis de ambiente
  schemas.py              # contrato HTTP (Pydantic)
  inference/
    preprocess.py         # decodificação e pré-processamento das imagens
training/                 # treino
  train.py                # script principal (2 fases: cabeça + fine-tuning)
  prepare_dataset.py      # consolida datasets públicos no layout canônico
  dataset.py              # carregamento e augmentation
  model_def.py            # arquitetura (MobileNetV2 + cabeça)
tests/                    # suíte da API, do catálogo e do pré-processamento
models/                   # pesos + .labels.json (não versionados)
data/<modelo>/            # dataset por modelo (não versionado)
```

## Requisitos

**Python 3.10–3.12.** O TensorFlow 2.17 ainda não publica wheels para 3.13+,
então um ambiente 3.14 não consegue instalar as dependências.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
copy .env.example .env
```

Para rodar só a API em produção, `requirements.txt` basta —
`requirements-dev.txt` acrescenta pytest, httpx e ruff.

## Treinando

Passo a passo completo, com datasets e classes de cada modelo, em
**[MODELS.md](MODELS.md)**. O caminho curto:

```bash
# consolida os datasets públicos em data/leaf-foliar/<Classe>/
python -m training.prepare_dataset --model leaf-foliar \
    --source "C:/datasets/corn-or-maize-leaf-disease" --dry-run

# treina
python -m training.train --model leaf-foliar --epochs 15 --fine-tune-epochs 5
```

Ao final, `models/` contém os pesos da **melhor época** segundo `val_accuracy` e
o `.labels.json` com a ordem das classes — que é o que a API usa para traduzir o
índice previsto em nome.

`python -m training.train --help` lista todos os parâmetros.

### GPU no Windows

O TensorFlow 2.10 foi o último release com suporte a GPU no Windows nativo. Para
treinar com aceleração a partir do 2.11, use **WSL2** (ou Colab/Kaggle). No
Windows nativo o treino roda, mas só em CPU.

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
  "probabilities": [
    { "class_name": "Blight", "probability": 0.011 },
    { "class_name": "Common_Rust", "probability": 0.973 },
    { "class_name": "Gray_Leaf_Spot", "probability": 0.009 },
    { "class_name": "Healthy", "probability": 0.007 }
  ],
  "inference_ms": 42.8
}
```

O `/predict/batch` processa o lote inteiro numa única passada e acrescenta um
`summary` com a contagem por classe — útil para fechar o resultado de um voo. A
validação é tudo-ou-nada: se uma imagem for ilegível, a requisição falha com
`400` identificando o arquivo.

## Configuração

Todas opcionais, via `.env` (ver `.env.example`).

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `DEFAULT_MODEL` | `leaf-foliar` | Modelo usado quando nenhum é informado. |
| `MODELS_DIR` | `models` | Onde ficam os pesos. |
| `DATA_DIR` | `data` | Raiz dos datasets de treino. |
| `MODEL_PATH` | — | Sobrescreve os pesos **só do modelo padrão**. |
| `MIN_CONFIDENCE` | `0.0` | Abaixo disso a resposta vem como `abstained`. |
| `MAX_UPLOAD_MB` | `10` | Tamanho máximo por imagem. |
| `MAX_BATCH_SIZE` | `20` | Imagens por requisição em `/predict/batch`. |
| `CORS_ORIGINS` | `localhost:5173,localhost:3000` | Origens do front, separadas por vírgula. |
| `LOG_LEVEL` | `INFO` | Nível de log. |

Com `CORS_ORIGINS=*` as credenciais de CORS são desativadas automaticamente — a
combinação `*` + credenciais é inválida por spec e, na prática, liberaria
qualquer site.

## Paridade entre treino e inferência

O modo de falha mais perigoso do projeto é um pré-processamento diferente entre
treino e produção: não gera erro, só derruba a acurácia em silêncio. Os dois
lados compartilham o mesmo código (`app/inference/preprocess.py`), tanto no
redimensionamento — bilinear, sem preservar aspect ratio, igual ao
`image_dataset_from_directory` — quanto na normalização para `[-1, 1]`. Há um
teste que compara a implementação com a do Keras sempre que o TensorFlow está
instalado.

## Testes e lint

```bash
pytest
ruff check .
ruff format .
```

A suíte roda **sem TensorFlow**: as rotas são exercitadas com um dublê do modelo
Keras. O único teste que exige o TF é o de paridade do pré-processamento, que é
pulado automaticamente quando ele não está instalado.

## Docker

```bash
docker build -t valtexa-disease-backend .
docker run -p 8000:8000 -v "$PWD/models:/app/models" valtexa-disease-backend
```

Os pesos não entram na imagem — são binários grandes e versionados à parte, então
`models/` é montado como volume em tempo de execução.

## Roadmap

- [ ] Treinar `leaf-foliar` e `coffee-foliar`, validar cada um num conjunto de teste próprio.
- [ ] Piloto comercial com drone terceirizado (voo baixo + `leaf-foliar`/`coffee-foliar`).
- [ ] Coleta própria de café no Sul de Minas — ver prioridade em [MODELS.md](MODELS.md).
- [ ] Coleta de campo para as classes exclusivas do rover — ver [MODELS.md](MODELS.md).
- [ ] Camada de fusão: risco do drone → tasking do rover → diagnóstico confirmado.
- [ ] Autenticação entre front e backend (API key / JWT).
- [ ] Integrar o front (`VALTEXA FRONT`), hoje uma landing page sem chamadas à API.
- [ ] Deploy (cloud, servidor próprio ou edge na ground station).
