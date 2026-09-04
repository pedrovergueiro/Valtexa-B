# Corn Disease Detection — Backend

Backend de inferência (FastAPI) para o modelo de classificação de doenças em
folhas de milho, baseado no notebook
[corn-diseases-detection](https://www.kaggle.com/code/prabeshlc/corn-diseases-detection)
do Kaggle. Recebe imagens (ex.: capturadas por drone) e retorna a classe de
doença prevista com as probabilidades por classe.

Classes: `Blight`, `Common_Rust`, `Gray_Leaf_Spot`, `Healthy`.

## Estrutura

```
app/                  # API FastAPI (inferência)
  main.py             # rotas: /health, /classes, /predict, /predict/batch
  model.py            # carregamento do modelo + lógica de predição
  config.py           # constantes/env vars (classes, tamanho de imagem, paths)
  inference/
    preprocess.py      # decodificação e pré-processamento de imagem
training/             # treino do modelo (reproduz a arquitetura do notebook)
  train.py            # script principal de treino
  dataset.py          # carregamento/augmentation do dataset
  model_def.py         # arquitetura (MobileNetV2 + head de classificação)
models/                # pesos treinados (.keras) — não versionado
data/                  # dataset local (não versionado)
tests/                 # testes da API
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env
```

## Treinando o modelo

1. Baixe um dataset de folhas de milho com as 4 classes acima — por exemplo,
   o [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset).
2. Organize em `data/` com uma subpasta por classe:
   ```
   data/
     Blight/*.jpg
     Common_Rust/*.jpg
     Gray_Leaf_Spot/*.jpg
     Healthy/*.jpg
   ```
3. Rode o treino:
   ```bash
   python -m training.train --data-dir data --epochs 15 --fine-tune-epochs 5
   ```
   Isso gera `models/corn_disease_model.keras` e `models/labels.json`.

Se você já tem pesos treinados a partir do notebook original, basta colocar o
arquivo `.keras`/`.h5` em `models/corn_disease_model.keras` (ajuste
`MODEL_PATH` no `.env` se usar outro nome/formato) — não é necessário treinar
de novo.

## Rodando a API

```bash
uvicorn app.main:app --reload
```

Docs interativas: http://localhost:8000/docs

### Endpoints

- `GET /health` — status da API e se o modelo está carregado.
- `GET /classes` — lista de classes suportadas.
- `POST /predict` — multipart/form-data, campo `file` com uma imagem. Retorna
  classe prevista, confiança e probabilidades por classe.
- `POST /predict/batch` — multipart/form-data, campo `files` (múltiplos
  arquivos). Útil para processar todas as imagens de um voo de drone de uma
  vez; retorna também um resumo com contagem por classe.

## Testes

```bash
pytest
```

## Docker

```bash
docker build -t corn-disease-backend .
docker run -p 8000:8000 -v ${PWD}/models:/app/models corn-disease-backend
```

## Roadmap / próximos passos

- [ ] Treinar o modelo com o dataset final do projeto e validar acurácia.
- [ ] Definir como as imagens do drone chegam ao backend (upload direto,
      fila/storage intermediário, streaming) — ajustar endpoints conforme.
- [ ] Autenticação entre front e backend (API key / JWT), se necessário.
- [ ] Deploy (cloud, servidor próprio, ou edge no próprio drone/ground station).
- [ ] Repositório GitHub para push deste backend (a ser enviado pelo usuário).
