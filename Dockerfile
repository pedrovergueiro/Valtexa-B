FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1: dependência nativa do backend que não vem na imagem slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# O índice /whl/cpu traz o torch sem as bibliotecas CUDA: a imagem cai de vários
# gigabytes para algumas centenas de MB, e a API serve em CPU de qualquer forma.
# Para servir em GPU, troque pelo índice cu12x correspondente à placa.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY app ./app

# Os pesos ficam fora da imagem — são binários grandes, versionados à parte, e
# entram por volume em tempo de execução:
#   docker run -v "$PWD/models:/app/models" ...
RUN mkdir -p /app/models \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV MODEL_PATH=/app/models/corn_disease_model.keras
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
