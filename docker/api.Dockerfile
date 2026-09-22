# Servicio de inferencia: clasificación de función de cita + recuperación de pasajes.
#
# Los pesos de los modelos NO se copian a la imagen: se montan como volumen de solo
# lectura en /models (ver docker-compose.yml). Así la imagen no crece 420 MB y los
# pesos siguen viniendo de DVC, que es donde están versionados.
#
# Python 3.12: torch todavía no publica ruedas para 3.13/3.14.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SCIF_MODELS_DIR=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# torch desde el índice CPU de PyTorch: la rueda por defecto arrastra las librerías
# CUDA y multiplica por seis el tamaño de la imagen. Aquí nunca hay GPU.
COPY src/api/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
                --extra-index-url https://pypi.org/simple \
                -r /tmp/requirements.txt \
 && rm -rf /root/.cache

# Código: el cargador de referencia y el servicio
COPY models/__init__.py models/scif_predict.py ./models/
COPY src/__init__.py ./src/
COPY src/api ./src/api

# Usuario sin privilegios
RUN useradd --create-home --uid 10001 scif && chown -R scif:scif /app
USER scif

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4).status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
