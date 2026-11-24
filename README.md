# Clasificación de la función retórica de citas y recuperación de pasajes

El proyecto clasifica el **contexto de una cita** en una de **9 funciones retóricas**
(*Application, Background, Basis, Comparison, Evidence, Further_Reading, Gap,
Identification_of_the_Originator, Modification_Improvement*) y recupera los
**pasajes más afines (Top-3 chunks)** del artículo citado. Se sirve como una API
(FastAPI) con un tablero web, empaquetados en Docker.

## Problema y contexto

La identificación de la función retórica que cumple una cita dentro de un
artículo científico (por ejemplo, si sirve de base, comparación o evidencia)
es una tarea compleja porque los contextos de cita suelen ser breves o
ambiguos, y la información relevante del artículo citado suele estar
distribuida en secciones específicas.

## Pregunta de negocio

¿Es posible, a partir de artículos científicos en inglés, recuperar
automáticamente los fragmentos más relevantes de un artículo citado y clasificar
el propósito de cada cita (función de cita) para apoyar tareas como la revisión
de literatura y el análisis de impacto bibliométrico?

## Alcance

Construcción de un dataset de intención de cita con 9 clases (contextos de cita
reales, partición sin fuga entre artículos citantes), entrenamiento y comparación
de clasificadores (línea base léxica, embeddings congelados, ajuste fino de
SciBERT y SPECTER) y despliegue del mejor modelo como servicio.

## Conjunto de datos

- **Dataset de intención de cita** (`dataset/`, versionado con DVC, ver `dataset.dvc`):
  9 clases con desbalance controlado consolidado a partir de SciTail, SciCite,
  Semantic Scholar y ACL-ARC. Partición final auditada sin fuga agrupada por
  documento citante (`seed = 42`): **14.937** registros totales (**10.416** de
  entrenamiento, **2.280** de validación y **2.241** de prueba revisados
  manualmente[cite: 5]; el manifiesto intermedio en `data/processed/dataset_manifest.json`
  registraba 20.655 ejemplos antes de la curaduría final sin fuga).
- **Corpus crudo de arXiv** (`data/raw/`, versionado con DVC, ver `data/raw.dvc`):
  artículos en inglés extraídos con `scripts/extract_arxiv_data.py`.

El script de extracción descarga por defecto todos los artículos que coincidan
con la consulta (sin límite fijo). Es incremental (guarda cada artículo apenas
se procesa) y reanudable: si se interrumpe, al volver a ejecutarlo continúa donde
quedó. El parámetro opcional `--max_results` limita la cantidad en pruebas
rápidas. Descargar el corpus completo puede tardar horas y ocupar varios GB:
córrelo en segundo plano (por ejemplo con `nohup`) y vigila el espacio en disco.

## Resultados de los modelos, hardware y latencia

### Especificaciones de Hardware

- **Entrenamiento y evaluación experimental:** Google Colab con GPU **NVIDIA L4**
  (secuencias máximas de 512 tokens, tamaño de lote 16, 4 épocas, optimizador
  AdamW con `lr = 2e-5` y calentamiento lineal del 10 %)[cite: 5].
  - Duración promedio de ajuste fino completo (*fine-tuning*): **29,7 min** por corrida[cite: 5].
  - Duración promedio de control congelado (*frozen encoder*): **8,5 min**[cite: 5].
- **Despliegue de producción (API + Tablero):** Contenedores Docker (`FastAPI` +
  `Nginx`) sobre instancia **AWS EC2** (`x86_64`, Ubuntu), con umbral operativo
  máximo (SLA) de **80 ms** por petición para operación interactiva en tiempo real[cite: 5].

### Comparativa de Modelos y Latencias (Partición Test sin fuga, `seed = 42`)

Métricas en la partición de prueba (detalle, curvas y matrices en
[`docs/MODELOS_ENTREGA2.md`](docs/MODELOS_ENTREGA2.md) y en MLflow):

| Corrida / Modelo | Configuración de entrada | Macro F1 | Micro F1 | Latencia promedio por petición (GPU, 512 tokens) | Tasa de procesamiento |
|---|---|:---:|:---:|:---:|:---:|
| **Run 1: SciBERT Metadata**[cite: 5] | Contexto + Título y resumen completo (Config. A)[cite: 5] | 0,608[cite: 5] | 0,624[cite: 5] | 44,0 ms[cite: 5] | 22,7 req/s[cite: 5] |
| **Run 2: SciBERT Top-3 Chunks (modelo recomendado)**[cite: 5] | Contexto + Sección IMRaD + Top-3 Chunks + pérdida ponderada (Config. B)[cite: 5] | **0,663**[cite: 5] | **0,678**[cite: 5] | **48,5 ms** *(11,71 ms mediana forward)*[cite: 5] | 20,6 req/s *(85,4 ej./s)*[cite: 5] |
| **Run 3: SPECTER Top-3 Chunks**[cite: 5] | Contexto + Sección IMRaD + Top-3 Chunks + pérdida ponderada (Config. B)[cite: 5] | 0,653[cite: 5] | 0,667[cite: 5] | 49,0 ms[cite: 5] | 20,4 req/s[cite: 5] |
| **Run 4: SciBERT Frozen (Control)**[cite: 5] | Config. B con codificador congelado (solo cabeza lineal)[cite: 5] | 0,367[cite: 5] | 0,442[cite: 5] | 22,0 ms[cite: 5] | 45,5 req/s[cite: 5] |
| **Run 5: SciBERT Sin Pesos (Ablación)**[cite: 5] | Config. B con entropía cruzada estándar sin reponderar[cite: 5] | 0,657[cite: 5] | 0,677[cite: 5] | 48,0 ms[cite: 5] | 20,8 req/s[cite: 5] |

*(Nota: la línea base léxica `v1: TF-IDF + Regresión Logística` registrada en
`models/scif-v1-tfidf-logreg/` se mantiene empaquetada en el repositorio como
alternativa ligera sin dependencia de pesos neuronales)*[cite: 5, 7].

## Estructura del repositorio

```
.
├── data/                 # datos crudos y procesados (versionados con DVC)
├── models/               # modelos empaquetados + cargador de referencia (scif_predict.py)
├── src/
│   ├── api/              # servicio de inferencia (FastAPI)
│   └── dashboard/        # tablero web estático (HTML/JS) que consume la API
├── scripts/              # extracción, preparación de datos, entrenamiento y métricas
├── tests/                # pruebas automatizadas (pytest)
├── notebooks/            # exploración y procesamiento de datos
├── docker/               # Dockerfiles, nginx y configuración del tablero
├── docker-compose.yml    # despliegue de API + tablero
├── infra/                # aprovisionamiento de EC2 y MLflow
├── docs/                 # EDA, resultados de modelos, manuales y soportes
├── .env.example          # plantilla de variables de entorno
└── .github/workflows/    # integración continua
```

## Variables de entorno

Copia la plantilla y rellénala (el archivo `.env` **no** se sube a Git):

```bash
cp .env.example .env
```

| Variable | Para qué sirve | Valor por defecto |
|---|---|---|
| `SCIF_PORT` | Puerto publicado del tablero con `docker compose` | `8080` |
| `SCIF_MODELS_DIR` | Carpeta de modelos de la API (sin Docker) | `<repo>/models` (`/models` en Docker) |
| `SCIF_ALLOWED_ORIGINS` | Orígenes CORS permitidos (separados por coma) | `http://localhost:8080,http://127.0.0.1:8080` |
| `MLFLOW_TRACKING_URI` | Servidor MLflow para `scripts/train.py` | `file:./mlruns` |
| `SCIF_SKIP_MODEL_LOGGING` | `1` evita guardar el modelo en MLflow | no definida |
| `AWS_ACESS_KEY` y `AWS_SECRET_ACCESS_KEY` | Credenciales AWS para DVC (S3) y `infra/`; las carga `source scripts/awsenv.sh` | vacías |

Nota: `AWS_ACESS_KEY` se escribe con una sola «C» en «ACESS» porque así la lee
`scripts/awsenv.sh`.

## Instrucciones de uso

1. Crear el entorno virtual del proyecto (Python 3.12). En Windows (Git Bash):

   ```bash
   python -m venv env-proj
   source env-proj/Scripts/activate
   pip install -r requirements-dev.txt
   ```

   En Linux/macOS:

   ```bash
   python3.12 -m venv env-proj
   source env-proj/bin/activate
   pip install -r requirements-dev.txt
   ```

2. Ejecutar el script de extracción:

   ```bash
   # Prueba rápida
   python scripts/extract_arxiv_data.py --query "cat:cs.CL" --max_results 100 --out_dir data/raw

   # Descarga completa (recomendado en segundo plano)
   nohup python scripts/extract_arxiv_data.py --query "cat:cs.CL" --out_dir data/raw > log.txt 2>&1 &
   ```

3. Versionar los datos con DVC y subirlos al remoto en S3 (necesitas las
   credenciales AWS: `source scripts/awsenv.sh`):

   ```bash
   dvc add data/raw
   git add data/raw.dvc data/.gitignore
   git commit -m "Datos extraídos de arXiv para citation contexts"
   dvc push
   git push
   ```

   Si el comando `dvc` no se reconoce en la terminal, usa `python -m dvc`
   (por ejemplo `python -m dvc push`).

4. Entrenamiento (entorno aparte, con las dependencias de entrenamiento):

   ```bash
   python -m venv .venv-train && source .venv-train/bin/activate   # en Windows: .venv-train/Scripts/activate
   pip install -r requirements-train.txt
   export MLFLOW_TRACKING_URI=http://localhost:5000     # opcional
   python scripts/train.py --help
   ```

   El procedimiento completo para regenerar SciBERT está en
   [`models/README.md`](models/README.md).

## Despliegue con Docker

Forma recomendada de levantar el prototipo: un solo comando y un solo puerto.

```bash
docker compose up -d --build     # tablero en http://localhost:8080
```

| Servicio | Imagen | Puerto | Función |
|---|---|---|---|
| `api` | `scif-api` (~410 MB) | interno (8000) | FastAPI con los modelos; no se publica al exterior |
| `dashboard` | `scif-dashboard` (~76 MB) | `8080` | nginx sirve el tablero y hace proxy de `/api/` hacia `api:8000` |

El navegador habla con un único origen, así que **no interviene CORS** y en la
nube basta abrir un puerto. Los pesos se montan desde `./models` en modo solo
lectura: no se copian a la imagen y siguen viniendo de DVC.

```bash
dvc pull models/scif-scibert.dvc       # pesos de SciBERT (~440 MB), antes de levantar
docker compose logs -f api             # registros
docker compose ps                      # estado y healthcheck
docker compose down                    # detener
SCIF_PORT=9000 docker compose up -d    # otro puerto
```

### Despliegue en EC2

```bash
ssh -i <llave>.pem ubuntu@<IP>
git clone <repo> && cd pds-proyecto-tema1
dvc pull models/scif-scibert.dvc      # pesos de SciBERT
bash infra/deploy_ec2.sh
```

El script instala Docker si falta, construye las imágenes **en la propia
instancia** (es x86_64; construirlas en un equipo arm64 daría una arquitectura
incompatible), levanta los servicios y espera al healthcheck. Hay que abrir el
puerto 8080/tcp en el grupo de seguridad. La API necesita salida a internet hacia
`arxiv.org` para la recuperación de pasajes. Más detalles en
[`infra/README.md`](infra/README.md) y [`docs/MANUAL_INSTALACION.md`](docs/MANUAL_INSTALACION.md).

## API de inferencia y tablero

El clasificador de función de cita y la recuperación de pasajes se sirven con
FastAPI y se operan desde un tablero web. El tablero **solo muestra información
devuelta por la API**: no incorpora datos simulados. Guía de uso del tablero:
[`docs/MANUAL_USUARIO.md`](docs/MANUAL_USUARIO.md).

Para desarrollo sin contenedores:

```bash
# Entorno del servicio (torch todavía no publica ruedas para Python 3.14)
python3.12 -m venv .venv-api && source .venv-api/bin/activate
pip install -r src/api/requirements.txt

# 1) API: desde la raíz del repo
uvicorn src.api.main:app --port 8000

# 2) Tablero: en otra terminal
python src/dashboard/serve.py        # http://127.0.0.1:8080
```

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Estado del servicio (úsalo como *health check*) |
| `GET` | `/models` | Modelos servibles, su disponibilidad y las 9 etiquetas canónicas |
| `POST` | `/predict` | Clasifica un `citation_context` + `rhetorical_section` |
| `POST` | `/retrieve` | Top-k pasajes del artículo citado (por `arxiv_id` o `cited_text`) más similares al contexto |

Documentación interactiva (Swagger): `http://localhost:8000/docs` en desarrollo.

### Ejemplos con `curl`

Con Docker, la API se alcanza a través del tablero: usa `http://localhost:8080/api`.
En desarrollo sin contenedores usa `http://localhost:8000`.

```bash
# 1) Estado del servicio
curl http://localhost:8080/api/

# 2) Modelos disponibles
curl http://localhost:8080/api/models

# 3) Clasificar una cita
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "citation_context": "In contrast to previous approaches that rely solely on surface lexical cues, our framework models discourse-level rhetorical intent.",
        "rhetorical_section": "Introduction",
        "model": "scibert"
      }'

# 4) Recuperar los 3 pasajes más afines del artículo citado
curl -X POST http://localhost:8080/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{
        "citation_context": "We build our citation encoder on top of SciBERT and keep its in-domain vocabulary unchanged.",
        "arxiv_id": "1903.10676",
        "top_k": 3
      }'
```

Respuesta de `/predict` (valores ilustrativos):

```json
{
  "predicted_label": "Comparison",
  "predicted_label_display": "Comparison",
  "confidence": 0.84,
  "probabilities": { "Comparison": 0.84, "Gap": 0.08, "...": 0.0 },
  "ambiguous": false,
  "model_id": "scibert",
  "input_text": "<contexto> [SEC] <sección canónica>",
  "latency_ms": 88.4,
  "status": "success"
}
```

`ambiguous` vale `true` cuando la diferencia entre las dos probabilidades más
altas es menor a `0.10`.

La recuperación descarga el PDF de arXiv, lo segmenta en fragmentos de hasta 300
palabras respetando oraciones y secciones, y los compara con el contexto de cita
usando el mismo encoder SciBERT del clasificador (*mean pooling* + coseno). Los
documentos ya procesados quedan en caché en memoria. Requiere salida a internet
hacia `arxiv.org` y los pesos de SciBERT.

## Modelos y línea base léxica

La API conoce dos modelos (ver `src/api/model_loader.py`):

| `id` (campo `model`) | Carpeta | Origen de los pesos |
|---|---|---|
| `scibert` (por defecto, recomendado) | `models/scif-scibert/` | `dvc pull models/scif-scibert.dvc` |
| `tfidf-logreg` (línea base, ~5 MB) | `models/scif-v1-tfidf-logreg/` | versionado en Git |

Los pesos se buscan en `models/` o en la ruta de `SCIF_MODELS_DIR`.

Si los pesos de SciBERT no están en disco, **la API arranca igual**, pero:

- `/predict` con `"model": "scibert"` responde `503` (indica que falta `dvc pull`).
- La línea base sigue disponible: se elige con `"model": "tfidf-logreg"` (o en el
  tablero, donde los modelos sin pesos aparecen deshabilitados).
- `/retrieve` responde `503`, porque usa el encoder de SciBERT.

Si los pesos de SciBERT no están en disco, **la API arranca igual** manteniendo
continuidad operativa mediante selección explícita del modelo disponible:

- `/predict` con `"model": "scibert"` responde `503` (indicando que falta ejecutar `dvc pull`).
- La línea base léxica sigue operativa seleccionando `"model": "tfidf-logreg"` en la petición o desde el tablero (donde los modelos sin pesos locales aparecen deshabilitados).
- `/retrieve` responde `503`, porque requiere el encoder denso de SciBERT.

## Pruebas

```bash
pytest
```

Las pruebas corren automáticamente en cada push y pull request a `main` mediante
`.github/workflows/tests.yml` (Python 3.12).
