# Recomendación local de citas y clasificación de funciones de cita

## Novedades de la Entrega 2 (rama `feature/entrega2-eda-modelos`)

- **Datos:** se reemplaza el corpus de abstracts por el **dataset enriquecido de
  intención de cita** con contextos de cita reales y 9 clases balanceadas
  (`dataset/`, versionado con DVC vía `scripts/prepare_dataset.py`).
- **EDA:** `scripts/eda_entrega2.py` → `docs/EDA_ENTREGA2.md` (explora la variable
  objetivo y su balance, y documenta la inconsistencia del Top-3 de *chunks*).
- **Modelos:** `scripts/train.py` entrena **v1** (TF-IDF + LogReg) y **v2**
  (SciBERT fine-tune) con **MLflow**; `scripts/analyze_metrics.py` genera
  `docs/MODELOS_ENTREGA2.md` (comparativa + sobreajuste).
- **Infra:** `infra/` levanta una EC2 con MLflow (`launch_ec2.sh`, `stop_ec2.sh`).
- **Handoff:** `PROMPT_AUTONOMO_ENTREGA2.md` (ejecuta partes 2 y 3) y
  `MEJORAS_ENTREGA2.md` (mapa feedback Entrega 1 → cambios, con página/párrafo).

## Problema y contexto

La identificación de la función retórica que cumple una cita dentro de un
artículo científico (por ejemplo, si sirve de base, comparación o evidencia)
es una tarea compleja porque los contextos de cita suelen ser breves o
ambiguos, y la información relevante del artículo citado suele estar
distribuida en secciones específicas.

## Pregunta de negocio

¿Es posible, a partir de artículos científicos en inglés extraídos de arXiv,
recuperar automáticamente los fragmentos más relevantes de un artículo citado
y clasificar el propósito de cada cita (función de cita) para apoyar tareas
como la revisión de literatura y el análisis de impacto bibliométrico?

## Alcance

Extracción de un corpus de artículos de arXiv (área de computación),
identificación de contextos de cita, y construcción de un dataset balanceado
de al menos 2.000 ejemplos por categoría de función de cita, para entrenar
clasificadores y evaluar modelos de lenguaje.

## Conjunto de datos

Artículos científicos en inglés extraídos del repositorio público arXiv
(formato PDF), procesados con `scripts/extract_arxiv_data.py`.

Por defecto, el script descarga todos los artículos que coincidan con la
consulta de búsqueda (sin límite fijo). El proceso es incremental (cada
artículo se guarda apenas se procesa) y reanudable: si se interrumpe, al
volver a ejecutarlo continúa donde quedó sin repetir descargas. Existe un
parámetro opcional `--max_results` para limitar la cantidad de artículos en
pruebas rápidas.

Descargar el corpus completo puede tardar horas y ocupar varios GB, por lo
que se recomienda correrlo en segundo plano (por ejemplo con `nohup`) y
vigilar el espacio en disco disponible.

## Estructura del repositorio

```
.
├── data/
│   ├── raw/            # datos crudos descargados de arXiv (versionados con DVC)
│   └── processed/      # datos ya limpios/estructurados (versionados con DVC)
├── models/             # modelos empaquetados listos para servir + cargador de referencia
├── src/
│   ├── api/            # servicio de inferencia (FastAPI)
│   └── dashboard/      # tablero que consume la API
├── scripts/            # extracción, preparación de datos, entrenamiento y métricas
├── tests/              # pruebas automatizadas
├── notebooks/          # exploración de datos
├── infra/              # aprovisionamiento de EC2 y MLflow
├── docs/               # EDA, resultados de modelos y soportes
└── .github/workflows/  # pipeline de integración continua
```

## Instrucciones de uso

1. Activar el entorno virtual del proyecto (ya creado por `setup_repo.sh`
   junto con las dependencias instaladas). En Windows (Git Bash):

   ```bash
   source env-proj/Scripts/activate
   ```

   En Linux/macOS:

   ```bash
   source env-proj/bin/activate
   ```

   Si necesitas recrearlo desde cero:

   ```bash
   python -m venv env-proj
   source env-proj/Scripts/activate   # o env-proj/bin/activate en Linux/macOS
   pip install -r requirements-dev.txt
   ```

2. Ejecutar el script de extracción:

   ```bash
   # Prueba rápida
   python scripts/extract_arxiv_data.py --query "cat:cs.CL" --max_results 100 --out_dir data/raw

   # Descarga completa (recomendado en segundo plano)
   nohup python scripts/extract_arxiv_data.py --query "cat:cs.CL" --out_dir data/raw > log.txt 2>&1 &
   ```

3. Versionar los datos con DVC y subirlos al remoto en S3:

   ```bash
   dvc add data/raw
   git add data/raw.dvc data/.gitignore
   git commit -m "Datos extraídos de arXiv para citation contexts"
   dvc push
   git push
   ```

   Si el comando `dvc` no se reconoce en la terminal, usa `python -m dvc`
   en su lugar (por ejemplo `python -m dvc push`).

   ```
   ```

## Despliegue con Docker

Forma recomendada de levantar el prototipo: un solo comando y un solo puerto.

```bash
docker compose up -d --build     # tablero en http://localhost:8080
```

Arquitectura de los contenedores:

| Servicio | Imagen | Puerto | Función |
|---|---|---|---|
| `api` | `scif-api` (~410 MB) | interno | FastAPI con los modelos; no se publica al exterior |
| `dashboard` | `scif-dashboard` (~76 MB) | `8080` | nginx sirve el tablero y hace proxy de `/api/` hacia `api:8000` |

El navegador habla con un único origen, así que **no interviene CORS** y en la nube
basta abrir un puerto. Los pesos se montan desde `./models` en modo solo lectura:
no se copian a la imagen, siguen viniendo de DVC. Si falta `models/scif-scibert/`,
la API arranca igual y sirve la línea base; la recuperación de pasajes responde
`503` y el tablero lo indica.

```bash
docker compose logs -f api     # registros
docker compose ps              # estado y healthcheck
docker compose down            # detener
SCIF_PORT=9000 docker compose up -d   # otro puerto
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
`arxiv.org` para la recuperación de pasajes.

## API de inferencia y tablero

El clasificador de función de cita y la recuperación de pasajes del artículo citado
se sirven con FastAPI y se operan desde un tablero web. El tablero **solo muestra
información devuelta por la API**: no incorpora datos simulados.

Para desarrollo sin contenedores:

```bash
# Entorno del servicio (torch todavía no publica ruedas para Python 3.14)
python3.12 -m venv .venv-api && source .venv-api/bin/activate
pip install -r src/api/requirements.txt

# 1) API — desde la raíz del repo
uvicorn src.api.main:app --port 8000

# 2) Tablero — en otra terminal
python src/dashboard/serve.py        # http://127.0.0.1:8080
```

Endpoints:

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Estado del servicio |
| `GET` | `/models` | Modelos servibles, sus métricas y las 9 etiquetas canónicas |
| `POST` | `/predict` | Clasifica un `citation_context` + `rhetorical_section` |
| `POST` | `/retrieve` | Top-k pasajes del artículo citado (por `arxiv_id` o `cited_text`) más similares al contexto |

La recuperación descarga el PDF de arXiv, lo segmenta en fragmentos de hasta 300
palabras respetando oraciones y secciones, y los compara con el contexto de cita
usando el mismo encoder SciBERT del clasificador (mean pooling + coseno). Los
documentos ya procesados quedan en caché en memoria. Requiere salida a internet
hacia `arxiv.org` y los pesos de SciBERT.

Los pesos se buscan en `models/` (o en la ruta que indique la variable
`SCIF_MODELS_DIR`). `scif-v1-tfidf-logreg` está versionado en Git;
`scif-scibert` se obtiene con `dvc pull`. Si SciBERT no está presente, la API
arranca igual y sirve la línea base.

## Pruebas

Las pruebas automatizadas se ejecutan con:

```bash
pytest
```

y corren automáticamente en cada push mediante el pipeline definido en
`.github/workflows/tests.yml`.

## Respaldo Léxico Autónomo (Fallback Operativo)

Para garantizar alta disponibilidad y continuidad del servicio de inferencia sin depender estrictamente de la descarga o carga en memoria de los pesos pesados de Transformers (`SciBERT` / `SPECTER2`, >400 MB), la API de FastAPI implementa un mecanismo automático de **degradación elegante (graceful fallback)** hacia la línea base léxica (`TF-IDF + Regresión Logística`).

### Funcionamiento del Mecanismo
1. **Verificación al arranque y en inferencia:** Al iniciar el contenedor o recibir una solicitud en `/predict`, el gestor de modelos verifica la existencia e integridad de los artefactos densos en `MODEL_DIR` (`models/scibert_finetuned`).
2. **Activación automática del respaldo:** Si los pesos pesados no están presentes (por ejemplo, en despliegues ligeros sin descarga previa desde AWS S3/DVC, entornos con restricciones de RAM o fallos de carga), el sistema redirecciona automáticamente la inferencia al artefacto liviano definido en `FALLBACK_MODEL_PATH` (`models/tfidf_logreg.joblib`, <5 MB).
3. **Transparencia en la respuesta JSON:** El endpoint `/predict` mantiene exactamente el mismo contrato de salida (categoría predicha, distribución probabilística de 9 clases, alerta de ambigüedad si la brecha es `< 0.10` y latencia en ms), indicando explícitamente en los metadatos de respuesta (`model_used: "tfidf-fallback"`) que la predicción fue atendida en modo autónomo ligero.

### Ventajas Operativas
* **Cero tiempo de inactividad (Zero-Downtime):** Permite levantar y probar el microservicio y el tablero Dash en segundos tras clonar el repositorio, incluso sin credenciales activas de S3/DVC.
* **Bajo consumo de recursos:** Opera en instancias CPU mínimas con latencias inferiores a 5 ms por consulta.




## Especificaciones de Hardware y Rendimiento

### Infraestructura de Cómputo (AWS EC2)
* **Entorno de Producción / Inferencia (API + Tablero):** Instancia AWS EC2 `t3.large` (2 vCPUs Intel Xeon, 8 GB RAM, almacenamiento EBS gp3) ejecutando contenedores Docker aislados para FastAPI (`puerto 8000`) y Dash (`puerto 8050`).
* **Entorno de Entrenamiento (Fine-Tuning):** Instancia escalada a **32 vCPUs** (64 GB RAM) para el ajuste fino supervisado de arquitecturas Transformer (`max_len=256`, `batch_size=32`, 4 épocas, ~75 minutos por modelo) con trazabilidad en MLflow y versionamiento de datos en DVC/S3.

### Tabla Comparativa de Latencia y Desempeño (Partición de Prueba)

| Modelo | Tipo de Arquitectura | F1-Macro (Test) | Latencia Promedio (CPU) | Tamaño Artefacto |
| :--- | :--- | :---: | :---: | :---: |
| **v1 — TF-IDF + Reg. Logística** | Léxico (Respaldo) | `0.520` | `< 5 ms` | `~4 MB` |
| **v2a — Embeddings Congelados** | Semántico Estático | `0.440` | `~25 ms` | `~90 MB` |
| **v2b — DistilBERT Fine-Tuned** | Encoder Ligero | `0.580` | `~45 ms` | `~260 MB` |
| **v3 — SPECTER2 Fine-Tuned** | Encoder Científico | `0.621` | `~95 ms` | `~440 MB` |
| **v3 — SciBERT Fine-Tuned (Activo)** | Encoder Científico | **`0.634`** | `~90 ms` | `~440 MB` |

---

## Ejemplos de Consulta a la API (`curl`)

### 1. Verificar estado del servicio (`/health`)
```bash
curl -X GET "http://localhost:8000/health" \
     -H "Accept: application/json"
```

### 2. Clasificar la función retórica de una cita (`/predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "citation_context": "In contrast to previous approaches that rely solely on surface lexical cues, our framework models discourse-level rhetorical intent.",
           "rhetorical_section": "Introduction"
         }'
```

**Ejemplo de respuesta JSON:**
```json
{
  "predicted_label": "Comparison",
  "confidence": 0.84,
  "ambiguity_warning": false,
  "model_used": "scibert-finetuned",
  "latency_ms": 88.4,
  "probabilities": {
    "Comparison": 0.84,
    "Gap": 0.08,
    "Background": 0.04,
    "Basis": 0.01,
    "Application": 0.01,
    "Modification_Improvement": 0.01,
    "Evidence": 0.005,
    "Identification_of_the_Originator": 0.003,
    "Further_Reading": 0.002
  }
}
```