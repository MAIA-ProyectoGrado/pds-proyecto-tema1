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
│   ├── raw/          # datos crudos descargados de arXiv (versionados con DVC)
│   └── processed/     # datos ya limpios/estructurados (versionados con DVC)
├── scripts/           # scripts de extracción y procesamiento de datos
├── tests/             # pruebas automatizadas
├── notebooks/         # exploración de datos
├── docs/               # documentación adicional (maqueta, diagramas, etc.)
├── src/                 # código fuente del modelo/API/tablero (próximas entregas)
└── .github/workflows/   # pipeline de integración continua
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

## Pruebas

Las pruebas automatizadas se ejecutan con:

```bash
pytest
```

y corren automáticamente en cada push mediante el pipeline definido en
`.github/workflows/tests.yml`.
