# Datos

Esta carpeta trae una muestra del dataset y la descripción de su estructura. El
dataset completo (83 MB) está versionado con DVC en `dataset/` y se descarga sin
credenciales:

| Forma | Cómo |
|---|---|
| Enlace directo | [dataset.zip](https://scif-dvc-e645d210.s3.us-east-2.amazonaws.com/descargas/dataset.zip): se descomprime en la raíz del repositorio y crea `dataset/` |
| DVC | `pip install dvc` y luego `dvc pull -r publico dataset.dvc` |

## Muestra

[`muestra_test.csv`](muestra_test.csv) tiene 90 citas del conjunto de prueba, 10 por
cada función según la revisión humana (`human_label`), elegidas al azar con semilla 42.
Tiene las mismas columnas que `dataset/test_enriquecido_top3_chunks.csv`.

## Archivos del dataset completo

| Archivo | Filas | Qué es |
|---|--:|---|
| `train_enriquecido_top3_chunks.csv` | 10.416 | Entrenamiento |
| `val_enriquecido_top3_chunks.csv` | 2.280 | Validación |
| `test_enriquecido_top3_chunks.csv` | 2.241 | Prueba, con la revisión humana (`human_label`) |
| `dataset_definitivo_total_chunks.csv` | 14.937 | Las tres particiones juntas |
| `checkpoint_citas_clasificadas_llama.csv` | 41.172 | Pre-etiquetado con Llama 3.2 3B, antes del balanceo |
| `anotacion/hoja_anotacion_test.xlsx` | 2.241 | Segunda anotación independiente del test, usada para el acuerdo entre anotadores |
| `jueces/` | | Respuestas de los tres modelos jueces y su voto (`scripts/evaluate_jueces.py`) |
| `evaluacion_llm/` | | Respuestas de los modelos de lenguaje en la evaluación (`scripts/evaluate_llm.py`) |

La partición se hizo por artículo citante: ningún artículo que cita aparece en dos
particiones.

## Fuentes

| `source_dataset` | Citas | Origen |
|---|--:|---|
| `scitail` | 6.152 | SciTail: premisas e hipótesis de ciencia |
| `scicite` | 4.536 | SciCite |
| `Semantic_Scholar_*` | 3.998 | API de Semantic Scholar, con varias estrategias de búsqueda |
| `acl_arc` | 251 | ACL-ARC |

## Columnas

| Columna | Descripción |
|---|---|
| `citation_context` | Oración o fragmento donde aparece la cita |
| `rhetorical_section` | Sección del artículo citante donde está la cita |
| `citing_paper_id`, `cited_paper_id` | Identificadores del artículo que cita y del citado |
| `pair_id` | Par citante y citado, separados por `__` |
| `source_dataset` | Fuente de la cita (tabla anterior) |
| `sentence2` | Premisa de SciTail, cuando la cita viene de esa fuente |
| `cited_paper_title`, `cited_paper_abstract` | Título y abstract del artículo citado, cuando se obtuvieron |
| `tiene_evidencia` | Si hay abstract, título o premisa del artículo citado |
| `top{1,2,3}_cited_chunk` | Las 3 oraciones del artículo citado más parecidas a la cita, elegidas con MiniLM |
| `top{1,2,3}_section` | De dónde sale cada oración: `Abstract`, `Title` o `Hypothesis_Premise` |
| `top{1,2,3}_similarity` | Similitud coseno con la cita |
| `top{1,2,3}_position_char` | Posición de la oración dentro del texto de origen |
| `label` | Etiqueta automática (pre-etiquetado con Llama), con la que se entrenó |
| `human_label` | Solo en prueba: etiqueta de la revisión humana |
| `certeza_evaluacion` | Solo en prueba: `ALTA_CERTEZA` si el puntaje de Llama para `label` es de al menos 0,70, `DUDA` si no |
| `etiqueta_original`, `fuente_unificada` | Solo en entrenamiento y validación: etiqueta de la fuente original y nombre de la fuente |

Las 9 funciones (`label` y `human_label`) son `Application`, `Background`, `Basis`,
`Comparison`, `Evidence`, `Further_Reading`, `Gap`,
`Identification_of_the_Originator` y `Modification_Improvement`. Su definición está
en la [guía de anotación](../docs/anotacion/GUIA_ANOTACION.md).
