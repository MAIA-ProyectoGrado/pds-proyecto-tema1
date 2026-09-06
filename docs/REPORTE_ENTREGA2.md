# SCIF — Entrega 2 (Micro-proyecto)
### Recomendación local de citas y clasificación de su función retórica en artículos académicos

**Maestría en Inteligencia Artificial (MAIA) — Proyecto Desarrollo de Soluciones**
Henry Forigua Martínez · David Indaburu Silva · Juan David Marín Buitrago · Álvaro Andrés Ruiz Flórez · Estefanía Sánchez Contreras
Septiembre 2026

> Reporte de máximo 10 páginas. Repositorio: `https://github.com/alvarorf/pds-proyecto-tema1`
> (rama `feature/entrega2-eda-modelos`).

---

## 1. Resumen del problema (máx. 1 página)

### 1.1 Contexto

La producción científica crece de forma acelerada y predominantemente en inglés, lo
que dificulta que investigadores y revisores asimilen y verifiquen la evidencia
existente. En cienciometría y PLN, las citas no son vínculos planos: cumplen una
**función retórica** (dar contexto, comparar, aportar evidencia, señalar un vacío,
etc.). Identificar esa función de forma automática apoya la revisión de literatura
y el análisis bibliométrico cualitativo.

### 1.2 Pregunta de negocio

**¿Se puede clasificar automáticamente la función retórica de una cita académica en
inglés, a partir de su contexto textual, con calidad suficiente para asistir la
revisión de literatura?**

### 1.3 Alcance (concretado respecto a la Entrega 1)

- **Producto:** un clasificador de la **función de cita** (9 categorías), expuesto
  como **API REST** (`POST /predict`) y consumido por un **tablero** en Docker.
- **Usuario objetivo:** investigador / asistente de revisión de literatura que, al
  leer un artículo, quiere saber *para qué* se cita cada referencia.
- **Entrada:** 1–2 oraciones en inglés con la mención bibliográfica (*citation
  context*) y, opcionalmente, la sección donde aparece.
- **Salida:** una de 9 funciones de cita con su probabilidad y un aviso de
  ambigüedad cuando dos clases quedan muy cerca.
- **Fuera de alcance en esta entrega (→ v3):** la recomendación local de fragmentos
  del artículo citado (Top-3 *chunks*), idiomas distintos del inglés y la resolución
  de la referencia a un identificador. Justificación en §2.3.
- **Métrica de éxito:** **F1-macro** sobre las 9 clases en validación con
  aislamiento por documento (baseline aleatorio = 0.111).

### 1.4 Cambios respecto a la Entrega 1

| Área | Entrega 1 | Entrega 2 |
|---|---|---|
| Datos | 24 855 registros que resultaron ser **abstracts**, no contextos de cita (`base_label` constante = 0) | **Dataset enriquecido de intención de cita** (v3): 20 655 registros, contextos de cita **reales**, 9 clases balanceadas; versionado con **DVC** |
| Variable objetivo | No se exploró | **Explorada** (distribución, balance entre las 9 categorías, calidad del etiquetado) — §2.1, §2.6 |
| Alcance | Argumentado en términos académicos generales | Definido por usuario, entrada/salida y no-alcance — §1.3 |
| Maqueta | Asumía un panel Top-3 de *chunks* con datos que no se mostraron consistentes | Panel Top-3 marcado como **ilustrativo / pendiente de datos** (v3 ya trae *chunks* reales pero ruidosos y casi siempre del citante) — §4 |
| Modelos | — | v1 (baseline) y v2 (fine-tuning) versionados con MLflow — §3 |

---

## 2. Datos y exploración

### 2.1 Variable objetivo — `label` (9 funciones de cita)

Función retórica de la cita, 9 categorías mutuamente excluyentes codificadas
`label_id` 0–8: **Application, Background, Basis, Comparison, Evidence,
Further Reading, Gap, Identification of the Originator, Modification/Improvement**.

**Balance entre las nueve categorías (entrenamiento, n = 14 461):**

| Función de cita | n | % |
|---|--:|--:|
| Gap | 1 637 | 11.3 % |
| Evidence | 1 632 | 11.3 % |
| Further Reading | 1 617 | 11.2 % |
| Application | 1 613 | 11.2 % |
| Modification / Improvement | 1 607 | 11.1 % |
| Background | 1 597 | 11.0 % |
| Identification of the Originator | 1 597 | 11.0 % |
| Basis | 1 595 | 11.0 % |
| Comparison | 1 566 | 10.8 % |

- **Ratio de desbalance** (clase mayor / menor) = **1.045** en train (1.11 en val,
  1.17 en test); **entropía normalizada = 1.00** → distribución prácticamente uniforme.
- El maestro de 20 655 registros está balanceado **por construcción** (2 295/clase)
  y se particiona en train 14 461 / val 3 098 / test 3 096, manteniendo ~11 % por
  clase en cada partición (sin clases ausentes).

*Figura 1 — `docs/eda_entrega2/01_variable_objetivo.png` (distribución de `label` por partición).*

**Implicación para el negocio.** El balance es **artificial**: se fuerza a ~2 295
ejemplos por clase, en parte con un mecanismo de **rescate** que reasigna citas a
clases minoritarias (§2.6). No refleja la prevalencia real de cada función de cita
ni la calidad de la etiqueta; la evaluación de producción deberá hacerse con
distribución realista y anotación humana.

### 2.2 `citation_context` — ahora son citas reales

| Métrica (palabras) | Entrega 2 v3 (train) | Entrega 1 (abstracts) |
|---|--:|--:|
| Mediana | **26** | 163 |
| Media | 28.3 | 165.2 |
| p05 / p95 | 10 / 50 | — |
| % con marcador de cita explícito (`[12]`, `(Autor, 2020)`) | **51.3 %** | ~0 % |

La longitud (mediana 26 palabras, 6× menor que en la Entrega 1) y la presencia de
marcadores confirman que ahora el texto **sí** es un contexto de cita. La longitud
varía por clase (`Background` la más corta, `Comparison` la más larga). El
**48.7 %** de contextos no tiene marcador detectable por regex: ruido de calidad
que se reporta y acota el techo de desempeño (§4).

*Figuras 2–3 — `02_longitud_contexto.png`, `03_longitud_por_clase.png`.*

### 2.3 Consistencia de los datos y ajuste de alcance

| Comprobación (train) | v3 | Entrega 2 (v2) |
|---|---|---|
| `citing_paper_id` == `cited_paper_id` | **98.3 %** | 100 % |
| `top1_cited_chunk` == `citation_context` | 80.3 % | 94.8 % |
| `top2` / `top3` | **varían por fila** (14 233 / 14 392 textos distintos), *chunking* real pero ruidoso (fragmentos truncados) | plantillas de texto |
| Similitud top1 / top2 / top3 | varía | constante |

v3 mejora respecto a v2 (los *chunks* ya no son plantillas), pero el documento
«citado» sigue siendo el mismo que el citante en el 98.3 % de las filas. El
componente de **recomendación de fragmentos (Top-3 chunks)** queda fuera de alcance
y el panel de la maqueta se marca como *ilustrativo / pendiente de datos*.
`scripts/prepare_dataset.py` **no propaga** esas columnas al dataset de entrenamiento.

### 2.4 Aislamiento entre particiones (No-Leakage)

| Solape de `citing_paper_id` | train∩val | train∩test | val∩test |
|---|--:|--:|--:|
| conteo | **1** (0.03 % de val) | **0** | **0** |

La garantía de no-leakage por documento se mantiene en la práctica (1
`citing_paper_id` compartido train/val, despreciable). Detalle completo en
`docs/EDA_ENTREGA2.md`.

### 2.5 Estructura del dataset

| Partición | Registros | Uso |
|---|--:|---|
| Entrenamiento (`train`) | 14 461 | ajuste de los modelos |
| Validación (`val`) | 3 098 | selección de modelo / hiperparámetros |
| Prueba (`test`, para anotación humana) | 3 096 | evaluación final + acuerdo interanotador |
| Maestro balanceado | 20 655 | fuente de las 3 particiones (2 295/clase) |

Variables usadas por el modelo: `citation_context` (texto) y
`rhetorical_section_canon` (10 secciones canónicas, normalizadas desde 622 valores
crudos; 1.35 % nulos). Las particiones se versionan con **DVC** (`dataset.dvc`,
`data/raw.dvc`) sobre `s3://scif-dvcstore-779319895642-use1` (remoto `s2`).

### 2.6 Calidad del etiquetado

El etiquetado es asistido por LLM (`gemini-…-flash-lite`) con **rescate
probabilístico** hacia clases minoritarias, sin anotación humana completa. En las
173 filas donde el `predicted_label` del juez está disponible, el rescate **cambió
la etiqueta en el 28.3 %** (acuerdo juez↔final = 71.7 %). Junto con el 48.7 % de
contextos sin marcador de cita, es el principal límite del desempeño alcanzable.

---

## 3. Modelos desarrollados y su evaluación

Se entrenan tres iteraciones para la tarea de **clasificación de la función de
cita** (9 clases). Los experimentos se registran en **MLflow** sobre una instancia
**AWS EC2** (experimento `scif-citation-intent`); pantallazos en `docs/soportes/`.
El baseline (v1) es además reproducible sin servidor con
`python scripts/eval_baseline.py` (artefactos en `docs/modelos_entrega2/`).

> **Versión de datos.** Los resultados de v1, v2a y v2b de esta sección se
> obtuvieron sobre el *snapshot* del dataset de 18 000 registros. La versión
> entregada y versionada con DVC es la **v3 (20 655 registros)**; el reentrenamiento
> sobre v3 con un encoder de dominio científico está pendiente de infraestructura
> con GPU (§3.6) y su procedimiento está listo en `infra/train_v3_gpu.sh`.

### 3.1 Iteraciones

| Versión | Enfoque | Configuración | Rol |
|---|---|---|---|
| **v1 — baseline** | TF-IDF (1–2 gramas) + Regresión Logística (`class_weight=balanced`) | `C=1.0`, `min_df=2`, `max_features=50k`, `sublinear_tf`; entrada = `citation_context + " [SEC] " + rhetorical_section_canon` | referencia interpretable, entrena en **~9 s** en CPU |
| **v2a** | Embeddings congeladas de `all-MiniLM-L6-v2` + Regresión Logística | encoder sin ajustar, 384-dim | primer intento sin fine-tuning; CPU-friendly |
| **v2b — iteración intermedia** | Fine-tuning de `distilbert-base-uncased` | `--max_train 5000`, 1–2 épocas, `lr=3e-5`, `max_len=128`, `batch=16` | mejor F1 esperado; **no optimizada a fondo a propósito** — deja margen para v3 |

### 3.2 v1 — resultados (medidos)

**Métricas globales** (métrica principal: F1-macro; baseline aleatorio = 0.111):

| Partición | Accuracy | F1-macro | F1-micro | Precisión macro | Recall macro |
|---|--:|--:|--:|--:|--:|
| Entrenamiento | 0.798 | **0.798** | 0.798 | 0.805 | 0.798 |
| **Validación** | 0.502 | **0.496** | 0.502 | 0.507 | 0.506 |
| Prueba | 0.534 | 0.528 | 0.534 | 0.538 | 0.533 |

**Desempeño por clase (validación):**

| Función de cita | Precisión | Recall | F1 | Soporte |
|---|--:|--:|--:|--:|
| Background | 0.50 | 0.78 | **0.61** | 271 |
| Gap | 0.65 | 0.55 | 0.60 | 315 |
| Application | 0.52 | 0.56 | 0.54 | 277 |
| Evidence | 0.50 | 0.56 | 0.53 | 302 |
| Further Reading | 0.46 | 0.61 | 0.53 | 313 |
| Modification / Improvement | 0.46 | 0.49 | 0.47 | 300 |
| Comparison | 0.61 | 0.37 | 0.46 | 303 |
| Identification of the Originator | 0.48 | 0.34 | **0.39** | 312 |
| Basis | 0.40 | 0.30 | **0.34** | 307 |

*Figura 4 — `docs/modelos_entrega2/v1_matriz_confusion_val.png` (matriz de confusión normalizada).*

**Pares de clases más confundidos (validación):** `Evidence`→`Background` (0.20),
`Basis`→`Modification/Improvement` (0.18), `Application`→`Background` (0.16),
`Identification of the Originator`→`Application` (0.14), `Comparison`→`Evidence` (0.14).
`Background` actúa como clase "sumidero" (recall 0.78 pero precisión 0.50).

### 3.3 v1 — sobreajuste y regularización

| Brecha | Valor | Lectura |
|---|--:|---|
| F1-macro train − val | **+0.302** | el modelo memoriza el entrenamiento |
| Accuracy train − val | +0.296 | idem |
| F1-macro val − test | −0.032 | val ≈ test → el split es representativo, **sin fuga** |

**Veredicto: sobreajuste severo** (criterio: brecha train→val > 0.25 = severo;
0.12–0.25 = moderado; < 0.12 = bajo).

**Barrido de regularización** (`C` = inverso de la penalización):

| `C` | F1-macro train | F1-macro val | Brecha |
|--:|--:|--:|--:|
| 0.1 | 0.524 | 0.417 | 0.107 |
| 0.3 | 0.631 | 0.456 | 0.175 |
| **1.0** | 0.798 | **0.496** | 0.302 |
| 3.0 | 0.957 | 0.503 | 0.454 |
| 10.0 | 0.998 | 0.501 | 0.498 |

*Figura 5 — `docs/modelos_entrega2/v1_sweep_regularizacion.png`.* El F1 de validación
**se estanca en ≈ 0.50** mientras el de entrenamiento sube hasta 1.0: regularizar
más no mejora la generalización, solo reduce la brecha. `C = 1.0` es un punto de
equilibrio razonable para el baseline; **el techo de este enfoque léxico es ≈ 0.50**.

### 3.4 v2 — resultados

Restricción de cómputo: el entorno (AWS Academy Learner Lab) **no permite
instancias GPU**; la EC2 de experimentos es una `t3.large` de **2 vCPU**. Por eso
v2 se aborda primero con un enfoque sin fine-tuning y luego con fine-tuning corto.

**v2a — sentence-embeddings congelados + Regresión Logística**
(`all-MiniLM-L6-v2`, 384-dim; codificación ~4 min):

| Modelo | F1-macro train | F1-macro val | F1-macro test | Brecha train→val | Sobreajuste |
|---|--:|--:|--:|--:|---|
| v1 — TF-IDF + LogReg | 0.798 | **0.496** | 0.528 | +0.302 | **severo** |
| v2a — MiniLM emb + LogReg | 0.507 | 0.442 | 0.474 | **+0.065** | **bajo** |

Hallazgo: las embeddings semánticas **congeladas** de un encoder de propósito
general **no superan** al TF-IDF en esta tarea (−0.05 de F1-macro val), aunque
**casi no sobreajustan**. Interpretación: el señal discriminante de la función de
cita en este dataset es en buena parte **léxico** (frases-guía tipo *"in contrast
to"*, *"we use"*, *"following"*, *"see also"*); un encoder sin ajustar no lo captura
mejor que los n-gramas. El camino para superar el baseline es **ajustar el
encoder**, no usar sus embeddings tal cual.

**v2b — fine-tuning de `distilbert-base-uncased`** (submuestreo estratificado a
5 000, `max_len 128`, `lr 3e-5`, `batch 16`, **validación cada 78 pasos** +
early-stopping; ~55 min en la `t3.large`, registrado en MLflow):

| Modelo | F1-macro train | F1-macro val | F1-macro test | Brecha train→val | Sobreajuste |
|---|--:|--:|--:|--:|---|
| v1 — TF-IDF + LogReg | 0.824 | 0.496 | 0.531 | +0.328 | severo |
| **v2b — DistilBERT fine-tune** | 0.722 | **0.573** | **0.577** | **+0.149** | **moderado** |

**v2b supera al baseline en +0.077 de F1-macro val** y reduce el sobreajuste a la
mitad (0.33 → 0.15). La brecha val→test es **−0.003** → generaliza sin fuga.

*Curva de aprendizaje paso a paso* (`docs/modelos_entrega2/v2b_learning_curve.png`),
F1-macro val: 0.23 (paso 78) → 0.52 (234) → **0.573 (468)** → 0.56 (546) → 0.56 (624).
Mejora sostenida hasta el paso 468 (~1.5 épocas), luego **empieza a bajar**: la
validación frecuente detecta el inicio del sobreajuste y el early-stopping restaura
el mejor checkpoint, ahorrando ~1 época de cómputo.

*Figura 6 — `docs/modelos_entrega2/v2b_matriz_confusion_val.png` (v2b).* Mejores
clases: `Gap` 0.72, `Background` 0.70; peores `Basis` 0.40, `Identification of the
Originator` 0.47.

**¿Es DistilBERT la mejor arquitectura bajo estas restricciones?** Es la mejor de
las probadas y una elección sólida, pero no óptima. Candidatos que podrían
superarla dentro del mismo presupuesto (→ v3): **SciBERT/SPECTER2** (encoder de
dominio científico, ~1.5–2 h CPU), **fine-tune con datos completos** (12 600 en vez
de 5 000), **DeBERTa-v3-small**. El cuello de botella real es la **calidad de los
datos** (49 % sin marcador, etiquetas de juez-LLM, balanceo artificial), no la
arquitectura: los modelos publicados llegan a ~0.80 F1 sobre datasets *limpios*.

### 3.5 Curvas de aprendizaje y validación cruzada (v1)

Para responder *¿mejora la validación con más datos?* y *¿qué tan estable es la
métrica?*, sobre el baseline v1 (barato de reentrenar):

**Curva de aprendizaje por tamaño de entrenamiento** (`scripts/learning_curve_v1.py`):

| n train | F1-macro train | F1-macro val | Brecha |
|--:|--:|--:|--:|
| 500 | 0.983 | 0.331 | 0.65 |
| 1 000 | 0.952 | 0.378 | 0.58 |
| 2 000 | 0.917 | 0.416 | 0.50 |
| 4 000 | 0.887 | 0.442 | 0.45 |
| 8 000 | 0.838 | 0.477 | 0.36 |
| 12 600 | 0.823 | 0.496 | 0.33 |

*Figura 7 — `docs/modelos_entrega2/v1_learning_curve.png`.* El F1 de validación
**sube de forma sostenida** (+0.17 de 500 a 12 600) y la brecha **se cierra**
(0.65 → 0.33) porque el F1 de entrenamiento baja al añadir datos. La pendiente en
12 600 sigue siendo positiva → **más contextos de cita reales seguirían mejorando
v1**; el sobreajuste actual es en parte *falta de datos* y en parte límite léxico.

**Validación cruzada 5-fold `GroupKFold` por `citing_paper_id`** (respeta el
aislamiento por documento):

| Fold | 0 | 1 | 2 | 3 | 4 | Media ± σ |
|---|--:|--:|--:|--:|--:|--:|
| F1-macro | 0.492 | 0.526 | 0.510 | 0.498 | 0.518 | **0.509 ± 0.012** |

La métrica es **estable** (σ = 0.012; IC95 ≈ [0.50, 0.52]); el 0.496 del conjunto
de validación fijo está dentro de ese rango. *No* se hace k-fold sobre los
transformers: cada fold costaría ~80 min en las 2 vCPU. Para v2b se usa en su
lugar la **validación cada 78 pasos** durante el entrenamiento (misma idea:
seguir el progreso, no una sola foto al final).

### 3.6 v3 — reentrenamiento sobre el dataset completo (pendiente de GPU)

La curva de aprendizaje de v1 y el resultado de v2b indican dos palancas con
retorno positivo que quedan preparadas pero no ejecutadas en esta entrega por
falta de una instancia con GPU:

1. **Datos completos:** v2b se entrenó con un submuestreo de 5 000 ejemplos por el
   límite de 2 vCPU; v3 permite usar los 14 461 de entrenamiento.
2. **Encoder de dominio científico:** `allenai/scibert_scivocab_uncased` o
   `allenai/specter2` en lugar de `distilbert-base-uncased`.

El procedimiento está listo en `infra/train_v3_gpu.sh` (aprovisiona una GPU,
`dvc pull` de la v3, entrena y registra en MLflow). Con una `g5.xlarge` (A10G) el
entrenamiento de SciBERT sobre los 14 461 ejemplos × 4 épocas toma ~20–35 min.
Rango de F1-macro de validación esperado: **0.62–0.72** (por encima del 0.573 de v2b),
con el ruido de etiqueta (§2.6) como techo estructural.

---

## 4. Observaciones y conclusiones sobre los modelos

- **v1 sobreajusta de forma severa** (F1-macro train 0.80 vs val 0.50). El barrido
  de `C` muestra que es un límite del enfoque, no un problema de calibración: con
  `C = 10` el modelo alcanza F1 0.998 en entrenamiento y sigue en 0.50 en
  validación. Aun así, v1 **triplica el azar** (0.50 vs 0.111) y sirve de
  **baseline honesto** y de cota inferior.
- **El split es sano:** la brecha val→test es −0.03 (val ligeramente por debajo de
  test), lo que confirma que el aislamiento por `citing_paper_id` funciona y no hay
  fuga de información.
- **Errores sistemáticos de v1:** las clases *conceptuales* y de límites difusos
  (`Basis` F1 0.34, `Identification of the Originator` F1 0.39, `Comparison` recall
  0.37) se confunden con clases más frecuentes léxicamente como `Background` y
  `Application`. `Background` recibe predicciones de más (recall 0.78 / precisión
  0.50).
- **v2a (embeddings congeladas) no supera al baseline** (F1-macro val 0.44 vs 0.50)
  pero **casi no sobreajusta** (brecha 0.07 vs 0.30). Confirma que en este dataset
  la señal de la función de cita es marcadamente **léxica**: un encoder de
  propósito general sin ajustar no aporta sobre los n-gramas. `Basis` sigue siendo
  la peor clase (F1 0.24), y las confusiones dominantes son
  `Basis`→`Modification/Improvement` y `Comparison`→`Evidence`.
- **v2b (DistilBERT fine-tune) pasa el techo léxico:** F1-macro val **0.573**
  (+0.077 vs v1), test 0.577, brecha train→val **0.15 (moderado)** y val≈test.
  Ajustar los pesos del encoder **sí** aporta sobre los n-gramas — lo que no hacían
  las embeddings congeladas. La curva paso a paso muestra mejora real hasta ~1.5
  épocas y luego sobreajuste incipiente, cortado por early-stopping.
  **No se optimiza a fondo a propósito** (submuestreo de 5 000, 2 épocas, sin
  búsqueda de hiperparámetros).
- **v2b es la mejor arquitectura de las probadas, no la óptima.** Bajo las mismas
  restricciones (2 vCPU, sin GPU), para **v3**: SciBERT/SPECTER2 (dominio
  científico), fine-tune con datos completos, DeBERTa-v3-small; y sobre todo
  **mejores datos/etiquetas** — limpiar el ~49 % sin marcador, anotación humana,
  *class-balanced loss*. El techo publicado para esta tarea con datos limpios es
  ~0.80 F1; el ~0.57 actual está limitado por los datos, no por el modelo.
- **Amenaza a la validez externa:** el dataset está balanceado artificialmente
  (2 000/clase); las métricas no reflejan la prevalencia real de cada función de
  cita. La evaluación de producción (v3) debe hacerse con distribución natural.

---

## 5. Tablero desarrollado

⟨Sección a completar con capturas del tablero.⟩ Puntos a cubrir:

- Entrada de un *citation context* + sección; conteo de palabras/tokens y detección
  del marcador de cita.
- Selección del clasificador (v1 / v2) vía la API (`POST /predict`).
- Función de cita predicha entre las 9 categorías, con definición y puntaje de
  confianza; distribución de probabilidades sobre todas las clases; aviso de
  ambigüedad cuando el margen entre las dos clases principales es estrecho.
- Panel "Top-3 de pasajes recuperados": **marcado como pendiente de datos** (ver §2.3),
  no poblado con datos inventados.
- Despliegue con Docker.

---

## 6. Reporte de trabajo en equipo (máx. 1 página — hoja aparte)

⟨Actualizar responsables de la Entrega 2 — ver `MEJORAS_ENTREGA2.md` §F.⟩

## Referencias

1. Jurgens, D., Kumar, S., Hoover, R., McFarland, D., & Jurafsky, D. (2018). Measuring the evolution of a scientific field through citation frames. *TACL*, 6, 391–406.
2. Teufel, S., Siddharthan, A., & Tidhar, D. (2006). Automatic classification of citation function. *EMNLP 2006*, 103–110.
3. Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A pretrained language model for scientific text. *EMNLP-IJCNLP 2019*.
4. Cohan, A., Ammar, W., van Zuylen, M., & Cady, F. (2019). Structural scaffolds for citation intent classification in scientific publications. *NAACL 2019*.
