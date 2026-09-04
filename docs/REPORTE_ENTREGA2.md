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
| Datos | 24 855 registros que resultaron ser **abstracts**, no contextos de cita (`base_label` constante = 0) | **Dataset enriquecido de intención de cita**: 18 000 registros, contextos de cita **reales**, 9 clases balanceadas |
| Variable objetivo | No se exploró | **Explorada** (distribución, balance entre las 9 categorías, implicaciones) — §2.1 |
| Alcance | Argumentado en términos académicos generales | Definido por usuario, entrada/salida y no-alcance — §1.3 |
| Maqueta | Asumía un panel Top-3 de *chunks* con datos que no se mostraron consistentes | Panel Top-3 marcado como **ilustrativo / pendiente de datos**; la maqueta operativa es el clasificador — §4 |
| Modelos | — | v1 (baseline) y v2 (iteración intermedia), versionados con MLflow — §3 |

---

## 2. Datos y exploración

### 2.1 Variable objetivo — `label` (9 funciones de cita)

Función retórica de la cita, 9 categorías mutuamente excluyentes codificadas
`label_id` 0–8: **Application, Background, Basis, Comparison, Evidence,
Further Reading, Gap, Identification of the Originator, Modification/Improvement**.

**Balance entre las nueve categorías (entrenamiento, n = 12 600):**

| Función de cita | n | % |
|---|--:|--:|
| Background | 1 432 | 11.4 % |
| Application | 1 425 | 11.3 % |
| Basis | 1 412 | 11.2 % |
| Further Reading | 1 411 | 11.2 % |
| Evidence | 1 393 | 11.1 % |
| Comparison | 1 392 | 11.0 % |
| Identification of the Originator | 1 387 | 11.0 % |
| Modification / Improvement | 1 377 | 10.9 % |
| Gap | 1 371 | 10.9 % |

- **Ratio de desbalance** (clase mayor / menor) = **1.04** en train (1.16–1.17 en
  val/test); **entropía normalizada = 1.00** → distribución prácticamente uniforme.
- El maestro de 18 000 registros está balanceado **por construcción** (2 000/clase)
  y se particiona en train 12 600 / val 2 700 / test 2 700, manteniendo ~11 % por
  clase en cada partición (sin clases ausentes).

*Figura 1 — `docs/eda_entrega2/01_variable_objetivo.png` (distribución de `label` por partición).*

**Implicación para el negocio.** El balance es **artificial**: no refleja la
prevalencia real de cada función de cita en un corpus natural. Es adecuado para
entrenar y comparar modelos con F1-macro en esta fase, pero la evaluación de
producción (v3) deberá hacerse con distribución realista y, probablemente,
reponderando la pérdida.

### 2.2 `citation_context` — ahora son citas reales

| Métrica (palabras) | Entrega 2 (train) | Entrega 1 (abstracts) |
|---|--:|--:|
| Mediana | **26** | 163 |
| Media | 28.4 | 165.2 |
| p05 / p95 | 10 / 50 | — |
| % con marcador de cita explícito (`[12]`, `(Autor, 2020)`) | **~51 %** | ~0 % |

La longitud (mediana 26 palabras, 6× menor que en la Entrega 1) y la presencia de
marcadores confirman que ahora el texto **sí** es un contexto de cita. La longitud
varía por clase (`Background` la más corta, `Comparison` la más larga → hay señal
de longitud). ~49 % de contextos no tienen marcador detectable por regex: es ruido
de calidad de datos que se reporta y se limpiará en v3.

*Figuras 2–3 — `02_longitud_contexto.png`, `03_longitud_por_clase.png`.*

### 2.3 Consistencia de los datos y ajuste de alcance

La exploración confirma que **no existe un artículo citado real** distinto del
citante:

| Comprobación (train) | Resultado |
|---|---|
| `citing_paper_id` == `cited_paper_id` | 100 % |
| `top1_cited_chunk` == `citation_context` | 94.8 % |
| Similitud en top1 / top2 / top3 | valores **constantes** (0.92 / 0.74 / 0.61) |
| top2 / top3 | plantillas de texto genéricas |

Por eso, en la Entrega 2 el componente de **recomendación de fragmentos (Top-3
chunks)** queda fuera de alcance y el panel correspondiente de la maqueta se marca
como *ilustrativo / pendiente de datos*. `scripts/prepare_dataset.py` **no propaga**
esas columnas al dataset estandarizado para evitar que un modelo aprenda del
artefacto.

### 2.4 Aislamiento entre particiones (No-Leakage)

| Solape de `citing_paper_id` | train∩val | train∩test | val∩test |
|---|--:|--:|--:|
| conteo | **0** | **0** | **0** |

La garantía de no-leakage por documento se cumple. El detalle completo del EDA está
en `docs/EDA_ENTREGA2.md` y es reproducible con `python scripts/eda_entrega2.py`.

### 2.5 Estructura del dataset

| Partición | Registros | Uso |
|---|--:|---|
| Entrenamiento (`train`) | 12 600 | ajuste de los modelos |
| Validación (`val`) | 2 700 | selección de modelo / hiperparámetros |
| Prueba (`test`, para anotación humana) | 2 700 | evaluación final + acuerdo interanotador |
| Maestro balanceado | 18 000 | fuente de las 3 particiones (2 000/clase) |

Variables usadas por el modelo: `citation_context` (texto) y
`rhetorical_section_canon` (10 secciones canónicas, normalizadas desde 558 valores
crudos; 1.5 % nulos). El archivo maestro y las particiones se versionan con **DVC**
sobre `s3://citation-dvcstore-tema1`.

---

## 3. Modelos desarrollados y su evaluación

Se entrenan dos iteraciones para la tarea de **clasificación de la función de
cita** (9 clases). Los experimentos se registran en **MLflow** sobre una instancia
**AWS EC2** (experimento `scif-citation-intent`); pantallazos en `docs/soportes/`.
El baseline (v1) es además reproducible sin servidor con
`python scripts/eval_baseline.py` (artefactos en `docs/modelos_entrega2/`).

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

**v2b — fine-tuning de `distilbert-base-uncased`** (`--max_train 5000`, 1–2 épocas,
`max_len 128`; ~1.5–3 h en 2 vCPU, registrado en MLflow sobre la EC2):

| Modelo | F1-macro train | F1-macro val | F1-macro test | Brecha train→val | Sobreajuste |
|---|--:|--:|--:|--:|---|
| v2b — DistilBERT fine-tune | ⟨pendiente EC2⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

*(v2b se completa con el run en EC2; `scripts/analyze_metrics.py` llena la tabla.)*

*Figura 6 — `docs/modelos_entrega2/v2_matriz_confusion_val.png` (v2a).*

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
- **v2b (DistilBERT fine-tune) — hipótesis:** al ajustar los pesos del encoder se
  espera **superar** el 0.50 del baseline (rango esperado 0.55–0.68) con brecha
  train→val moderada. Es la primera iteración que puede pasar el techo léxico.
  **No se optimiza a fondo a propósito** (submuestreo de 5 000, 1–2 épocas, sin
  búsqueda de hiperparámetros): el margen restante es el objetivo de **v3**
  (fine-tuning completo con GPU, SciBERT/SPECTER, *class-balanced loss*, más
  contextos reales, limpiar el ~49 % de contextos sin marcador detectable).
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
