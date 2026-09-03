# EDA — Entrega 2 (SCIF)

> Reproducible con `python scripts/eda_entrega2.py --data_dir data/raw --out_dir docs/eda_entrega2`.
> Figuras en `docs/eda_entrega2/*.png`; métricas crudas en `docs/eda_entrega2/eda_stats.json`.

Este EDA se hace sobre el **dataset enriquecido de intención de cita** (nuevo en
la Entrega 2), que reemplaza al corpus de 24 855 abstracts de la Entrega 1. Está
escrito para responder de forma explícita el feedback recibido.

| Feedback Entrega 1 | Dónde se responde |
|---|---|
| «no se explora la variable objetivo (base-label) … ni su balance entre las nueve categorías» | §2 |
| «la propia exploración demuestra que el archivo no contiene contextos de cita reales» | §3 |
| «la maqueta asume datos que no se mostraron consistentes en la exploración de variables» | §4 |
| «aterrizar la utilidad … no se define el alcance» | §6 + `MEJORAS_ENTREGA2.md` |

---

## 1. Estructura e integridad

| Partición | Filas | `citing_paper_id` únicos | Nulos `rhetorical_section` | Duplicados de fila |
|---|--:|--:|--:|--:|
| train | 12 600 | 12 600 | 186 (1.48 %) | 0 |
| val   | 2 700  | 2 700  | 37 (1.37 %)  | 0 |
| test (para anotación humana) | 2 700 | 2 700 | 42 (1.56 %) | 0 |
| master 18k balanceado | 18 000 | 18 000 | 265 | 0 |

Columnas del CSV enriquecido: `citation_context`, `rhetorical_section`,
`etiqueta_rescatada`, `label`, `citing_paper_id`, `cited_paper_id`, `pair_id`,
`top1_cited_chunk`, `top2_cited_chunk`, `top3_cited_chunk`.

- `label` es **idéntica** a `etiqueta_rescatada` en el 100 % de las filas (columna redundante).
- `train + val + test = 18 000` filas → las tres particiones son un **split del master 18k**.
- 1 registro por `citing_paper_id`; `pair_id = citing_<n>__cited_<n>`.

---

## 2. Variable objetivo — `label` (9 funciones de cita)

Es la variable que la Entrega 1 llamó `base_label` y que **no se había explorado**.
Definición operativa: función retórica de la cita, 9 categorías mutuamente
excluyentes (`label_id` 0–8 en orden alfabético, ver `data/processed/dataset_manifest.json`).

### 2.1 Balance entre las nueve categorías

**train (n = 12 600)**

| Función de cita | n | % |
|---|--:|--:|
| Background | 1 432 | 11.4 % |
| Application | 1 425 | 11.3 % |
| Basis | 1 412 | 11.2 % |
| Further_Reading | 1 411 | 11.2 % |
| Evidence | 1 393 | 11.1 % |
| Comparison | 1 392 | 11.0 % |
| Identification_of_the_Originator | 1 387 | 11.0 % |
| Modification_Improvement | 1 377 | 10.9 % |
| Gap | 1 371 | 10.9 % |

- **Ratio de desbalance** (clase mayor / clase menor): **1.04** en train, 1.16 en val, 1.17 en test.
- **Entropía normalizada de la distribución: 1.00** (train) → prácticamente uniforme.
- El *master 18k* está balanceado **por construcción** a 2 000 ejemplos/clase.

> **Lectura para el negocio.** El desbalance «fuerte» que la literatura reporta
> para esta tarea (Jurgens et al., 2018) **no aplica aquí porque el balanceo es
> artificial**: el dataset se construyó forzando ~2 000 ejemplos por clase. Esto
> es cómodo para entrenar y comparar modelos, pero implica que las métricas de
> validación **no reflejan la prevalencia real** de cada función de cita en un
> corpus natural. En producción (v3) habrá que re-evaluar con una distribución
> realista y, muy probablemente, reponderar.

Figura: `docs/eda_entrega2/01_variable_objetivo.png` (distribución por partición).

### 2.2 Consistencia de la etiqueta entre particiones

Cada clase mantiene ~11 % en train / val / test → el split preserva la
estratificación. No hay clases ausentes en ninguna partición.

---

## 3. `citation_context` — ¿ahora sí son citas reales?

**Sí.** A diferencia de la Entrega 1 (donde el texto eran *abstracts* completos),
ahora cada registro es **una o dos oraciones alrededor de una mención bibliográfica**.

| Métrica (palabras) | train | val | test | Entrega 1 (abstracts) |
|---|--:|--:|--:|--:|
| media | 28.4 | 29.0 | 28.1 | 165.2 |
| mediana | 26 | 26 | 26 | 163 |
| p05 / p95 | 10 / 50 | 10 / 51 | 10 / 49 | — |
| máx | 246 | 407 | 217 | 482 |
| % con marcador de cita explícito (`[12]`, `(Autor, 2020)`, `Autor et al. (2020)`) | **50.6 %** | 52.6 % | 52.2 % | ~0 % |
| % muy corto (≤ 8 palabras) | 2.7 % | 3.1 % | 2.9 % | — |

- Longitud coherente con un *citation context* (mediana 26 palabras) y **6× más
  corta** que en la Entrega 1: prueba cuantitativa de que el problema señalado en
  el feedback está corregido.
- **~50 % de los contextos no tienen un marcador de cita detectable por regex.**
  Parte es ruido real (frases de enunciados de ejercicios, encabezados), parte
  son citas narrativas sin paréntesis. Es un **límite de calidad** a vigilar: se
  reporta y se deja la limpieza fina como tarea de datos.
- La longitud varía por clase (`03_longitud_por_clase.png`): `Background` es la
  más corta (≈ 19 palabras de media) y `Comparison` la más larga (≈ 32); hay,
  por tanto, una señal de longitud que un modelo lineal puede explotar.

Figuras: `02_longitud_contexto.png`, `03_longitud_por_clase.png`.

---

## 4. `cited_paper_id` y `top{1,2,3}_cited_chunk` — inconsistencia que afecta a la maqueta

El feedback dice: *«la maqueta asume datos que no se mostraron consistentes en la
exploración de variables»*. Confirmado y cuantificado:

| Comprobación | Resultado |
|---|---|
| `citing_paper_id` == `cited_paper_id` (sin el prefijo) | **100 %** de las filas |
| `top1_cited_chunk.text` == `citation_context` | **94.8 %** de las filas |
| Valores de similitud distintos en `top1` / `top2` / `top3` | `{0.85, 0.92}` / `{0.73, 0.74}` / `{0.61}` |
| Textos distintos en `top2_cited_chunk` (de 12 600 filas) | 1 162, dominados por la plantilla *«Foundational background and methodology related to \<sección\>»* |
| Textos distintos en `top3_cited_chunk` | 1 121, plantilla *«Experimental setup, results and empirical evidence for \<sección\>»* |

**Diagnóstico.** No existe un documento citado real distinto del citante:
`cited_paper_id` es un espejo de `citing_paper_id`. El «Top-3 de fragmentos
recuperados» del CSV es **sintético** (top-1 = copia del contexto; top-2/top-3 =
plantillas con similitud constante). Es exactamente el mismo hallazgo de la
Entrega 1, ahora con evidencia numérica.

**Consecuencia sobre el alcance de la Entrega 2:**

1. Se modela **solo la clasificación de la función de cita** (9 clases) a partir
   de `citation_context` + sección. Es una tarea bien definida y con datos
   consistentes.
2. El componente de **recomendación local de citas (Top-3 chunks)** se declara
   **fuera del alcance de esta iteración** y pasa a *trabajo futuro (v3)*: requiere
   descargar el PDF del artículo citado real, segmentarlo en *chunks* ≤ 300
   palabras y calcular similitud con SciBERT. La maqueta se ajusta para marcar ese
   panel como *ilustrativo / pendiente de datos* (ver `MEJORAS_ENTREGA2.md`).
3. `scripts/prepare_dataset.py` **no propaga** las columnas `top*_cited_chunk` al
   dataset estandarizado, para evitar que un modelo aprenda del artefacto.

---

## 5. Fuga de información entre particiones (No-Leakage Guarantee)

| Solape de `citing_paper_id` | conteo |
|---|--:|
| train ∩ val | **0** |
| train ∩ test | **0** |
| val ∩ test | **0** |
| Contextos de texto idénticos entre particiones | 11 (0.06 %) — mismo texto, distinto `citing_paper_id`; no rompe el aislamiento por documento |

La garantía de no-leakage por documento **se cumple**. Los 11 textos repetidos se
documentan como frases genéricas cortas ("We use the same setup as …").

---

## 6. Diversidad temática del corpus (alcance)

Top términos de contenido en `citation_context` (train): *model, learning, data,
method, network, training, based, using, approach, performance, results, dataset,
tasks, image, language, framework, algorithm, features, proposed, feature…*

El núcleo es ML/NLP/visión, **pero hay contextos claramente fuera de Ciencias de
la Computación** (radiación en trabajadores de granito, células endoteliales de
cerebro de ratón, síndrome de Thurston, composición porcentual de elementos
químicos). Es decir: el corpus **no está restringido a `cs.AI` / `cs.LG` /
`cs.CL`** como afirmaba el alcance de la Entrega 1.

→ En el reporte se ajusta el alcance: **«contextos de cita en inglés de literatura
científica multidominio, con predominio de Ciencias de la Computación»**, y se
añade el dominio como variable a monitorear en v3.

---

## 7. Implicaciones para el modelado (Entrega 2)

| Hallazgo | Decisión de modelado |
|---|---|
| Variable objetivo balanceada (9 clases, ratio 1.04) | Métrica principal **F1-macro**; `accuracy` es interpretable (baseline aleatorio = 0.111) |
| Texto corto (mediana 26 palabras) | `max_length` 256 tokens sobra; el costo de fine-tuning es bajo |
| Señal de sección retórica (`04_seccion_x_label.png`) | Se concatena `[SEC] <sección canónica>` al texto de entrada |
| `rhetorical_section` con 558 valores crudos y 1.5 % nulos | Normalización a 10 secciones canónicas en `prepare_dataset.py` |
| Top-3 chunks sintéticos | Excluidos del entrenamiento; retrieval → v3 |
| Balanceo artificial | Se reporta como amenaza a la validez externa; no se optimiza *accuracy* de producción todavía |

Modelos entrenados (ver `docs/MODELOS_ENTREGA2.md`):

- **v1 baseline** — TF-IDF (1–2 gramas) + Regresión Logística.
- **v2 optimizado (iteración intermedia)** — fine-tuning de SciBERT; deja margen deliberado para v3.
