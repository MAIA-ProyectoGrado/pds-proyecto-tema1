# EDA — Entrega 2 (SCIF)

> Reproducible con `python scripts/eda_entrega2.py --data_dir dataset --out_dir docs/eda_entrega2`.
> Figuras en `docs/eda_entrega2/*.png`; métricas crudas en `docs/eda_entrega2/eda_stats.json`.
> Dataset **v3** (20 655 registros), versionado con DVC (`dataset.dvc`).

Este EDA se hace sobre el **dataset enriquecido de intención de cita**, que
reemplaza al corpus de 24 855 *abstracts* de la Entrega 1. Está escrito para
responder de forma explícita el feedback recibido.

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
| train | 14 461 | 14 434 | 195 (1.35 %) | 0 |
| val   | 3 098  | 3 093  | 56 (1.81 %)  | 0 |
| test (para anotación humana) | 3 096 | 3 093 | 47 (1.52 %) | 0 |
| master balanceado (`dataset_20655_balanceado_enriched.csv`) | 20 655 | — | 298 | 0 |

Columnas del CSV: `citation_context`, `rhetorical_section`, `etiqueta_rescatada`,
`label`, `citing_paper_id`, `cited_paper_id`, `pair_id`,
`top1/2/3_cited_chunk`, y **nuevas en v3**: `source_dataset`, `predicted_label`,
`scores_json` (presentes solo en 173 filas de procedencia *Semantic Scholar Async*).

- `label` es **idéntica** a `etiqueta_rescatada` en el 100 % de las filas.
- `train + val + test = 20 655` filas → las tres particiones son un **split del
  master balanceado** (2 295 ejemplos por clase antes de particionar).
- `pair_id = citing_<id>__cited_<id>`. En ~98 % `<id>` es común (auto-referencia,
  ver §4).

---

## 2. Variable objetivo — `label` (9 funciones de cita)

Es la variable que la Entrega 1 llamó `base_label` y que **no se había explorado**.
Definición operativa: función retórica de la cita, 9 categorías mutuamente
excluyentes (`label_id` 0–8 en orden alfabético, ver `data/processed/dataset_manifest.json`).

### 2.1 Balance entre las nueve categorías

**train (n = 14 461)**

| Función de cita | n | % |
|---|--:|--:|
| Gap | 1 637 | 11.3 % |
| Evidence | 1 632 | 11.3 % |
| Further_Reading | 1 617 | 11.2 % |
| Application | 1 613 | 11.2 % |
| Modification_Improvement | 1 607 | 11.1 % |
| Background | 1 597 | 11.0 % |
| Identification_of_the_Originator | 1 597 | 11.0 % |
| Basis | 1 595 | 11.0 % |
| Comparison | 1 566 | 10.8 % |

- **Ratio de desbalance** (clase mayor / clase menor): **1.045** en train,
  1.11 en val, 1.17 en test. **Entropía normalizada de la distribución: 1.00**
  (train) → prácticamente uniforme.
- El *master* está balanceado **por construcción** a 2 295 ejemplos/clase.

> **Lectura para el negocio.** El desbalance «fuerte» que la literatura reporta
> para esta tarea (Jurgens et al., 2018) **no aplica aquí porque el balanceo es
> artificial**: el dataset se construyó forzando ~2 295 ejemplos por clase, en
> parte con un **mecanismo de rescate** que reasigna citas a clases minoritarias
> (ver §7). Esto es cómodo para entrenar y comparar modelos, pero implica que las
> métricas de validación **no reflejan la prevalencia real** de cada función de
> cita ni la calidad de la etiqueta. En producción habrá que re-evaluar con
> distribución realista y anotación humana.

Figura: `docs/eda_entrega2/01_variable_objetivo.png`.

### 2.2 Consistencia de la etiqueta entre particiones

Cada clase mantiene ~11 % en train / val / test → el split preserva la
estratificación. No hay clases ausentes en ninguna partición.

---

## 3. `citation_context` — ¿son citas reales?

**Sí.** A diferencia de la Entrega 1 (donde el texto eran *abstracts* completos),
ahora cada registro es **una o dos oraciones alrededor de una mención bibliográfica**.

| Métrica (palabras) | train | val | test | Entrega 1 (abstracts) |
|---|--:|--:|--:|--:|
| media | 28.3 | 29.0 | 28.3 | 165.2 |
| mediana | 26 | 26 | 26 | 163 |
| p05 / p95 | 10 / 50 | 10 / 51 | 10 / 50 | — |
| máx | 407 | 375 | 228 | 482 |
| % con marcador de cita explícito (`[12]`, `(Autor, 2020)`, `Autor et al. (2020)`) | **51.3 %** | 51.3 % | 51.3 % | ~0 % |
| % muy corto (≤ 8 palabras) | 2.9 % | 2.6 % | 2.7 % | — |

- Longitud coherente con un *citation context* (mediana 26 palabras), **6× más
  corta** que en la Entrega 1: el problema señalado en el feedback está corregido.
- **48.7 % de los contextos no tienen un marcador de cita detectable por regex.**
  Parte es ruido real (encabezados, entradas de bibliografía, frases genéricas sin
  cita), parte son citas narrativas sin paréntesis. Es un **límite de calidad**
  cuantificado que acota el techo de desempeño (§7).
- La longitud varía por clase (`03_longitud_por_clase.png`): `Background` la más
  corta, `Comparison` la más larga → hay señal de longitud.

Figuras: `02_longitud_contexto.png`, `03_longitud_por_clase.png`.

---

## 4. `cited_paper_id` y `top{1,2,3}_cited_chunk` — consistencia (feedback maqueta)

El feedback dice: *«la maqueta asume datos que no se mostraron consistentes»*.
Estado en v3:

| Comprobación (train) | Resultado v3 | (Entrega 2, v2) |
|---|---|---|
| `citing_paper_id` == `cited_paper_id` (sin prefijo) | **98.3 %** | 100 % |
| `top1_cited_chunk.text` == `citation_context` | **80.3 %** | 94.8 % |
| Textos distintos en `top2_cited_chunk` (de 14 461) | **14 233** | 1 162 (plantillas) |
| Textos distintos en `top3_cited_chunk` | 14 392 | 1 121 (plantillas) |
| Similitud `top1` / `top2` / `top3` | varía (~0.72 / ~0.55 / ~0.48) | constante (0.92 / 0.74 / 0.61) |

**Diagnóstico.** v3 mejora respecto a v2: `top2`/`top3` ya **no son plantillas**
(varían por fila y las similitudes no son constantes), lo que indica un *chunking*
real — aunque **ruidoso**: el fragmento más común de `top2`/`top3` es `"2021)."`,
señal de fragmentos truncados. Y sigue siendo cierto que en el **98.3 %** de las
filas el documento «citado» es el mismo que el citante (auto-referencia
`citing_legacy_N == cited_legacy_N`); solo el **1.7 %** tiene un par citante/citado
distinto y con identificador real (columna `par_citado_real`).

**Consecuencia sobre el alcance de la Entrega 2:**

1. Se modela **solo la clasificación de la función de cita** (9 clases) a partir
   de `citation_context` + sección retórica. Tarea bien definida y consistente.
2. El componente de **recomendación local de citas (Top-3 chunks)** se declara
   **fuera del alcance de esta iteración**: aunque v3 ya trae *chunks* reales,
   provienen mayoritariamente del propio artículo citante y con ruido de
   segmentación. La maqueta marca ese panel como *ilustrativo / pendiente de datos*.
3. `scripts/prepare_dataset.py` **no propaga** las columnas `top*_cited_chunk` al
   dataset de entrenamiento; sí propaga `par_citado_real` para análisis.

---

## 5. Fuga de información entre particiones (No-Leakage)

| Solape de `citing_paper_id` | conteo |
|---|--:|
| train ∩ val | **1** (0.03 % de val) |
| train ∩ test | **0** |
| val ∩ test | **0** |
| Contextos de texto idénticos entre particiones | 19 (0.09 %) — mismo texto, distinto `citing_paper_id` |

La garantía de no-leakage por documento **se mantiene en la práctica**: 1
`citing_paper_id` compartido entre train y val (frente a 0 en la Entrega 2) es
despreciable pero se documenta. Los 19 textos repetidos son frases genéricas
cortas que no rompen el aislamiento por documento.

---

## 6. Diversidad temática del corpus (alcance)

Top términos de contenido en `citation_context` (train): *based, learning, models,
model, data, methods, training, policy, results, studies, performance, recent,
approach, task, propose, method, framework, network, image, language…*

El núcleo es ML / NLP / RL, **pero hay contextos claramente fuera de Ciencias de
la Computación** (Cox proportional hazards, β-catenin y E-cadherina, dispersión de
semillas, ternary plots de cationes y aniones). El corpus **no está restringido a
`cs.AI` / `cs.LG` / `cs.CL`** como afirmaba la Entrega 1 → alcance ajustado a
**«literatura científica multidominio, con predominio de Ciencias de la Computación»**.

---

## 7. Calidad del etiquetado (asistido por LLM + rescate)

El etiquetado es **asistido por LLM** (`gemini-…-flash-lite`, esquema estructurado)
con un **rescate probabilístico** que reasigna citas a clases minoritarias. No hay
anotación humana completa (el 15 % «test humano» está reservado para ese fin).

| Señal | Valor |
|---|--:|
| Filas con `predicted_label` (juez LLM) disponible | 173 |
| Acuerdo `predicted_label` == `label` final | **71.7 %** |
| → citas **reasignadas por el rescate** en ese subconjunto | **28.3 %** |
| Contextos sin marcador de cita detectable (train) | **48.7 %** |

**Implicación.** El rescate cambia casi 1 de cada 3 etiquetas donde se puede medir,
y casi la mitad de los contextos no contienen una cita detectable. Una inspección
manual de ejemplos por clase confirma etiquetas discutibles (p. ej. *"Training uses
Adam with learning rate 1e-3, batch size 256…"* etiquetado como
`Identification_of_the_Originator`). Este **ruido de etiqueta** —no la
arquitectura— es el principal límite del desempeño alcanzable (ver
`docs/MODELOS_ENTREGA2.md`, §5).

---

## 8. Implicaciones para el modelado

| Hallazgo | Decisión de modelado |
|---|---|
| Variable objetivo balanceada (9 clases, ratio 1.045) | Métrica principal **F1-macro** (aleatorio = 0.111); accuracy interpretable |
| Texto corto (mediana 26 palabras) | `max_length` ≤ 256 sobra; fine-tuning barato |
| Señal de sección retórica (`04_seccion_x_label.png`) | Se concatena `[SEC] <sección canónica>` a la entrada |
| `rhetorical_section` con 622 valores crudos y 1.35 % nulos | Normalización a 10 secciones canónicas en `prepare_dataset.py` |
| Top-3 chunks reales pero ruidosos, casi siempre del citante | Excluidos del entrenamiento; retrieval → trabajo futuro |
| Balanceo artificial + rescate + 48.7 % sin marcador | Amenaza a la validez externa y techo de desempeño; se documenta |
| Dataset v3 más grande (14 461 train vs 12 600) | La curva de aprendizaje de v1 predice ganancia → v3 reentrena con datos completos |

Modelos (ver `docs/MODELOS_ENTREGA2.md`): **v1** TF-IDF + LogReg · **v2a**
embeddings MiniLM congeladas + LogReg · **v2b** DistilBERT fine-tuned · **v3**
(planificado) encoder de dominio científico con datos completos.
