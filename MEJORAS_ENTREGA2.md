# Mejoras para la Entrega 2 — respuesta al feedback de la Entrega 1

**Estado Entrega 1:** 85/100 (14+15+8+20+18+10).
Este documento indica, para cada comentario del evaluador, **dónde** está el texto
en `Entrega 1 Proyectos.pdf` (página y párrafo), **qué** se cambia y **con qué
evidencia**. El texto propuesto está listo para pegar. Los `⟨…⟩` se completan con
los números de los runs de MLflow (`scripts/analyze_metrics.py` los deja casi todos).

> Reglas del enunciado (semana 4): el reporte de la Entrega 2 es **nuevo y de máx.
> 10 páginas**, debe **"Resaltar posibles cambios con respecto a la primera
> entrega"**, e incluir modelos + evaluación + soporte MLflow/EC2 + tablero + equipo.
> Por eso los cambios de abajo se redactan como **secciones del reporte nuevo**,
> citando de dónde vienen respecto a la Entrega 1.

---

## Resumen del mapeo

| # | Comentario del evaluador | Rúbrica | Ubicación en Entrega 1 (PDF) | Acción |
|---|---|---|---|---|
| A | "aterrizar la utilidad, así se argumenta en términos académicos generales y no se define el alcance" | Problema y contexto 14/15 | **pág. 2** párr. 6 (cierre "Este proyecto desarrolla una solución integral…") y **pág. 3** "ALCANCE DEL PROYECTO" viñetas *Datos* / *Limitaciones* | Reescribir el cierre con **usuario, entrada, salida y no-alcance concretos** |
| B | "la propia exploración demuestra que el archivo no contiene contextos de cita reales" | Datos 8/10 | **pág. 4** "DESCRIPCIÓN DEL CONJUNTO DE DATOS" párr. 1–2 y **pág. 5** "ESTRUCTURACIÓN DEL DATASET" (lista de 5 variables, `base_label`) | Sustituir por la descripción del **dataset enriquecido** (citas reales) + tabla de variables nueva |
| C | "no se explora la variable objetivo (base-label) SEGÚN lo que definieron ni su balance entre las nueve categorías" | Exploración **20/30** | **pág. 5–8** "EXPLORACIÓN DE LOS DATOS" completa (Estructura, Longitud, Año, Vocabulario, Conclusiones) | Anteponer **§2 (variable objetivo)** y rehacer la exploración sobre el dataset nuevo |
| D | "la maqueta asume datos que no se mostraron consistentes en la exploración de variables" | Maqueta 18/20 | **pág. 8–10** "MAQUETA DEL PROTOTIPO": bloque *Resultados* (viñeta "Top-3 de pasajes recuperados…") y capturas pág. 9–11 | Marcar el panel **Top-3 como ilustrativo/pendiente de datos**; la maqueta operativa de la Entrega 2 es el clasificador de 9 clases |
| E | Soporte de experimentos (requisito nuevo de la semana 4) | — | No existe en Entrega 1 | Nueva sección "Experimentos (MLflow + EC2)" con `docs/soportes/01..03` |
| F | Reporte de trabajo en equipo | 10/10 | **pág. 12** tabla de responsables | Actualizar con las actividades de la Entrega 2 |

---

## A — Aterrizar la utilidad y el alcance

**Dónde:** `Entrega 1 Proyectos.pdf`, **pág. 2, último párrafo** de "PROBLEMA A
ABORDAR Y CONTEXTO" (el que empieza *"Este proyecto desarrolla una solución
integral que combina embeddings…"*) y **pág. 3, "ALCANCE DEL PROYECTO"**.

**Problema:** el cierre habla de "optimizar la revisión de literatura, la
validación de afirmaciones y el análisis de impacto bibliométrico con IA" — es
una promesa académica amplia, sin usuario ni entregable medible.

**Texto propuesto (reemplaza ese párrafo):**

> **Utilidad concreta y alcance.** El producto de este micro-proyecto es un
> **clasificador de la función retórica de una cita** expuesto vía API y tablero.
> *Usuario objetivo:* un investigador o asistente de revisión de literatura que,
> al leer un artículo, quiere saber **para qué** se está citando cada referencia
> (¿es antecedente, comparación, evidencia, un vacío que el trabajo ataca…?).
> *Entrada:* una o dos oraciones en inglés que contienen la mención bibliográfica
> (*citation context*) y, opcionalmente, la sección donde aparece.
> *Salida:* una de **9 funciones de cita** con su probabilidad y un aviso de
> ambigüedad cuando dos clases quedan muy cerca.
> *Fuera de alcance en esta entrega:* (i) la **recomendación local de fragmentos
> del artículo citado** (Top-3 *chunks*), que requiere el texto completo del
> documento citado y se difiere a una versión posterior (v3); (ii) idiomas
> distintos del inglés; (iii) la resolución de la referencia a un identificador
> bibliográfico. El valor entregable se mide con **F1-macro sobre las 9 clases**
> en un conjunto de validación con aislamiento por documento.

**En "ALCANCE / Datos" (pág. 3)** cambiar *"enfocado primordialmente en Ciencias
de la Computación e Inteligencia Artificial"* por:

> Contextos de cita en inglés de **literatura científica multidominio** (predominio
> de Ciencias de la Computación, pero con presencia de biomedicina, física y
> química — ver EDA §6). El dominio se registra como variable a monitorear.

**Evidencia:** `docs/EDA_ENTREGA2.md` §6 (términos y ejemplos fuera de CS).

---

## B — Descripción del conjunto de datos (citas reales)

**Dónde:** `Entrega 1 Proyectos.pdf`, **pág. 4 "DESCRIPCIÓN DEL CONJUNTO DE DATOS"**
(párrafos 1–2 y la lista de 9 categorías) y **pág. 5 "ESTRUCTURACIÓN DEL DATASET"**
(lista de 5 variables, con `base_label` descrita como *"categoría base de
referencia"* y `rhetorical_section` *"inicialmente Abstract"*).

**Problema:** la Entrega 1 describía un archivo de 24 855 registros que, según su
propia exploración (pág. 6), **eran abstracts, no contextos de cita**
(`base_label` constante = 0, `rhetorical_section` constante = "Abstract").

**Texto propuesto (reemplaza "DESCRIPCIÓN…" y "ESTRUCTURACIÓN…"):**

> **Conjunto de datos (actualizado en la Entrega 2).** Se reemplaza el corpus
> preliminar de abstracts por un **dataset enriquecido de intención de cita** con
> contextos de cita **reales**: fragmentos de 1–2 oraciones alrededor de una
> mención bibliográfica, extraídos de artículos científicos en inglés.
>
> | Partición | Registros | `citing_paper_id` únicos | Uso |
> |---|--:|--:|---|
> | Entrenamiento | 12 600 | 12 600 | fit de los modelos |
> | Validación | 2 700 | 2 700 | selección de modelo / hiperparámetros |
> | Prueba (para anotación humana) | 2 700 | 2 700 | evaluación final + conjunto para acuerdo interanotador |
> | **Maestro balanceado** | **18 000** | 18 000 | 2 000 ejemplos por clase; se particiona en los 3 anteriores |
>
> **Variables** (`data/processed/dataset_manifest.json`):
>
> | Variable | Tipo | Descripción |
> |---|---|---|
> | `citation_context` | texto | oración(es) en inglés con la mención bibliográfica. Mediana **26 palabras** (Entrega 1: 163 → eran abstracts). ~51 % con marcador de cita explícito detectable. |
> | `rhetorical_section` | categórica | sección de origen; 558 valores crudos → se normaliza a **10 secciones canónicas** (`rhetorical_section_canon`). 1.5 % nulos. |
> | `label` (= `base_label`) | categórica (9) | **variable objetivo**: función retórica de la cita. |
> | `citing_paper_id` | id | artículo citante; **clave de la regla de no-leakage** entre particiones. |
> | `cited_paper_id`, `pair_id`, `top{1,2,3}_cited_chunk` | — | presentes en el CSV crudo pero **no se usan** en la Entrega 2: `cited_paper_id` es un espejo de `citing_paper_id` y los *chunks* son sintéticos (ver punto D). |
>
> **Trazabilidad.** El archivo maestro y las particiones se versionan con **DVC**
> sobre `s3://citation-dvcstore-tema1`; el cambio de hash en `data/raw.dvc`
> corresponde a esta nueva versión de los datos.

**Evidencia:** `docs/EDA_ENTREGA2.md` §1 y §3; `data/processed/dataset_manifest.json`.

---

## C — Exploración de la variable objetivo y su balance (el cambio de mayor peso: 20/30)

**Dónde:** `Entrega 1 Proyectos.pdf`, **pág. 5–8, "EXPLORACIÓN DE LOS DATOS"**
completa. La exploración actual cubre estructura, longitud, año y vocabulario de
los abstracts, y sus "Conclusiones" (pág. 8) admiten que el archivo no sirve.
**No hay ni un gráfico de `base_label`.**

**Acción:** rehacer la sección sobre el dataset nuevo y **empezar por la variable
objetivo**. Insertar como **primer bloque** de "EXPLORACIÓN DE LOS DATOS":

> **1. Variable objetivo — `label` (9 funciones de cita).**
> Definición operativa: función retórica de la cita, 9 categorías mutuamente
> excluyentes (Background, Gap, Basis, Comparison, Application,
> Modification/Improvement, Evidence, Identification of the Originator,
> Further Reading), codificadas `label_id` 0–8.
>
> *Balance (entrenamiento, n = 12 600):* cada clase entre **10.9 % y 11.4 %**.
> Ratio de desbalance (clase mayor / menor) = **1.04**; entropía normalizada de la
> distribución = **1.00**. En validación y prueba el ratio sube a 1.16–1.17,
> siempre lejos del desbalance "fuerte" que reporta la literatura para esta tarea.
>
> *Figura:* `docs/eda_entrega2/01_variable_objetivo.png` (barras por partición).
>
> *Lectura para el negocio:* el balance es **artificial** — el maestro se
> construyó forzando ~2 000 ejemplos por clase. Es cómodo para entrenar y comparar
> modelos con F1-macro, pero implica que las métricas **no reflejan la prevalencia
> real** de cada función de cita; en una versión de producción (v3) habrá que
> reevaluar con distribución natural y, probablemente, reponderar la pérdida.
>
> *Consistencia del split:* las 9 clases mantienen ~11 % en train/val/test y
> ninguna está ausente → la partición preserva la estratificación.

Después de ese bloque, versión resumida del resto de la exploración (sobre el
dataset nuevo), citando figuras de `docs/eda_entrega2/`:

- **Longitud del contexto** (`02_longitud_contexto.png`, `03_longitud_por_clase.png`):
  mediana 26 palabras (6× menor que en la Entrega 1 — prueba de que ahora sí son
  citas); `Background` es la clase más corta, `Comparison` la más larga → hay
  señal de longitud.
- **Sección retórica × función de cita** (`04_seccion_x_label.png`):
  `P(label | sección)` muestra asociaciones (p. ej. `Application` sube en Métodos),
  por eso la sección canónica se concatena como *feature*.
- **No-leakage** (§5 del EDA): 0 solapamiento de `citing_paper_id` entre train/val/test.
- **Ruido**: ~49 % de contextos sin marcador de cita por regex; se reporta como
  límite de calidad de datos.

**Conclusiones de la exploración (reemplazan las de pág. 8):**

1. La variable objetivo está **definida y explorada**: 9 clases, prácticamente
   uniformes por construcción (ratio 1.04, entropía 1.00).
2. Los `citation_context` son **citas reales** (mediana 26 palabras, ~51 % con
   marcador explícito), a diferencia del corpus de la Entrega 1.
3. El componente de recomendación de *chunks* **no tiene datos consistentes** y se
   difiere a v3 (ver punto D).
4. El balance artificial es una **amenaza a la validez externa** que se documenta
   y se retoma en v3.

**Evidencia:** `docs/EDA_ENTREGA2.md` §2–§6 + `docs/eda_entrega2/*.png` + `eda_stats.json`.

---

## D — Maqueta y datos consistentes

**Dónde:** `Entrega 1 Proyectos.pdf`, **pág. 8–9 "MAQUETA DEL PROTOTIPO"**, bloque
*Resultados* — en concreto la viñeta *"Top-3 de pasajes recuperados del artículo
citado, con su sección de origen, posición y similitud y mapa de calor…"* — y las
capturas de pág. 9–11 que muestran ese panel poblado.

**Problema (confirmado con números):**

| Comprobación (train) | Resultado |
|---|---|
| `citing_paper_id` == `cited_paper_id` | **100 %** |
| `top1_cited_chunk.text` == `citation_context` | **94.8 %** |
| Similitudes distintas en top1 / top2 / top3 | {0.85, 0.92} / {0.73, 0.74} / {0.61} (constantes) |
| top2 / top3 | plantillas *"Foundational background… / Experimental setup…"* |

No existe artículo citado real: el "Top-3" del CSV es un **placeholder**.

**Texto propuesto (nota a añadir bajo la maqueta):**

> **Nota sobre el panel "Top-3 de pasajes recuperados".** La exploración de datos
> (EDA §4) muestra que el dataset disponible **no contiene el texto del artículo
> citado** (`cited_paper_id` es un espejo del citante y los fragmentos son
> plantillas). Por lo tanto, en la Entrega 2 ese panel se marca como
> **ilustrativo / pendiente de datos** y **no** se conecta a un modelo. La maqueta
> operativa de esta entrega se limita a lo que los datos sí soportan: entrada de un
> *citation context* + sección, y salida de la **función de cita** (9 clases) con
> probabilidades, aviso de ambigüedad y señales léxicas. La recuperación de
> *chunks* reales (descargar el PDF citado, segmentar ≤ 300 palabras, similitud con
> SciBERT) queda como **trabajo futuro (v3)**.

Además: en las capturas del tablero de la Entrega 2, el panel Top-3 debe
mostrarse **vacío con la etiqueta "pendiente de datos"**, no poblado con datos
inventados.

**Evidencia:** `docs/EDA_ENTREGA2.md` §4; `scripts/eda_entrega2.py` (bloque `retrieval_top3`).

---

## E — Nueva sección "Experimentos (MLflow + EC2)"

**Dónde:** sección nueva del reporte de la Entrega 2 (no existe en Entrega 1).
Requisito explícito de la semana 4.

**Contenido a incluir:**

1. **Modelos** (de `docs/MODELOS_ENTREGA2.md`, tabla comparativa):

   | Modelo | F1-macro val | F1-macro test | Brecha train→val | Sobreajuste |
   |---|--:|--:|--:|---|
   | v1 — TF-IDF + Regresión Logística | **0.497** | 0.531 | **0.326** | severo |
   | v2 — SciBERT fine-tune (iteración intermedia) | ⟨v2_val_f1⟩ | ⟨v2_test_f1⟩ | ⟨v2_gap⟩ | ⟨v2_veredicto⟩ |

   *(v1 ya medido localmente; v2 lo llena `analyze_metrics.py` tras el run en EC2.)*

2. **Análisis de error y sobreajuste** (de `docs/MODELOS_ENTREGA2.md` §3):
   - v1 memoriza el entrenamiento (train F1 0.82 vs val 0.50): sobreajuste severo
     del modelo lineal con bi-gramas; es el **baseline honesto**.
   - v2 reduce la brecha a ⟨…⟩ y sube F1-macro val en **+⟨…⟩ puntos**; **no se
     optimiza más a propósito** — el margen restante es el objetivo de v3
     (early-stopping agresivo, weight-decay, *class-balanced loss*, datos reales).
   - Clases más confundidas (matriz de confusión de v2 en MLflow): ⟨p. ej.
     `Basis`↔`Application`, `Identification_of_the_Originator`↔`Background`⟩.
   - Brecha val→test pequeña (v1: −0.03) → el split es representativo, sin fuga.

3. **Soportes** (`docs/soportes/`):
   - `01_ec2_consola.png` — consola EC2 con **Instance ID + IP pública + usuario/rol IAM**.
   - `02_security_group.png` — *inbound rules* con **puerto 5000** habilitado.
   - `03_mlflow_ui.png` — MLflow en `http://<IP>:5000` con **la IP visible** y los runs v1/v2.
   - Estado final de la instancia: **stopped** (no terminated).

**Evidencia:** experimento MLflow `scif-citation-intent`; `docs/MODELOS_ENTREGA2.md`.

---

## F — Reporte de trabajo en equipo

**Dónde:** `Entrega 1 Proyectos.pdf`, **pág. 12, tabla "REPORTE DEL TRABAJO EN EQUIPO"**.

Actualizar con las actividades de la Entrega 2 (ajustar nombres según quien ejecute):

| Actividad (Entrega 2) | Responsable |
|---|---|
| Consolidación y estandarización del dataset enriquecido + DVC | ⟨…⟩ |
| EDA de la variable objetivo y calidad de datos (`scripts/eda_entrega2.py`) | ⟨Henry Forigua⟩ |
| Servidor de experimentos (EC2 + MLflow) y soportes | ⟨Alvaro Ruiz⟩ |
| Modelos v1 / v2 y análisis de sobreajuste | ⟨…⟩ |
| Tablero según maqueta ajustada | ⟨David Indaburu⟩ |
| Redacción del reporte y cierre de feedback | ⟨Estefania Sanchez / Juan D. Marin⟩ |

---

## Checklist de inserción en el reporte final (≤ 10 páginas)

- [ ] **pág. 1** (resumen ≤ 1 pág.): contexto + pregunta + **alcance concreto (A)** + **1 párrafo de datos nuevos (B)**, con la frase "Cambios respecto a la Entrega 1".
- [ ] **Exploración**: bloque de **variable objetivo (C)** + 2 figuras (`01_variable_objetivo.png`, `04_seccion_x_label.png`).
- [ ] **Modelos y evaluación**: tabla comparativa + análisis de sobreajuste **(E)**.
- [ ] **Observaciones/conclusiones**: v1 sobreajusta, v2 intermedio, balanceo artificial, retrieval → v3.
- [ ] **Tablero**: descripción + nota de maqueta ajustada **(D)** + capturas con Top-3 marcado "pendiente".
- [ ] **Soportes**: 3 pantallazos EC2/MLflow **(E)**; EC2 detenida.
- [ ] **1 pág.**: reporte de trabajo en equipo **(F)**.
