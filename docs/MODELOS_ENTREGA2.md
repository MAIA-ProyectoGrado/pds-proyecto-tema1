# Modelos desarrollados y evaluación — Entrega 2 (SCIF)

Tarea: **clasificación de la función de cita** (9 clases). Métrica principal
**F1-macro** (baseline aleatorio = 0.111). Experimentos en **MLflow** sobre EC2
(experimento `scif-citation-intent`). Reproducible: `scripts/eval_baseline.py`,
`scripts/eval_v2.py`, `scripts/learning_curve_v1.py`, `scripts/train.py`.

Restricción de cómputo: AWS Academy Learner Lab → **sin GPU**, máx. `t3.large`
(2 vCPU / 8 GB).

## 1. Comparativa de enfoques

| # | Enfoque | F1 val | F1 test | Brecha train→val | Sobreajuste | Costo entren. |
|---|---|--:|--:|--:|---|--:|
| v1 | word TF-IDF (1-2g) + LogReg | 0.496 | 0.531 | 0.33 | **severo** | ~9 s |
| — | word+char TF-IDF + LogReg | 0.522 | 0.552 | 0.35 | severo | ~30 s |
| v2a | MiniLM emb. congeladas + LogReg | 0.442 | 0.474 | 0.07 | bajo | ~4 min |
| — | word+char TF-IDF ⊕ MiniLM + LogReg | 0.529 | 0.550 | 0.34 | severo | ~4 min |
| **v2b** | **DistilBERT fine-tune** (5 k, 2 ép., `max_len` 128) | **0.573** | **0.577** | **0.15** | **moderado** | ~55 min CPU |

**v2b (DistilBERT fine-tune) es el mejor modelo**: +0.077 de F1-macro val sobre
v1, con la mejor generalización (val ≈ test, brecha train→val la más baja de los
que superan el baseline).

## 2. Detalle de v1 — baseline (TF-IDF + Regresión Logística)

- Config: `TfidfVectorizer(1-2g, min_df=2, max_features=50k, sublinear_tf)` +
  `LogisticRegression(C=1.0, class_weight='balanced')`; entrada
  `citation_context + " [SEC] " + rhetorical_section_canon`.
- **Métricas**: F1-macro train 0.82 / **val 0.496** / test 0.531.
- **Sobreajuste severo**: brecha train→val 0.33. El barrido de `C`
  (`v1_sweep_regularizacion.png`) muestra que con `C=10` el train llega a F1 0.998
  y la validación se queda en 0.50 → **techo del enfoque léxico ≈ 0.50**.
- **Validación cruzada** 5-fold `GroupKFold` por `citing_paper_id`:
  `[0.497, 0.522, 0.511, 0.498, 0.525]` → **0.510 ± 0.012** (métrica estable).
- **Curva de aprendizaje** (`v1_learning_curve.png`): F1 val sube 0.32 → 0.50 al
  pasar de 500 a 12 600 ejemplos, con pendiente aún positiva → **más datos reales
  seguirían ayudando**.
- Peores clases (val): `Basis` F1 0.34, `Identification of the Originator` 0.39.

## 3. Detalle de v2a — embeddings congeladas + LogReg

`all-MiniLM-L6-v2` (384-dim, sin ajustar) + LogReg. **F1 val 0.442** (−0.05 vs v1)
pero brecha 0.07 (no sobreajusta). Conclusión: las embeddings semánticas
**congeladas** de un encoder de propósito general **no capturan** la función
retórica mejor que los n-gramas → hace falta **ajustar** el encoder.

## 4. Detalle de v2b — DistilBERT fine-tune (modelo elegido)

- Config: `distilbert-base-uncased`, submuestreo estratificado a 5 000, `lr 3e-5`,
  `batch 16`, `max_len 128`, **validación cada 78 pasos** + early-stopping (patience 2).
- **Curva de aprendizaje paso a paso** (`v2b_learning_curve.png`), F1-macro val:

  | paso | 78 | 156 | 234 | 312 | 390 | **468** | 546 | 624 |
  |---|--:|--:|--:|--:|--:|--:|--:|--:|
  | F1 val | 0.23 | 0.43 | 0.52 | 0.54 | 0.56 | **0.573** | 0.566 | 0.562 |

  Mejora sostenida hasta el paso 468 (~1.5 épocas); luego **empieza a bajar** →
  el early-stopping detiene y restaura el mejor checkpoint. La validación cada 78
  pasos **detectó el inicio del sobreajuste** y evitó ~1 época de cómputo inútil.
- **Métricas finales**: F1-macro train 0.72 / **val 0.573** / **test 0.577**;
  accuracy val 0.572.
- **Sobreajuste moderado**: brecha train→val 0.15 (vs 0.33 de v1);
  brecha val→test **−0.003** → generaliza muy bien, sin fuga.
- Desempeño por clase (val): mejores `Gap` 0.72, `Background` 0.70,
  `Application` 0.63; peores `Basis` 0.40, `Identification of the Originator` 0.47.
  Matriz de confusión: `v2b_matriz_confusion_val.png`.

## 5. ¿Es DistilBERT la mejor arquitectura posible bajo estas restricciones?

Es la mejor **de las probadas** y una elección sólida, pero no necesariamente
óptima. Dentro del mismo presupuesto (2 vCPU, sin GPU, ≤ 15 h):

- **SciBERT / SPECTER2 fine-tune**: encoder preentrenado en texto científico (línea
  SOTA de *citation intent*). ~2× más lento que DistilBERT en CPU (~1.5–2 h sobre
  5 k) → factible. Ganancia esperada +3–8 pts. **Primer candidato de v3.**
- **Fine-tune con datos completos** (12 600, no 5 k): la curva de aprendizaje de v1
  indica que más datos ayudan; ~2.5–3 h de CPU, cabe en presupuesto. +2–5 pts.
- **DeBERTa-v3-small**: suele ser el encoder pequeño más fuerte, tamaño similar.
- **No vale la pena** bajo la restricción: modelos *-large*, ensembles.

**El cuello de botella es la calidad de los datos, no la arquitectura.** Los
modelos publicados alcanzan ~0.80 F1 en datasets *limpios* de citation intent;
aquí el ~0.57 se explica por: ~49 % de contextos sin marcador de cita detectable,
etiquetas generadas por un juez-LLM (no anotación humana completa), balanceo
artificial. La palanca principal de v3 es **mejores datos / etiquetas**, y sólo
después el cambio de arquitectura.

## 6. Trazabilidad MLflow

| Run | run_id | Estado | F1 val |
|---|---|---|--:|
| `v1_baseline_tfidf_logreg` | `7803f9f615ad49b6b3725fddd036fac3` | FINISHED | 0.496 |
| `v2_embed_lr_all-MiniLM-L6-v2` | (ver UI) | FINISHED | 0.442 |
| `v2_distilbert-base-uncased` | `feb11c72723e41b2adce334e08ae7a3a` | FINISHED | **0.573** |

Ambos modelos principales quedan en el Model Registry como `scif-citation-intent`
(v1 = versión 1, v2b = versión 2). **v2b** es el que empaqueta la API.
