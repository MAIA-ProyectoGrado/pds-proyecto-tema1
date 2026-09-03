# Modelos desarrollados y evaluación — Entrega 2 (SCIF)

> **Plantilla.** Las filas de v1 son medidas reales obtenidas localmente
> (`scripts/train.py --stage v1`). Las de v2 y los `run_id` los completa
> `scripts/analyze_metrics.py` tras el entrenamiento en la EC2. Fuente de verdad:
> MLflow, experimento `scif-citation-intent`.

Tarea: **clasificación de la función de cita** (9 clases, balanceadas por
construcción). Métrica principal **F1-macro**; baseline aleatorio = 0.111.

## 1. Tabla comparativa

| Modelo | F1-macro train | F1-macro val | F1-macro test | Acc val | Acc test |
|---|--:|--:|--:|--:|--:|
| v1 — TF-IDF (1–2 gramas) + Regresión Logística | 0.823 | **0.497** | 0.531 | 0.503 | 0.538 |
| v2 — SciBERT fine-tune (iteración intermedia) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

## 2. Sobreajuste y brechas de generalización

| Modelo | Brecha train→val (F1) | Brecha val→test (F1) | Veredicto |
|---|--:|--:|---|
| v1 | **0.326** | −0.034 | severo |
| v2 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

Criterio: brecha train→val > 0.25 = sobreajuste **severo**; 0.12–0.25 = **moderado**; < 0.12 = **bajo**.

## 3. Lectura de resultados

- **v1 sobreajusta de forma severa**: train F1-macro 0.82 vs val 0.50. El modelo
  lineal con bi-gramas memoriza n-gramas del entrenamiento. Bajar `C` y subir
  `min_df` reduce la brecha pero también el F1 de validación → es el **baseline honesto**.
- v1 ya está **muy por encima del azar** (0.50 vs 0.111): la tarea es aprendible
  desde el texto del contexto + la sección.
- La **brecha val→test es ~0** (−0.03) → el split respeta el aislamiento por
  documento y es representativo; no hay fuga.
- Clases con más error en v1 (`classification_report_v1_val.json`):
  `Basis` (F1 0.34) e `Identification_of_the_Originator` (F1 0.39) se confunden
  con `Application` y `Background` respectivamente.
- **v2 (pendiente de rellenar):** se espera F1-macro val ⟨0.60–0.72⟩ y una brecha
  train→val ⟨moderada⟩. **No se optimiza más a propósito**; el margen restante es
  el objetivo de **v3** (early-stopping agresivo, `weight_decay`↑, *class-balanced
  loss*, incorporar contextos de cita reales adicionales y limpiar el ~49 % sin
  marcador).
- **Amenaza a la validez externa:** el dataset está balanceado artificialmente
  (2 000/clase). Las métricas no reflejan la prevalencia real de cada función de
  cita; v3 debe reevaluar con distribución natural.

## 4. Trazabilidad MLflow

- v1: run_id ⟨…⟩
- v2: run_id ⟨…⟩

Ambos modelos quedan en el Model Registry como `scif-citation-intent`
(v1 = versión 1, v2 = versión 2). El mejor pasa a `Staging`/`Production` y es el
que empaqueta la API (`api/`).
