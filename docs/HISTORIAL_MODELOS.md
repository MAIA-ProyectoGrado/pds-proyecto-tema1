# Historial de modelos

Evolución de los clasificadores de función de cita (9 clases) a lo largo del
proyecto. La métrica es el F1-macro. La API y el tablero sirven solo los dos
modelos v4; lo anterior queda archivado para trazabilidad.

| Versión | Modelo | Datos y partición | Entrada | F1 val | F1 test | Artefacto |
|---|---|---|---|--:|---|---|
| v1 | TF-IDF + regresión logística | Entrega 2, dataset v3 | contexto + sección | 0,516 | 0,518 | `models/archivo/scif-v1-tfidf-logreg/` (git) |
| v2a | MiniLM congelado + regresión logística | Entrega 2, dataset v3 | contexto + sección | 0,442 | 0,445 | Solo métricas |
| v2b | DistilBERT ajustado (5.000 ejemplos) | Entrega 2, snapshot de 18.000 | contexto + sección | 0,573 | 0,577 | Solo métricas |
| v3 | SciBERT ajustado, 4 épocas | Entrega 2, dataset v3 | contexto + sección | 0,640 | 0,634 | Solo métricas (los pesos no se guardaron) |
| v3 | SPECTER2 ajustado | Entrega 2, dataset v3 | contexto + sección | 0,637 | | Solo métricas |
| v3-1ep | SciBERT ajustado, 1 época | Entrega 2, dataset v3 | contexto + sección | | | `models/archivo/scif-v3-scibert-1ep/` (DVC) |
| **v4** | **SciBERT, configuración B** | Dataset final (14.937), test revisado por humanos | contexto + sección + 3 oraciones clave del abstract citado | 0,663 | **0,532** (revisión humana) · 0,667 (etiqueta automática) | `models/scif-scibert-chunks/` (DVC) |
| **v4** | **TF-IDF + regresión logística** | Dataset final (14.937), test revisado por humanos | contexto + sección | 0,506 | **0,469** (revisión humana) · 0,523 (etiqueta automática) | `models/scif-v4-tfidf-logreg/` (git) |

## Cómo leer la tabla

- **Las cifras de la entrega 2 y las de v4 no son comparables entre sí.** Vienen de
  datasets y particiones distintas. En v4, además, la prueba está revisada por
  humanos.
- **v4, test con dos referencias.** Cada cita del test tiene la etiqueta automática
  (`label`), con la que se entrenó y validó, y la revisión humana (`human_label`).
  Coinciden en el 73,9 % de las citas (kappa de Cohen 0,70): en el 92,1 % de las
  marcadas con certeza alta y en el 50,9 % de las marcadas en duda. La validación
  solo tiene etiqueta automática, así que su comparación directa es con el test
  contra `label`. El detalle por certeza, por clase y con y sin SciTail está en
  `docs/modelos_v4/test_referencias_metricas.json` (`scripts/evaluate_test_humano.py`).
- **v4 SciBERT, validación.** Coincide con la del notebook de entrenamiento (0,6633).
  `scripts/evaluate_v4.py` arma la entrada y llama al modelo igual que el notebook:
  le pasa solo `input_ids` y `attention_mask`.
- **v4 SciBERT, test por subconjunto (etiqueta automática).** Con oraciones clave:
  n = 1.790, F1 0,640. Sin oraciones clave: n = 451, F1 0,715, con solo 5 de las 9
  clases presentes. Los dos subconjuntos no se pueden comparar: las citas sin
  oraciones clave vienen casi todas de SciCite, cuyas clases son más fáciles.

## Fuentes

| Versión | Métricas | Cómo se reproducen |
|---|---|---|
| v1, v2a, v3 | `docs/historial/modelos_entrega2/model_results_v3.json` y `v3_*_report_*.json` | `scripts/train.py` (ver `docs/historial/MODELOS_ENTREGA2.md`) |
| v2b | `docs/historial/modelos_entrega2/model_results_ec2.json` | `scripts/train.py --stage v2` |
| v4 SciBERT | `docs/modelos_v4/scibert_chunks_metricas.json` | `scripts/convert_checkpoint.py` y `scripts/evaluate_v4.py` |
| v4 TF-IDF | `docs/modelos_v4/tfidf_metricas.json` | `scripts/train_tfidf_v4.py` |
| v4, test contra las dos referencias | `docs/modelos_v4/test_referencias_metricas.json` y `predicciones_test.csv` | `scripts/evaluate_test_humano.py` |

El estado completo del repositorio al cierre de la entrega 3 está en la etiqueta de
git `entrega-3`.
