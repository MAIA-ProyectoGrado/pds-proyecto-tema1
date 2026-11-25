"""Pruebas de `scripts/extraer_corridas_notebook.py` con salidas sintéticas del notebook."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.scif_predict import LABELS  # noqa: E402
from scripts.extraer_corridas_notebook import NOTEBOOK, leer_corrida, leer_notebook, tabla_f1  # noqa: E402

SALIDA = """INICIANDO EXPERIMENTO: Run2_SciBERT_ConfigB_Top3Chunks
Epoca 1/2 [424.2s] - Train Loss: 1.4694 | Val Loss: 1.0810 | Val Macro-F1: 0.5901
Epoca 2/2 [424.2s] - Train Loss: 0.9076 | Val Loss: 0.9846 | Val Macro-F1: 0.6408
REPORTE DE CLASIFICACION (VALIDATION):
                                  precision    recall  f1-score   support

""" + "\n".join(f"{c:>34s}       0.70      0.60      0.65       100" for c in LABELS) + """

                        accuracy                           0.68      2280
                       macro avg       0.67      0.66      0.66      2280
                    weighted avg       0.68      0.68      0.68      2280
"""


def test_lee_epocas_y_reporte():
    c = leer_corrida(SALIDA)
    assert c["corrida"] == "Run2_SciBERT_ConfigB_Top3Chunks" and c["nombre"].startswith("Run 2")
    assert [e["f1_macro_validacion"] for e in c["epocas"]] == [0.5901, 0.6408]
    assert c["epocas"][1]["perdida_validacion"] == 0.9846
    assert set(c["validacion"]["por_clase"]) == set(LABELS)
    assert c["validacion"]["exactitud"] == 0.68 and c["validacion"]["macro_avg"]["f1"] == 0.66


def test_texto_sin_experimento_no_es_una_corrida():
    assert leer_corrida("Epoca 1/2 [1s] - Train Loss: 1 | Val Loss: 1 | Val Macro-F1: 0.5") is None


def test_tabla_incluye_corridas_y_test():
    corrida = leer_corrida(SALIDA)
    ref = {"f1_por_clase": {c: 0.5 for c in LABELS}, "f1_macro": 0.5}
    test = {"modelos": {m: {"human_label": ref, "label": ref} for m in ("scibert", "tfidf-logreg")}}
    filas = tabla_f1([corrida], test)
    assert [f["clase"] for f in filas] == LABELS + ["Macro F1"]
    assert filas[0]["Run 2: SciBERT, config. B, validación"] == 0.65
    assert filas[-1]["SciBERT B, test (revisión humana)"] == 0.5


def test_el_notebook_tiene_las_cinco_corridas():
    corridas = leer_notebook(NOTEBOOK)
    assert [c["corrida"][:4] for c in corridas] == ["Run1", "Run2", "Run3", "Run4", "Run5"]
    assert corridas[1]["epocas"][-1]["f1_macro_validacion"] == 0.6633
