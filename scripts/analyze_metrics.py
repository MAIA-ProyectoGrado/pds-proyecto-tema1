"""
Analisis de metricas de validacion, error y sobreajuste a partir de
docs/model_results.json (generado por scripts/train.py).

Produce docs/MODELOS_ENTREGA2.md con:
  * tabla comparativa v1 vs v2 (train / val / test)
  * brechas de generalizacion (train-val y val-test)
  * veredicto de sobreajuste por modelo
  * checklist de conclusiones para el reporte

Uso:
    python scripts/analyze_metrics.py --results docs/model_results.json --out docs/MODELOS_ENTREGA2.md
"""
import argparse
import json
from pathlib import Path


def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/d"


def row(name, m):
    return (f"| {name} | {fmt(m.get('train_f1_macro'))} | {fmt(m.get('val_f1_macro'))} | "
            f"{fmt(m.get('test_f1_macro'))} | {fmt(m.get('val_accuracy'))} | "
            f"{fmt(m.get('test_accuracy'))} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="docs/model_results.json")
    ap.add_argument("--out", default="docs/MODELOS_ENTREGA2.md")
    args = ap.parse_args()

    res = json.loads(Path(args.results).read_text())
    lines = []
    P = lines.append

    P("# Modelos desarrollados y evaluación — Entrega 2 (SCIF)\n")
    P("> Generado por `scripts/analyze_metrics.py`. Fuente: MLflow "
      "(experimento `scif-citation-intent`).\n")
    P("Tarea: **clasificación de la función de cita** (9 clases, balanceadas). "
      "Métrica principal **F1-macro**; baseline aleatorio = 0.111.\n")

    P("## 1. Tabla comparativa\n")
    P("| Modelo | F1-macro train | F1-macro val | F1-macro test | Acc val | Acc test |")
    P("|---|--:|--:|--:|--:|--:|")
    if "v1" in res:
        P(row("v1 — TF-IDF + LogReg", res["v1"]["metrics"]))
    if "v2" in res:
        P(row("v2 — SciBERT fine-tune (iteración intermedia)", res["v2"]["metrics"]))
    P("")

    P("## 2. Sobreajuste y brechas de generalización\n")
    P("| Modelo | Brecha train→val (F1) | Brecha val→test (F1) | Veredicto |")
    P("|---|--:|--:|---|")
    for k in ("v1", "v2"):
        if k not in res:
            continue
        of = res[k].get("overfitting", {})
        P(f"| {k} | {fmt(of.get('gap_f1_macro_train_val'))} | "
          f"{fmt(of.get('gap_val_test_f1'))} | {of.get('veredicto','n/d')} |")
    P("")
    P("Criterio: brecha train→val > 0.25 = sobreajuste **severo**; 0.12–0.25 = **moderado**; "
      "< 0.12 = **bajo**.\n")

    P("## 3. Lectura de resultados (rellenar/confirmar en el reporte)\n")
    P("- [ ] v1 memoriza el train (brecha alta) → evidencia de sobreajuste del modelo lineal "
      "con n-gramas; regularización (C↓) y `min_df`↑ lo reducen a costa de F1.")
    P("- [ ] v2 mejora F1-macro de validación respecto a v1 en +____ puntos.")
    P("- [ ] v2 conserva brecha train→val moderada → hay margen para v3 "
      "(early-stopping más agresivo, weight-decay, class-balanced loss, más datos reales).")
    P("- [ ] Clases más confundidas (ver matrices en MLflow): `Basis`↔`Application`, "
      "`Identification_of_the_Originator`↔`Background` — coherente con su cercanía semántica.")
    P("- [ ] La brecha val→test es pequeña → el split es representativo y no hay fuga.")
    P("- [ ] El balanceo del dataset es artificial: en producción (v3) reevaluar con "
      "distribución realista y reponderar.\n")

    P("## 4. Trazabilidad MLflow\n")
    for k in ("v1", "v2"):
        if k in res:
            P(f"- {k}: run_id `{res[k].get('run_id','?')}`")
    P("\nModelos registrados en el Model Registry como `scif-citation-intent` "
      "(v1 = versión 1, v2 = versión 2). El mejor pasa a `Staging`/`Production` "
      "y es el que empaqueta la API.\n")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("OK ->", args.out)


if __name__ == "__main__":
    main()
