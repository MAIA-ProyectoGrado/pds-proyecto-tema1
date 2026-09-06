"""
Evaluacion del baseline v1 (TF-IDF + Regresion Logistica) SIN MLflow.

Reproduce las cifras de la seccion 3 del reporte y genera los artefactos:
  docs/modelos_entrega2/v1_metricas.json          metricas train/val/test
  docs/modelos_entrega2/v1_reporte_val.csv        precision/recall/F1 por clase (val)
  docs/modelos_entrega2/v1_matriz_confusion_val.png
  docs/modelos_entrega2/v1_sweep_regularizacion.png   F1 train vs val segun C

Uso:
    python scripts/eval_baseline.py --data_dir dataset --out docs/modelos_entrega2
    # tambien acepta --data_dir data/raw (particiones estandarizadas)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline

LABELS = sorted([
    "Application", "Background", "Basis", "Comparison", "Evidence",
    "Further_Reading", "Gap", "Identification_of_the_Originator",
    "Modification_Improvement",
])


def load(data_dir: Path):
    names = {
        "train": ["train.csv", "citation_intent_train_enriched.csv"],
        "val": ["val.csv", "citation_intent_val_enriched.csv"],
        "test": ["test.csv", "test_para_anotacion_humana_enriched.csv"],
    }
    out = {}
    for split, cands in names.items():
        path = next((data_dir / c for c in cands if (data_dir / c).exists()), None)
        if path is None:
            raise FileNotFoundError(f"No encuentro {split} en {data_dir}")
        df = pd.read_csv(path)
        sec_col = "rhetorical_section_canon" if "rhetorical_section_canon" in df.columns else "rhetorical_section"
        df["_text"] = df["citation_context"].astype(str) + " [SEC] " + df[sec_col].fillna("NA").astype(str)
        out[split] = df
    return out


def metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_micro": round(f1_score(y_true, y_pred, average="micro"), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--out", default="docs/modelos_entrega2")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    d = load(Path(args.data_dir))

    def make_pipe(C):
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                      max_features=50000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=3000, C=C, class_weight="balanced")),
        ])

    # --- modelo final (C=1.0) --------------------------------------------------
    import time
    pipe = make_pipe(1.0)
    t0 = time.time()
    pipe.fit(d["train"]["_text"], d["train"]["label"])
    fit_s = round(time.time() - t0, 1)

    result = {"config": {"vectorizer": "TF-IDF (1,2)-gramas, min_df=2, max_features=50000, sublinear_tf",
                         "classifier": "LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)",
                         "entrada": "citation_context + ' [SEC] ' + rhetorical_section_canon",
                         "fit_seconds": fit_s},
              "metricas": {}}
    preds = {}
    for split in ["train", "val", "test"]:
        preds[split] = pipe.predict(d[split]["_text"])
        result["metricas"][split] = metrics(d[split]["label"], preds[split])

    m = result["metricas"]
    result["sobreajuste"] = {
        "gap_f1_macro_train_val": round(m["train"]["f1_macro"] - m["val"]["f1_macro"], 4),
        "gap_accuracy_train_val": round(m["train"]["accuracy"] - m["val"]["accuracy"], 4),
        "gap_f1_macro_val_test": round(m["val"]["f1_macro"] - m["test"]["f1_macro"], 4),
        "veredicto": ("severo" if m["train"]["f1_macro"] - m["val"]["f1_macro"] > 0.25
                      else "moderado" if m["train"]["f1_macro"] - m["val"]["f1_macro"] > 0.12
                      else "bajo"),
    }

    # --- reporte por clase (val) --------------------------------------------------
    rep = classification_report(d["val"]["label"], preds["val"], labels=LABELS,
                                output_dict=True, zero_division=0)
    rows = [{"clase": l, "precision": round(rep[l]["precision"], 3),
             "recall": round(rep[l]["recall"], 3), "f1": round(rep[l]["f1-score"], 3),
             "soporte": int(rep[l]["support"])} for l in LABELS]
    pd.DataFrame(rows).to_csv(out / "v1_reporte_val.csv", index=False)
    result["reporte_val_por_clase"] = rows
    peor = sorted(rows, key=lambda r: r["f1"])[:3]
    result["peores_clases_val"] = [r["clase"] for r in peor]

    # --- matriz de confusion (val) ----------------------------------------------
    cm = confusion_matrix(d["val"]["label"], preds["val"], labels=LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ConfusionMatrixDisplay(cm, display_labels=[l.replace("_", " ")[:16] for l in LABELS]).plot(
        ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False, values_format=".2f")
    ax.set_title("v1 — matriz de confusion normalizada (validacion)")
    fig.tight_layout()
    fig.savefig(out / "v1_matriz_confusion_val.png", dpi=120)
    plt.close(fig)
    # pares mas confundidos (fuera de la diagonal)
    conf_pairs = []
    for i, a in enumerate(LABELS):
        for j, b in enumerate(LABELS):
            if i != j:
                conf_pairs.append((round(float(cm[i, j]), 3), a, b))
    result["pares_mas_confundidos_val"] = [
        {"real": a, "predicho": b, "tasa": v} for v, a, b in sorted(conf_pairs, reverse=True)[:5]
    ]

    # --- sweep de regularizacion ----------------------------------------------
    Cs = [0.1, 0.3, 1.0, 3.0, 10.0]
    sweep = []
    for C in Cs:
        p = make_pipe(C).fit(d["train"]["_text"], d["train"]["label"])
        tr = f1_score(d["train"]["label"], p.predict(d["train"]["_text"]), average="macro")
        va = f1_score(d["val"]["label"], p.predict(d["val"]["_text"]), average="macro")
        sweep.append({"C": C, "f1_train": round(tr, 4), "f1_val": round(va, 4),
                      "gap": round(tr - va, 4)})
    result["sweep_regularizacion"] = sweep
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([s["C"] for s in sweep], [s["f1_train"] for s in sweep], "o-", label="F1-macro train")
    ax.plot([s["C"] for s in sweep], [s["f1_val"] for s in sweep], "s-", label="F1-macro val")
    ax.set_xscale("log"); ax.set_xlabel("C (inverso de la regularizacion)")
    ax.set_ylabel("F1-macro"); ax.set_ylim(0, 1)
    ax.set_title("v1 — regularizacion vs sobreajuste")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "v1_sweep_regularizacion.png", dpi=120)
    plt.close(fig)

    (out / "v1_metricas.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nartefactos ->", out)


if __name__ == "__main__":
    main()
