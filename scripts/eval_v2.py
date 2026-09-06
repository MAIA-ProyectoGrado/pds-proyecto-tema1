"""
Evaluacion de v2 (sentence-embeddings congelados + Regresion Logistica) SIN MLflow.

v2 usa una representacion SEMANTICA (encoder de oraciones preentrenado) en lugar de
la LEXICA de v1 (TF-IDF), sin el costo de un fine-tuning completo -> apto para CPU
de 2 vCPU (Learner Lab).

Genera:
  docs/modelos_entrega2/v2_metricas.json
  docs/modelos_entrega2/v2_reporte_val.csv
  docs/modelos_entrega2/v2_matriz_confusion_val.png
  docs/modelos_entrega2/comparativa_v1_v2.json   (si existe v1_metricas.json)

Uso:
    python scripts/eval_v2.py --data_dir dataset --out docs/modelos_entrega2
    python scripts/eval_v2.py --st_model sentence-transformers/all-mpnet-base-v2   # mas lento, +calidad
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

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
        sec = "rhetorical_section_canon" if "rhetorical_section_canon" in df.columns else "rhetorical_section"
        df["_text"] = df["citation_context"].astype(str) + " [SEC] " + df[sec].fillna("NA").astype(str)
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
    ap.add_argument("--st_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    d = load(Path(args.data_dir))

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.st_model, device="cpu")

    t0 = time.time()
    emb = {s: enc.encode(d[s]["_text"].tolist(), batch_size=64,
                         show_progress_bar=True, normalize_embeddings=True)
           for s in ["train", "val", "test"]}
    encode_s = round(time.time() - t0, 1)

    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    t1 = time.time()
    clf.fit(emb["train"], d["train"]["label"])
    fit_s = round(time.time() - t1, 1)

    preds = {s: clf.predict(emb[s]) for s in ["train", "val", "test"]}
    result = {
        "config": {"encoder": args.st_model, "encoder_frozen": True,
                   "classifier": "LogisticRegression(C=1.0, class_weight='balanced')",
                   "entrada": "citation_context + ' [SEC] ' + rhetorical_section_canon",
                   "encode_seconds": encode_s, "fit_seconds": fit_s,
                   "embedding_dim": int(emb["train"].shape[1])},
        "metricas": {s: metrics(d[s]["label"], preds[s]) for s in ["train", "val", "test"]},
    }
    m = result["metricas"]
    result["sobreajuste"] = {
        "gap_f1_macro_train_val": round(m["train"]["f1_macro"] - m["val"]["f1_macro"], 4),
        "gap_accuracy_train_val": round(m["train"]["accuracy"] - m["val"]["accuracy"], 4),
        "gap_f1_macro_val_test": round(m["val"]["f1_macro"] - m["test"]["f1_macro"], 4),
        "veredicto": ("severo" if m["train"]["f1_macro"] - m["val"]["f1_macro"] > 0.25
                      else "moderado" if m["train"]["f1_macro"] - m["val"]["f1_macro"] > 0.12
                      else "bajo"),
    }

    rep = classification_report(d["val"]["label"], preds["val"], labels=LABELS,
                                output_dict=True, zero_division=0)
    rows = [{"clase": l, "precision": round(rep[l]["precision"], 3),
             "recall": round(rep[l]["recall"], 3), "f1": round(rep[l]["f1-score"], 3),
             "soporte": int(rep[l]["support"])} for l in LABELS]
    pd.DataFrame(rows).to_csv(out / "v2_reporte_val.csv", index=False)
    result["reporte_val_por_clase"] = rows

    cm = confusion_matrix(d["val"]["label"], preds["val"], labels=LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ConfusionMatrixDisplay(cm, display_labels=[l.replace("_", " ")[:16] for l in LABELS]).plot(
        ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False, values_format=".2f")
    ax.set_title(f"v2 ({args.st_model.split('/')[-1]}) — matriz de confusion (validacion)")
    fig.tight_layout(); fig.savefig(out / "v2_matriz_confusion_val.png", dpi=120); plt.close(fig)
    conf = [(round(float(cm[i, j]), 3), LABELS[i], LABELS[j])
            for i in range(9) for j in range(9) if i != j]
    result["pares_mas_confundidos_val"] = [
        {"real": a, "predicho": b, "tasa": v} for v, a, b in sorted(conf, reverse=True)[:5]]

    (out / "v2_metricas.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    v1p = out / "v1_metricas.json"
    if v1p.exists():
        v1 = json.loads(v1p.read_text())
        comp = {
            "v1": {"enfoque": "TF-IDF + LogReg",
                   **{k: v1["metricas"][k]["f1_macro"] for k in ["train", "val", "test"]},
                   "gap_train_val": v1["sobreajuste"]["gap_f1_macro_train_val"],
                   "veredicto": v1["sobreajuste"]["veredicto"]},
            "v2": {"enfoque": f"{args.st_model.split('/')[-1]} emb + LogReg",
                   **{k: m[k]["f1_macro"] for k in ["train", "val", "test"]},
                   "gap_train_val": result["sobreajuste"]["gap_f1_macro_train_val"],
                   "veredicto": result["sobreajuste"]["veredicto"]},
            "delta_val_f1_macro": round(m["val"]["f1_macro"] - v1["metricas"]["val"]["f1_macro"], 4),
        }
        (out / "comparativa_v1_v2.json").write_text(json.dumps(comp, indent=2, ensure_ascii=False))
        print(json.dumps(comp, indent=2, ensure_ascii=False))

    print(json.dumps({"config": result["config"], "metricas": m,
                      "sobreajuste": result["sobreajuste"]}, indent=2, ensure_ascii=False))
    print("\nartefactos ->", out)


if __name__ == "__main__":
    main()
