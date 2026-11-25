"""Evalúa el clasificador v4 (SciBERT, configuración B) sobre las particiones finales.

Reconstruye la entrada exactamente como el notebook de entrenamiento
(`models.scif_predict.build_pair_config_b`) y reporta, por partición y contra la
etiqueta automática (`label`): F1 macro y micro, precisión y exhaustividad macro,
F1 por clase y matriz de confusión. En el test separa además las citas con y sin
pasajes del artículo citado. La comparación del test contra la revisión humana
(`human_label`) está en `scripts/evaluate_test_humano.py`, que parte de las
predicciones que guarda este script.

Uso:
    python scripts/evaluate_v4.py --data-dir <carpeta con los CSV> \
        --model models/scif-scibert-chunks --out docs/modelos_v4/scibert_chunks_metricas.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.scif_predict import _valid, build_pair_config_b, model_kwargs  # noqa: E402

SPLITS = {"val": "val_enriquecido_top3_chunks.csv", "test": "test_enriquecido_top3_chunks.csv"}


def load_split(path: Path):
    import pandas as pd
    df = pd.read_csv(path)            # mismos valores faltantes (NaN) que en el notebook
    pairs = [
        build_pair_config_b(r.citation_context, r.rhetorical_section,
                            [r.top1_cited_chunk, r.top2_cited_chunk, r.top3_cited_chunk],
                            r.cited_paper_title)
        for r in df.itertuples(index=False)
    ]
    has_passages = [_valid(x) for x in df["top1_cited_chunk"]]
    return df, pairs, has_passages


def predict(model, tok, pairs, device, fmt, batch=16):
    """Tokeniza y llama al modelo igual que `models.scif_predict.Predictor`."""
    import torch
    preds = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            a, b = zip(*pairs[i:i + batch])
            enc = tok(list(a), list(b), max_length=fmt.get("max_len", 512), truncation=True,
                      padding=True, return_tensors="pt").to(device)
            preds.extend(model(**model_kwargs(enc, fmt)).logits.argmax(-1).cpu().tolist())
    return preds


def report(y_true, y_pred, labels):
    """Métricas de un conjunto. Los promedios macro usan solo las clases presentes en
    y_true: en un subconjunto donde falta una clase, promediar su F1 de 0 lo distorsiona."""
    from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    full = labels
    labels = [l for l in labels if l in set(y_true)]
    rep = classification_report(y_true, y_pred, labels=full, output_dict=True, zero_division=0)
    return {
        "n": len(y_true),
        "clases_presentes": len(labels),
        "f1_macro": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "f1_micro": round(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "f1_por_clase": {l: round(rep[l]["f1-score"], 4) for l in labels},
        "soporte_por_clase": {l: int(rep[l]["support"]) for l in full},
        "matriz_confusion": {"orden": full,
                             "valores": confusion_matrix(y_true, y_pred, labels=full).tolist()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--model", default=REPO_ROOT / "models" / "scif-scibert-chunks", type=Path)
    ap.add_argument("--out", default=REPO_ROOT / "docs" / "modelos_v4" / "scibert_chunks_metricas.json", type=Path)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="'cpu' evita saturar la memoria en equipos con poca RAM")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    fmt = json.loads((args.model / "input_format.json").read_text())

    out = {
        "modelo": "SciBERT configuración B (v4)",
        "carpeta_modelo": str(args.model.relative_to(REPO_ROOT)) if args.model.is_relative_to(REPO_ROOT) else str(args.model),
        "checkpoint_sha256": fmt.get("source_checkpoint", {}).get("sha256"),
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dispositivo": device,
        "nota": "La prueba humana no se usó para entrenar ni para elegir el checkpoint (se eligió con validación).",
        "particiones": {},
    }
    for split, fname in SPLITS.items():
        df, pairs, has_p = load_split(args.data_dir / fname)
        t = time.perf_counter()
        y_pred = [labels[i] for i in predict(model, tok, pairs, device, fmt, args.batch)]
        secs = time.perf_counter() - t
        y_true = df["label"].tolist()
        res = report(y_true, y_pred, labels)
        res["segundos_inferencia"] = round(secs, 1)
        if split == "test":
            con = [i for i, h in enumerate(has_p) if h]
            sin = [i for i, h in enumerate(has_p) if not h]
            res["con_pasajes"] = report([y_true[i] for i in con], [y_pred[i] for i in con], labels)
            res["sin_pasajes"] = report([y_true[i] for i in sin], [y_pred[i] for i in sin], labels)
        out["particiones"][split] = res
        pred_path = args.out.parent / f"scibert_chunks_predicciones_{split}.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        df.assign(prediccion=y_pred, con_pasajes=has_p)[
            ["citing_paper_id", "source_dataset", "rhetorical_section", "label", "prediccion", "con_pasajes"]
        ].to_csv(pred_path, index_label="fila")
        print(f"{split:5s} n={res['n']:5d}  F1-macro={res['f1_macro']:.4f}  F1-micro={res['f1_micro']:.4f}  ({secs:.0f} s)")
        if split == "test":
            for k in ("con_pasajes", "sin_pasajes"):
                print(f"      {k:12s} n={res[k]['n']:5d}  clases={res[k]['clases_presentes']}  "
                      f"F1-macro={res[k]['f1_macro']:.4f}  exactitud={res[k]['f1_micro']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
