"""Entrena la línea base TF-IDF + regresión logística sobre las particiones finales (v4).

Es el modelo liviano del despliegue: lee solo el contexto de cita y la sección
normalizada (`models.scif_predict.build_input`), pesa unos megabytes y responde en
milisegundos en CPU. Hiperparámetros: TF-IDF de unigramas y bigramas (frecuencia
mínima 2, hasta 50.000 términos, escala sublineal) y regresión logística con C = 1
y pesos de clase balanceados (`PARAMS`).

Uso:
    python scripts/train_tfidf_v4.py --data-dir <carpeta con los CSV> \
        --out models/scif-v4-tfidf-logreg
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

from models.scif_predict import LABELS, build_input  # noqa: E402
from scripts.evaluate_v4 import report  # noqa: E402

PARAMS = dict(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True,
              C=1.0, class_weight="balanced", max_iter=3000)


def features(df):
    return [build_input(c, s) for c, s in zip(df["citation_context"], df["rhetorical_section"])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", default=REPO_ROOT / "models" / "scif-v4-tfidf-logreg", type=Path)
    ap.add_argument("--metrics", default=REPO_ROOT / "docs" / "modelos_v4" / "tfidf_metricas.json", type=Path)
    args = ap.parse_args()

    import cloudpickle
    import pandas as pd
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    data = {s: pd.read_csv(args.data_dir / f"{s}_enriquecido_top3_chunks.csv")
            for s in ("train", "val", "test")}

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=PARAMS["ngram_range"], min_df=PARAMS["min_df"],
                                  max_features=PARAMS["max_features"], sublinear_tf=PARAMS["sublinear_tf"])),
        ("clf", LogisticRegression(C=PARAMS["C"], class_weight=PARAMS["class_weight"],
                                   max_iter=PARAMS["max_iter"])),
    ])
    t = time.perf_counter()
    pipe.fit(features(data["train"]), data["train"]["label"])
    fit_s = time.perf_counter() - t

    metrics = {
        "modelo": "TF-IDF + regresión logística (v4)",
        "entrada": "contexto [SEC] sección normalizada (sin pasajes)",
        "hiperparametros": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PARAMS.items()},
        "sklearn": sklearn.__version__,
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segundos_entrenamiento": round(fit_s, 1),
        "particiones": {},
    }
    for s in ("train", "val", "test"):
        pred = list(pipe.predict(features(data[s])))
        metrics["particiones"][s] = report(list(data[s]["label"]), pred, LABELS)
        print(f"{s:5s} n={len(pred):5d}  F1-macro={metrics['particiones'][s]['f1_macro']:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "model.pkl", "wb") as f:
        cloudpickle.dump(pipe, f)
    (args.out / "metadata.json").write_text(json.dumps(
        {k: metrics[k] for k in ("modelo", "entrada", "hiperparametros", "sklearn", "fecha_utc")},
        indent=2, ensure_ascii=False) + "\n")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    size_mb = (args.out / "model.pkl").stat().st_size / 1e6
    print(f"-> {args.out} ({size_mb:.1f} MB) | métricas -> {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
