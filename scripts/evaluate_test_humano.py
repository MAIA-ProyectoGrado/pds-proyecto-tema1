"""Métricas del test frente a sus dos etiquetas de referencia.

Cada cita del test (2.241) trae dos etiquetas:
  - `label`: la del etiquetado automático (Llama 3.2 3B con rescate probabilístico),
    la misma con la que se entrenó y se eligió el checkpoint;
  - `human_label`: la revisión humana del test.

`certeza_evaluacion` es la certeza de la etiqueta automática: ALTA_CERTEZA si el
puntaje de Llama para `label` es de al menos 0,70 y DUDA si no.

Para SciBERT usa las predicciones de `evaluate_v4.py`
(docs/modelos_v4/scibert_chunks_predicciones_test.csv), así que no carga el modelo.
Para TF-IDF predice con models/scif-v4-tfidf-logreg, en segundos y en CPU.

Para cada modelo y cada referencia reporta las métricas de `evaluate_v4.report`
sobre el test completo y sobre estos subconjuntos: certeza alta, certeza en duda,
sin SciTail, solo SciTail, con pasajes y sin pasajes. Reporta también el acuerdo
entre las dos etiquetas (coincidencia y kappa de Cohen).

Uso:
    python scripts/evaluate_test_humano.py --data-dir dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.scif_predict import LABELS, _valid, build_input  # noqa: E402
from scripts.evaluate_v4 import report  # noqa: E402

REFERENCES = ("human_label", "label")
OUT_DIR = REPO_ROOT / "docs" / "modelos_v4"


def subsets(df) -> dict[str, list[int]]:
    """Índices de cada subconjunto del test."""
    scitail = df["source_dataset"].eq("scitail")
    passages = df["top1_cited_chunk"].map(_valid)
    certeza = df["certeza_evaluacion"]
    masks = {
        "alta_certeza": certeza.eq("ALTA_CERTEZA"),
        "duda": certeza.eq("DUDA"),
        "sin_scitail": ~scitail,
        "solo_scitail": scitail,
        "con_pasajes": passages,
        "sin_pasajes": ~passages,
    }
    return {name: list(df.index[mask]) for name, mask in masks.items()}


def evaluate(df, pred: list[str], groups: dict[str, list[int]]) -> dict:
    out = {}
    for ref in REFERENCES:
        y = df[ref].tolist()
        res = report(y, pred, LABELS)
        res["subconjuntos"] = {
            name: {k: v for k, v in report([y[i] for i in idx], [pred[i] for i in idx], LABELS).items()
                   if k != "matriz_confusion"}
            for name, idx in groups.items()
        }
        out[ref] = res
    return out


def agreement(df) -> dict:
    from sklearn.metrics import cohen_kappa_score

    def pair(d):
        return {"n": len(d),
                "coincidencia": round(float((d["label"] == d["human_label"]).mean()), 4),
                "kappa_cohen": round(float(cohen_kappa_score(d["label"], d["human_label"])), 4)}

    res = pair(df)
    res["por_certeza"] = {c.lower(): pair(df[df["certeza_evaluacion"] == c])
                          for c in ("ALTA_CERTEZA", "DUDA")}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--scibert-pred", default=OUT_DIR / "scibert_chunks_predicciones_test.csv", type=Path)
    ap.add_argument("--tfidf", default=REPO_ROOT / "models" / "scif-v4-tfidf-logreg", type=Path)
    ap.add_argument("--out", default=OUT_DIR / "test_referencias_metricas.json", type=Path)
    args = ap.parse_args()

    import cloudpickle
    import pandas as pd

    df = pd.read_csv(args.data_dir / "test_enriquecido_top3_chunks.csv")
    sb = pd.read_csv(args.scibert_pred).set_index("fila").sort_index()
    if len(sb) != len(df) or not (sb["label"].to_numpy() == df["label"].to_numpy()).all():
        raise SystemExit("Las predicciones de SciBERT no corresponden al test (orden o etiquetas distintas).")

    with open(args.tfidf / "model.pkl", "rb") as f:
        tfidf = cloudpickle.load(f)
    preds = {
        "scibert": sb["prediccion"].tolist(),
        "tfidf-logreg": list(tfidf.predict([build_input(c, s) for c, s in
                                            zip(df["citation_context"], df["rhetorical_section"])])),
    }

    groups = subsets(df)
    out = {
        "descripcion": ("Test de 2.241 citas evaluado contra la revisión humana (human_label) "
                        "y contra la etiqueta automática (label)."),
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tamanos_subconjuntos": {k: len(v) for k, v in groups.items()},
        "acuerdo_label_vs_human_label": agreement(df),
        "modelos": {name: evaluate(df, p, groups) for name, p in preds.items()},
    }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    cols = ["source_dataset", "certeza_evaluacion", "label", "human_label"]
    df[cols].assign(con_pasajes=df["top1_cited_chunk"].map(_valid),
                    pred_scibert=preds["scibert"], pred_tfidf=preds["tfidf-logreg"]).to_csv(
        args.out.parent / "predicciones_test.csv", index_label="fila")

    acc = out["acuerdo_label_vs_human_label"]
    print(f"label vs human_label: coincidencia {acc['coincidencia']:.3f}, kappa {acc['kappa_cohen']:.3f}")
    for name, res in out["modelos"].items():
        for ref in REFERENCES:
            r = res[ref]
            print(f"{name:13s} contra {ref:11s}: F1 macro {r['f1_macro']:.4f} | exactitud {r['f1_micro']:.4f}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
