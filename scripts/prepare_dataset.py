"""
Consolidacion y estandarizacion del dataset (Entrega 2).

Entrada  : dataset/*.csv  (4 archivos enriquecidos)
Salida   : data/raw/     -> archivo maestro + particiones estandarizadas (CSV + JSONL)
           data/processed/dataset_manifest.json  -> resumen y checks de integridad

Estandarizacion:
  * columnas canonicas: pair_id, citing_paper_id, cited_paper_id,
    citation_context, rhetorical_section, rhetorical_section_canon,
    label, label_id, split
  * label_id: entero 0..8 estable (orden alfabetico de las 9 clases)
  * rhetorical_section_canon: normaliza ~560 valores crudos a 10 secciones
  * verifica no-leakage por citing_paper_id entre train/val/test

Uso:
    python scripts/prepare_dataset.py --src dataset --out_raw data/raw --out_proc data/processed
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

LABELS = sorted([
    "Application", "Background", "Basis", "Comparison", "Evidence",
    "Further_Reading", "Gap", "Identification_of_the_Originator",
    "Modification_Improvement",
])
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

SRC_FILES = {
    "train": "citation_intent_train_enriched.csv",
    "val": "citation_intent_val_enriched.csv",
    "test": "test_para_anotacion_humana_enriched.csv",
}
# El "maestro" estandarizado = concatenación de train+val+test (equivale al archivo
# dataset_<N>_balanceado_enriched.csv, que es el pool antes de particionar).
KEEP =["pair_id", "citing_paper_id", "cited_paper_id", "citation_context",
        "rhetorical_section", "label"]
# Columnas opcionales de procedencia (presentes desde la v3 del dataset; ~1 % de filas)
OPTIONAL = ["source_dataset", "predicted_label"]


def canon_section(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return "Desconocida"
    s2 = s.strip().lower()
    table = [
        ("abstract", "Abstract"), ("introduction", "Introduccion"),
        ("related work", "Trabajo_relacionado"), ("background", "Trabajo_relacionado"),
        ("method", "Metodos"), ("approach", "Metodos"),
        ("experiment", "Experimentos"), ("result", "Resultados"),
        ("discussion", "Discusion"), ("conclusion", "Conclusion"),
        ("scientific body", "Cuerpo_generico"),
    ]
    for key, lab in table:
        if key in s2:
            return lab
    return "Otra"


def standardize(df: pd.DataFrame, split: str) -> pd.DataFrame:
    cols = KEEP + [c for c in OPTIONAL if c in df.columns]
    df = df[cols].copy()
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = None
    df["citation_context"] = df["citation_context"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["rhetorical_section"] = df["rhetorical_section"].where(df["rhetorical_section"].notna(), None)
    df["rhetorical_section_canon"] = df["rhetorical_section"].map(canon_section)
    df["label"] = df["label"].astype(str).str.strip()
    bad = set(df["label"]) - set(LABELS)
    if bad:
        raise ValueError(f"Etiquetas no reconocidas en {split}: {bad}")
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)
    # ¿el par citante/citado es real (no auto-referencia legacy)?
    a = df["citing_paper_id"].astype(str).str.replace("citing_", "", regex=False).str.replace("_legacy", "", regex=False)
    b = df["cited_paper_id"].astype(str).str.replace("cited_", "", regex=False).str.replace("_legacy", "", regex=False)
    df["par_citado_real"] = (a != b)
    df["split"] = split
    return df[["pair_id", "citing_paper_id", "cited_paper_id", "par_citado_real",
               "citation_context", "rhetorical_section", "rhetorical_section_canon",
               "label", "label_id", "source_dataset", "predicted_label", "split"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--out_raw", default="data/raw")
    ap.add_argument("--out_proc", default="data/processed")
    args = ap.parse_args()

    src = Path(args.src)
    out_raw = Path(args.out_raw); out_raw.mkdir(parents=True, exist_ok=True)
    out_proc = Path(args.out_proc); out_proc.mkdir(parents=True, exist_ok=True)

    parts = {}
    for split, fname in SRC_FILES.items():
        parts[split] = standardize(pd.read_csv(src / fname), split)

    master = pd.concat(parts.values(), ignore_index=True)
    master.to_csv(out_raw / "citation_intent_master.csv", index=False)

    for split, df in parts.items():
        df.to_csv(out_raw / f"{split}.csv", index=False)
        with open(out_raw / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                f.write(json.dumps({
                    "text": r["citation_context"],
                    "section": r["rhetorical_section_canon"],
                    "label": r["label"],
                    "label_id": int(r["label_id"]),
                    "pair_id": r["pair_id"],
                }, ensure_ascii=False) + "\n")

    # ---- checks de integridad -------------------------------------------------
    ids = {s: set(df["citing_paper_id"]) for s, df in parts.items()}
    manifest = {
        "labels": LABELS,
        "label2id": LABEL2ID,
        "n_por_split": {s: len(df) for s, df in parts.items()},
        "balance_por_split": {
            s: df["label"].value_counts().sort_index().to_dict() for s, df in parts.items()
        },
        "no_leakage_citing_paper_id": {
            "train_val": len(ids["train"] & ids["val"]),
            "train_test": len(ids["train"] & ids["test"]),
            "val_test": len(ids["val"] & ids["test"]),
        },
        "contexto_duplicado_entre_splits": int(
            master["citation_context"].duplicated().sum()
        ),
        "nulos_rhetorical_section": {
            s: int(df["rhetorical_section"].isna().sum()) for s, df in parts.items()
        },
        "par_citado_real_frac": {
            s: round(float(df["par_citado_real"].mean()), 4) for s, df in parts.items()
        },
        "filas_con_source_dataset": {
            s: int(df["source_dataset"].notna().sum()) for s, df in parts.items()
        },
        "md5_master": hashlib.md5(
            (out_raw / "citation_intent_master.csv").read_bytes()
        ).hexdigest(),
        "advertencias": [
            "En ~98 % de las filas citing_paper_id y cited_paper_id son el mismo "
            "identificador (auto-referencia); solo ~2 % (par_citado_real=True) "
            "tiene un documento citado distinto. Las columnas top{1,2,3}_cited_chunk "
            "del CSV original NO se propagan al dataset estandarizado.",
            "El etiquetado es asistido por LLM con rescate probabilistico; en las "
            "filas con predicted_label disponible, el rescate cambio la etiqueta "
            "del ~28 %. Hay ruido de etiqueta apreciable (ver EDA).",
        ],
    }
    with open(out_proc / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    leak = manifest["no_leakage_citing_paper_id"]
    total_leak = sum(leak.values())
    if total_leak > 0:
        print(f"\nAVISO: {total_leak} citing_paper_id solapados entre particiones "
              f"({leak}). Tolerable si es <0.1 % de la particion menor.")
    assert total_leak <= 5, f"FUGA de informacion no trivial entre particiones: {leak}"
    print("\nOK - particiones estandarizadas en", out_raw)


if __name__ == "__main__":
    main()
