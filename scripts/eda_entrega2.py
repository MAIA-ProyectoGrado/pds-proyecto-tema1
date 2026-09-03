"""
EDA - Entrega 2 (SCIF)
======================

Analisis exploratorio del dataset enriquecido de intencion de cita, orientado
a la pregunta de negocio y a responder el feedback de la Entrega 1:

  * "no se explora la variable objetivo (base-label) segun lo que definieron
     ni su balance entre las nueve categorias"      -> Seccion 2
  * "la propia exploracion demuestra que el archivo no contiene contextos de
     cita reales"                                    -> Seccion 3 y 4
  * "la maqueta asume datos que no se mostraron consistentes en la
     exploracion de variables"                      -> Seccion 4 (Top-3 chunks)

Uso:
    python scripts/eda_entrega2.py --data_dir data/raw --out_dir docs/eda_entrega2

Genera:
    docs/eda_entrega2/*.png           figuras
    docs/eda_entrega2/eda_stats.json  metricas para citar en el reporte
"""
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CANON_LABELS = [
    "Background", "Gap", "Basis", "Comparison", "Application",
    "Modification_Improvement", "Evidence",
    "Identification_of_the_Originator", "Further_Reading",
]

CITATION_MARKER = re.compile(
    r"\[\d+(?:\s*[,-]\s*\d+)*\]"                       # [12], [3, 4], [5-7]
    r"|\((?:[A-Z][A-Za-z\-]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][A-Za-z\-]+)*),?\s+\d{4}[a-z]?\)"  # (Smith et al., 2020)
    r"|\b[A-Z][A-Za-z\-]+\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z\-]+)\s*\(?\d{4}[a-z]?\)?"           # Smith et al. (2020)
)


def load_splits(data_dir: Path):
    files = {
        "train": data_dir / "citation_intent_train_enriched.csv",
        "val": data_dir / "citation_intent_val_enriched.csv",
        "test": data_dir / "test_para_anotacion_humana_enriched.csv",
        "master_18k": data_dir / "dataset_18k_balanceado_enriched.csv",
    }
    return {k: pd.read_csv(v) for k, v in files.items() if v.exists()}


def entropy_balance(counts):
    total = sum(counts)
    probs = [c / total for c in counts if c > 0]
    h = -sum(p * math.log(p, 2) for p in probs)
    return h / math.log(len(probs), 2)  # normalizado 0..1 (1 = perfectamente balanceado)


def json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
    except ImportError:
        pass
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--out_dir", default="docs/eda_entrega2")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits = load_splits(data_dir)
    stats = {}

    # ------------------------------------------------------------------
    # 1. Estructura e integridad
    # ------------------------------------------------------------------
    stats["estructura"] = {}
    for name, df in splits.items():
        stats["estructura"][name] = {
            "filas": len(df),
            "columnas": list(df.columns),
            "nulos_por_columna": df.isna().sum().to_dict(),
            "duplicados_fila": int(df.duplicated().sum()),
            "duplicados_citation_context": int(df["citation_context"].duplicated().sum()),
        }

    tr, va = splits["train"], splits["val"]
    te = splits.get("test")

    # label == etiqueta_rescatada ?
    stats["label_vs_etiqueta_rescatada_iguales"] = bool(
        (tr["label"] == tr["etiqueta_rescatada"]).all()
        and (va["label"] == va["etiqueta_rescatada"]).all()
    )

    # ------------------------------------------------------------------
    # 2. VARIABLE OBJETIVO  (feedback principal)
    # ------------------------------------------------------------------
    target = {}
    for name in ["train", "val", "test", "master_18k"]:
        if name not in splits:
            continue
        vc = splits[name]["label"].value_counts()
        counts = [int(vc.get(l, 0)) for l in CANON_LABELS]
        target[name] = {
            "conteo": dict(zip(CANON_LABELS, counts)),
            "proporcion": {l: round(c / sum(counts), 4) for l, c in zip(CANON_LABELS, counts)},
            "min_clase": int(min(counts)),
            "max_clase": int(max(counts)),
            "ratio_desbalance": round(max(counts) / min(counts), 3),
            "entropia_normalizada": round(entropy_balance(counts), 4),
        }
    stats["variable_objetivo"] = target

    # figura: distribucion de la variable objetivo por particion
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.27
    xs = range(len(CANON_LABELS))
    for i, name in enumerate(["train", "val", "test"]):
        if name not in splits:
            continue
        vc = splits[name]["label"].value_counts()
        vals = [vc.get(l, 0) for l in CANON_LABELS]
        ax.bar([x + i * width for x in xs], vals, width, label=name)
    ax.set_xticks([x + width for x in xs])
    ax.set_xticklabels([l.replace("_", "\n") for l in CANON_LABELS], fontsize=8)
    ax.set_ylabel("n ejemplos")
    ax.set_title("Variable objetivo (label): distribucion por particion - 9 funciones de cita")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_variable_objetivo.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. citation_context: son citas reales?
    # ------------------------------------------------------------------
    ctx = {}
    for name in ["train", "val", "test"]:
        if name not in splits:
            continue
        df = splits[name].copy()
        df["wc"] = df["citation_context"].str.split().str.len()
        df["has_marker"] = df["citation_context"].str.contains(CITATION_MARKER)
        ctx[name] = {
            "palabras_media": round(float(df["wc"].mean()), 2),
            "palabras_mediana": float(df["wc"].median()),
            "palabras_p05": float(df["wc"].quantile(0.05)),
            "palabras_p95": float(df["wc"].quantile(0.95)),
            "palabras_min": int(df["wc"].min()),
            "palabras_max": int(df["wc"].max()),
            "pct_con_marcador_de_cita": round(float(df["has_marker"].mean()) * 100, 1),
            "pct_muy_corto_<=8_palabras": round(float((df["wc"] <= 8).mean()) * 100, 1),
        }
        if name == "train":
            df["wc_bin"] = df["wc"].clip(upper=120)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(df["wc_bin"], bins=40, color="#3b7dd8")
            ax.axvline(df["wc"].median(), color="red", ls="--",
                       label=f"mediana {df['wc'].median():.0f} palabras")
            ax.set_xlabel("longitud de citation_context (palabras, recortado a 120)")
            ax.set_ylabel("n ejemplos")
            ax.set_title("Longitud del contexto de cita (train). Entrega 1: mediana 163 (eran abstracts)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "02_longitud_contexto.png", dpi=120)
            plt.close(fig)
            # longitud por clase
            fig, ax = plt.subplots(figsize=(10, 4))
            order = df.groupby("label")["wc"].median().sort_values().index
            ax.boxplot([df.loc[df.label == l, "wc"].clip(upper=120) for l in order],
                       tick_labels=[l.replace("_", "\n") for l in order], showfliers=False)
            ax.set_ylabel("palabras")
            ax.set_title("Longitud del contexto por funcion de cita (train)")
            plt.setp(ax.get_xticklabels(), fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "03_longitud_por_clase.png", dpi=120)
            plt.close(fig)
    stats["citation_context"] = ctx

    # ------------------------------------------------------------------
    # 4. cited_paper_id y Top-3 chunks: consistencia (feedback maqueta)
    # ------------------------------------------------------------------
    def strip_prefix(s, p):
        return s.str[len(p):] if s.str.startswith(p).all() else s

    citing_eq_cited = float(
        (tr["citing_paper_id"].str.replace("citing_", "", regex=False)
         == tr["cited_paper_id"].str.replace("cited_", "", regex=False)).mean()
    )

    def parse_json_field(x, key):
        try:
            return json.loads(x).get(key)
        except Exception:
            return None

    tr_top1_txt = tr["top1_cited_chunk"].map(lambda x: parse_json_field(x, "text"))
    tr_top2_txt = tr["top2_cited_chunk"].map(lambda x: parse_json_field(x, "text"))
    tr_top3_txt = tr["top3_cited_chunk"].map(lambda x: parse_json_field(x, "text"))
    tr_top1_sim = tr["top1_cited_chunk"].map(lambda x: parse_json_field(x, "similarity_score"))
    tr_top2_sim = tr["top2_cited_chunk"].map(lambda x: parse_json_field(x, "similarity_score"))
    tr_top3_sim = tr["top3_cited_chunk"].map(lambda x: parse_json_field(x, "similarity_score"))

    stats["retrieval_top3"] = {
        "citing_id_==_cited_id_frac": round(citing_eq_cited, 4),
        "top1_text_==_citation_context_frac": round(float((tr_top1_txt == tr["citation_context"]).mean()), 4),
        "top2_textos_distintos": int(tr_top2_txt.nunique()),
        "top3_textos_distintos": int(tr_top3_txt.nunique()),
        "n_filas": len(tr),
        "top1_sim_unicos": sorted(set(round(float(s), 3) for s in tr_top1_sim.dropna()))[:5],
        "top2_sim_unicos": sorted(set(round(float(s), 3) for s in tr_top2_sim.dropna()))[:5],
        "top3_sim_unicos": sorted(set(round(float(s), 3) for s in tr_top3_sim.dropna()))[:5],
        "top2_valor_mas_comun": str(tr_top2_txt.value_counts().index[0])[:120],
        "top3_valor_mas_comun": str(tr_top3_txt.value_counts().index[0])[:120],
        "diagnostico": (
            "citing_paper_id y cited_paper_id son el MISMO identificador; top1 es "
            "una copia del citation_context; top2/top3 son plantillas de texto con "
            "similitud constante. El componente de 'recomendacion local de citas' "
            "NO tiene datos reales en esta version -> se modela solo la clasificacion "
            "de funcion de cita y se documenta el retrieval como trabajo futuro (v3)."
        ),
    }

    # ------------------------------------------------------------------
    # 5. Fuga de informacion entre particiones (No-Leakage Guarantee)
    # ------------------------------------------------------------------
    s_tr = set(tr["citing_paper_id"]); s_va = set(va["citing_paper_id"])
    s_te = set(te["citing_paper_id"]) if te is not None else set()
    stats["no_leakage"] = {
        "papers_train": len(s_tr), "papers_val": len(s_va), "papers_test": len(s_te),
        "overlap_train_val": len(s_tr & s_va),
        "overlap_train_test": len(s_tr & s_te),
        "overlap_val_test": len(s_va & s_te),
        "contextos_duplicados_train_val": int(tr["citation_context"].isin(set(va["citation_context"])).sum()),
    }

    # ------------------------------------------------------------------
    # 6. rhetorical_section: calidad y normalizacion
    # ------------------------------------------------------------------
    def canon_section(s):
        if not isinstance(s, str) or not s.strip():
            return "Desconocida"
        s2 = s.strip().lower()
        for key, lab in [
            ("abstract", "Abstract"), ("introduction", "Introduccion"),
            ("related work", "Trabajo relacionado"), ("background", "Trabajo relacionado"),
            ("method", "Metodos"), ("approach", "Metodos"),
            ("experiment", "Experimentos"), ("result", "Resultados"),
            ("discussion", "Discusion"), ("conclusion", "Conclusion"),
            ("scientific body", "Cuerpo (sin seccion fina)"),
        ]:
            if key in s2:
                return lab
        return "Otra"

    sec = {}
    for name in ["train", "val", "test"]:
        if name not in splits:
            continue
        df = splits[name]
        canon = df["rhetorical_section"].map(canon_section)
        sec[name] = {
            "valores_unicos_crudos": int(df["rhetorical_section"].nunique(dropna=True)),
            "pct_nulos": round(float(df["rhetorical_section"].isna().mean()) * 100, 2),
            "distribucion_canonica": canon.value_counts().to_dict(),
        }
    stats["rhetorical_section"] = sec

    # seccion canonica x label (train)
    canon_tr = tr["rhetorical_section"].map(canon_section)
    xt = pd.crosstab(canon_tr, tr["label"], normalize="index").round(3)
    xt.to_csv(out / "seccion_x_label_train.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(xt.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xt.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in xt.columns], fontsize=7)
    ax.set_yticks(range(len(xt.index)))
    ax.set_yticklabels(xt.index, fontsize=8)
    ax.set_title("P(funcion de cita | seccion retorica canonica) - train")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "04_seccion_x_label.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7. Diversidad tematica del corpus (feedback: aterrizar alcance)
    # ------------------------------------------------------------------
    from collections import Counter
    STOP = set("the a an of to in and for is are on with we our this that by as be it "
               "from at using used use can more most our their its than then also both "
               "which these those was were has have had not but or if into such each".split())
    words = Counter()
    for t in tr["citation_context"].str.lower().str.findall(r"[a-z]{3,}"):
        words.update(w for w in t if w not in STOP)
    stats["top_terminos_train"] = words.most_common(25)

    with open(out / "eda_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=json_default)

    print("EDA OK ->", out)
    print(json.dumps(stats["variable_objetivo"]["train"], indent=2, ensure_ascii=False))
    print(json.dumps(stats["citation_context"], indent=2, ensure_ascii=False))
    print(json.dumps(stats["retrieval_top3"], indent=2, ensure_ascii=False, default=json_default))
    print(json.dumps(stats["no_leakage"], indent=2, ensure_ascii=False))
    print(json.dumps(stats["rhetorical_section"]["train"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
