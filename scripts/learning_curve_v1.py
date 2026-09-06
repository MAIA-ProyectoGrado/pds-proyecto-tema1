"""
Curva de aprendizaje y validacion cruzada del baseline v1 (TF-IDF + LogReg).

Responde dos preguntas:
  1. ¿La validacion mejora al añadir mas datos de entrenamiento?  -> curva de
     aprendizaje (train vs val a 500, 1k, 2k, 4k, 8k, 12.6k ejemplos).
  2. ¿Que tan estable es la metrica de validacion?  -> 5-fold GroupKFold por
     `citing_paper_id` (respeta el aislamiento por documento).

Genera:
  docs/modelos_entrega2/v1_learning_curve.png
  docs/modelos_entrega2/v1_cv.json

Uso:
    python scripts/learning_curve_v1.py --data_dir dataset --out docs/modelos_entrega2
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


def load(data_dir: Path):
    def rd(cands):
        p = next(data_dir / c for c in cands if (data_dir / c).exists())
        df = pd.read_csv(p)
        sec = "rhetorical_section_canon" if "rhetorical_section_canon" in df.columns else "rhetorical_section"
        df["_text"] = df["citation_context"].astype(str) + " [SEC] " + df[sec].fillna("NA").astype(str)
        return df
    tr = rd(["train.csv", "citation_intent_train_enriched.csv"])
    va = rd(["val.csv", "citation_intent_val_enriched.csv"])
    return tr, va


def pipe():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--out", default="docs/modelos_entrega2")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tr, va = load(Path(args.data_dir))

    # ---- 1. curva de aprendizaje (tamaño de train) --------------------------
    sizes = [500, 1000, 2000, 4000, 8000, len(tr)]
    rng = np.random.default_rng(42)
    curve = []
    for n in sizes:
        idx = rng.permutation(len(tr))[:n]
        sub = tr.iloc[idx]
        p = pipe().fit(sub["_text"], sub["label"])
        f1_tr = f1_score(sub["label"], p.predict(sub["_text"]), average="macro")
        f1_va = f1_score(va["label"], p.predict(va["_text"]), average="macro")
        curve.append({"n_train": int(n), "f1_train": round(f1_tr, 4), "f1_val": round(f1_va, 4)})
        print(curve[-1])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([c["n_train"] for c in curve], [c["f1_train"] for c in curve], "o-", label="F1-macro train")
    ax.plot([c["n_train"] for c in curve], [c["f1_val"] for c in curve], "s-", label="F1-macro val")
    ax.set_xlabel("nº de ejemplos de entrenamiento")
    ax.set_ylabel("F1-macro"); ax.set_ylim(0, 1)
    ax.set_title("v1 (TF-IDF + LogReg) — curva de aprendizaje")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "v1_learning_curve.png", dpi=120); plt.close(fig)

    # ---- 2. GroupKFold CV por citing_paper_id ------------------------------
    alld = pd.concat([tr, va], ignore_index=True)
    groups = alld["citing_paper_id"] if "citing_paper_id" in alld.columns else alld.index
    gkf = GroupKFold(n_splits=5)
    scores = []
    for k, (i_tr, i_te) in enumerate(gkf.split(alld, alld["label"], groups)):
        p = pipe().fit(alld.iloc[i_tr]["_text"], alld.iloc[i_tr]["label"])
        s = f1_score(alld.iloc[i_te]["label"], p.predict(alld.iloc[i_te]["_text"]), average="macro")
        scores.append(round(float(s), 4)); print(f"fold {k}: F1-macro {s:.4f}")

    cv = {
        "learning_curve": curve,
        "groupkfold_por_citing_paper_id": {
            "n_splits": 5, "f1_macro_por_fold": scores,
            "media": round(float(np.mean(scores)), 4),
            "desv_std": round(float(np.std(scores)), 4),
            "ic95_aprox": [round(float(np.mean(scores) - 1.96 * np.std(scores) / np.sqrt(5)), 4),
                           round(float(np.mean(scores) + 1.96 * np.std(scores) / np.sqrt(5)), 4)],
        },
        "conclusion": (
            "La curva muestra si añadir datos ayuda; la brecha train-val que no "
            "cierra indica sobreajuste estructural del enfoque lexico. El CV por "
            "grupos da la variabilidad real de la metrica de validacion."
        ),
    }
    (out / "v1_cv.json").write_text(json.dumps(cv, indent=2, ensure_ascii=False))
    print(json.dumps(cv, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
