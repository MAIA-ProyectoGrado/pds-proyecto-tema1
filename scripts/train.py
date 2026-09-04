"""
Entrenamiento y registro de modelos - Entrega 2 (SCIF)
=====================================================

Tarea modelada: clasificacion de la FUNCION DE CITA (9 clases) a partir de
`citation_context` (+ seccion retorica canonica).  El componente de
recomendacion local de citas (Top-3 chunks) NO se modela en esta entrega
porque el dataset no contiene documento citado real (ver docs/EDA_ENTREGA2.md).

Dos iteraciones:
  v1  baseline    : TF-IDF (1-2 gramas) + Regresion Logistica (multinomial)
  v2  optimizado  : fine-tuning de un encoder cientifico (SciBERT por defecto)
                    -- NO busca ser optimo todavia; deja margen para v3.

Todo se registra en MLflow (params, metricas train/val/test, matrices de
confusion, curvas, y el modelo como artефacto + Model Registry).

Uso:
    export MLFLOW_TRACKING_URI=http://<IP_EC2>:5000
    python scripts/train.py --stage v1 --data_dir data/raw
    python scripts/train.py --stage v2 --data_dir data/raw --model allenai/scibert_scivocab_uncased --epochs 3
    python scripts/train.py --stage all --data_dir data/raw
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

TMP = Path(tempfile.gettempdir())

LABELS = sorted([
    "Application", "Background", "Basis", "Comparison", "Evidence",
    "Further_Reading", "Gap", "Identification_of_the_Originator",
    "Modification_Improvement",
])
L2I = {l: i for i, l in enumerate(LABELS)}
EXPERIMENT = "scif-citation-intent"
REGISTERED_MODEL = "scif-citation-intent"


# ----------------------------------------------------------------------------
def load(data_dir: Path, max_train: int | None = None):
    d = {}
    for s in ["train", "val", "test"]:
        df = pd.read_csv(data_dir / f"{s}.csv")
        df["citation_context"] = df["citation_context"].astype(str)
        df["rhetorical_section_canon"] = df["rhetorical_section_canon"].fillna("Desconocida")
        if s == "train" and max_train and len(df) > max_train:
            # submuestreo estratificado (util para entrenar v2 en CPU de 2 vCPU)
            df = df.groupby("label", group_keys=False).apply(
                lambda g: g.sample(n=max(1, round(max_train * len(g) / len(df))),
                                   random_state=42)
            ).reset_index(drop=True)
        d[s] = df
    return d


def metric_block(y_true, y_pred, prefix):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average="macro"),
        f"{prefix}_f1_micro": f1_score(y_true, y_pred, average="micro"),
        f"{prefix}_precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def log_confusion(mlflow, y_true, y_pred, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=[l[:14] for l in LABELS]).plot(
        ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False, values_format=".2f")
    ax.set_title(name)
    fig.tight_layout()
    p = str(TMP / f"{name}.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    mlflow.log_artifact(p, "figures")


def per_class_f1(mlflow, y_true, y_pred, tag):
    from sklearn.metrics import classification_report
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    p = str(TMP / f"classification_report_{tag}.json")
    Path(p).write_text(json.dumps(rep, indent=2))
    mlflow.log_artifact(p, "reports")
    for l in LABELS:
        mlflow.log_metric(f"{tag}_f1_{l}", rep[l]["f1-score"])


def _log_model_safe(log_fn):
    """Registra el modelo; si el Model Registry falla o tarda, no aborta el run.
    SCIF_SKIP_MODEL_LOGGING=1 lo salta por completo (util para smoke tests locales)."""
    if os.environ.get("SCIF_SKIP_MODEL_LOGGING") == "1":
        print("[info] SCIF_SKIP_MODEL_LOGGING=1 -> no se registra el artefacto del modelo")
        return
    try:
        log_fn(registered_model_name=REGISTERED_MODEL)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] registro en Model Registry fallo ({e}); guardo el modelo sin registrar")
        try:
            log_fn()
        except Exception as e2:  # noqa: BLE001
            print(f"[warn] log_model tambien fallo ({e2}); se omite el artefacto del modelo")


def overfitting_summary(mlflow, m):
    gap_f1 = m["train_f1_macro"] - m["val_f1_macro"]
    gap_acc = m["train_accuracy"] - m["val_accuracy"]
    val_test_gap = m["val_f1_macro"] - m["test_f1_macro"]
    mlflow.log_metric("overfit_gap_f1_macro_train_val", gap_f1)
    mlflow.log_metric("overfit_gap_accuracy_train_val", gap_acc)
    mlflow.log_metric("generalization_gap_val_test_f1", val_test_gap)
    verdict = ("severo" if gap_f1 > 0.25 else "moderado" if gap_f1 > 0.12 else "bajo")
    mlflow.set_tag("overfitting", verdict)
    return {"gap_f1_macro_train_val": gap_f1, "gap_accuracy_train_val": gap_acc,
            "gap_val_test_f1": val_test_gap, "veredicto": verdict}


# ----------------------------------------------------------------------------
def train_v1(data, args, mlflow):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    def feats(df):
        return df["citation_context"] + " [SEC] " + df["rhetorical_section_canon"]

    with mlflow.start_run(run_name="v1_baseline_tfidf_logreg") as run:
        params = dict(ngram_range=(1, 2), min_df=2, max_features=50000, C=1.0,
                      class_weight="balanced", sublinear_tf=True)
        mlflow.log_params({f"tfidf_{k}": v for k, v in params.items()})
        mlflow.set_tags({"stage": "v1", "family": "linear", "task": "citation-function-classification"})

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=params["ngram_range"], min_df=params["min_df"],
                                      max_features=params["max_features"], sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=3000, C=params["C"],
                                       class_weight=params["class_weight"])),
        ])
        t = time.time()
        pipe.fit(feats(data["train"]), data["train"]["label"])
        mlflow.log_metric("fit_seconds", time.time() - t)

        m = {}
        for s in ["train", "val", "test"]:
            pred = pipe.predict(feats(data[s]))
            m.update(metric_block(data[s]["label"], pred, s))
            if s in ("val", "test"):
                log_confusion(mlflow, data[s]["label"], pred, f"cm_v1_{s}")
                per_class_f1(mlflow, data[s]["label"], pred, f"v1_{s}")
        mlflow.log_metrics(m)
        of = overfitting_summary(mlflow, m)
        mlflow.log_dict(of, "overfitting_v1.json")

        import mlflow.sklearn
        _log_model_safe(lambda **kw: mlflow.sklearn.log_model(
            pipe, "model", pip_requirements=["scikit-learn", "pandas"], **kw))
        print("v1", json.dumps({k: round(v, 4) for k, v in m.items()}, indent=2))
        print("v1 overfitting:", of)
        return run.info.run_id, m, of


# ----------------------------------------------------------------------------
def train_v2(data, args, mlflow):
    """Dispatcher. --v2_method embed_lr (por defecto, CPU-friendly) | finetune."""
    if args.v2_method == "embed_lr":
        return train_v2_embed_lr(data, args, mlflow)
    return train_v2_finetune(data, args, mlflow)


def _texts(df):
    return (df["citation_context"] + " [SEC] " + df["rhetorical_section_canon"]).tolist()


def train_v2_embed_lr(data, args, mlflow):
    """v2 = embeddings de un sentence-encoder (congelado) + Regresion Logistica.

    Representacion semantica (vs. la lexica de v1) sin el costo de un fine-tuning
    completo -> adecuado para la instancia de 2 vCPU del Learner Lab.
    """
    import numpy as np  # noqa: F811
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    enc = SentenceTransformer(args.st_model, device="cpu")
    with mlflow.start_run(run_name=f"v2_embed_lr_{args.st_model.split('/')[-1]}") as run:
        mlflow.set_tags({"stage": "v2", "family": "sentence-embeddings + linear",
                         "encoder": args.st_model, "encoder_frozen": "true",
                         "task": "citation-function-classification",
                         "nota": "iteracion intermedia, no optimizada; fine-tuning -> v3"})
        mlflow.log_params(dict(v2_method="embed_lr", st_model=args.st_model,
                               clf="LogisticRegression(C=1.0, class_weight=balanced)",
                               max_train=args.max_train or len(data["train"])))
        t = time.time()
        emb = {s: enc.encode(_texts(data[s]), batch_size=64, show_progress_bar=False,
                             normalize_embeddings=True) for s in ["train", "val", "test"]}
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(emb["train"], data["train"]["label"])
        mlflow.log_metric("fit_seconds", time.time() - t)

        m = {}
        for s in ["train", "val", "test"]:
            pred = clf.predict(emb[s])
            m.update(metric_block(data[s]["label"], pred, s))
            if s in ("val", "test"):
                log_confusion(mlflow, data[s]["label"], pred, f"cm_v2_{s}")
                per_class_f1(mlflow, data[s]["label"], pred, f"v2_{s}")
        mlflow.log_metrics(m)
        of = overfitting_summary(mlflow, m)
        mlflow.log_dict(of, "overfitting_v2.json")

        import mlflow.sklearn
        _log_model_safe(lambda **kw: mlflow.sklearn.log_model(
            clf, "model", pip_requirements=["scikit-learn", "sentence-transformers"], **kw))
        print("v2", json.dumps({k: round(float(v), 4) for k, v in m.items()}, indent=2))
        print("v2 overfitting:", of)
        return run.info.run_id, m, of


def train_v2_finetune(data, args, mlflow):
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments, DataCollatorWithPadding,
                              EarlyStoppingCallback)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)

    def to_ds(df):
        text = (df["citation_context"] + " [SEC] " + df["rhetorical_section_canon"]).tolist()
        ds = Dataset.from_dict({"text": text, "labels": df["label"].map(L2I).tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                      batched=True, remove_columns=["text"])

    ds = {s: to_ds(data[s]) for s in ["train", "val", "test"]}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)}, label2id=L2I)

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, preds),
                "f1_macro": f1_score(p.label_ids, preds, average="macro")}

    with mlflow.start_run(run_name=f"v2_{args.model.split('/')[-1]}") as run:
        mlflow.set_tags({"stage": "v2", "family": "transformer", "base_model": args.model,
                         "task": "citation-function-classification", "device": device,
                         "nota": "iteracion intermedia, no optimizada; v3 pendiente"})
        targs = TrainingArguments(
            output_dir=str(TMP / "v2_out"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=64,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=(device == "cuda"),
            report_to=[],
            seed=args.seed,
        )
        mlflow.log_params(dict(base_model=args.model, epochs=args.epochs, batch=args.batch,
                               lr=args.lr, max_len=args.max_len, weight_decay=0.01,
                               warmup_ratio=0.1, seed=args.seed))
        trainer = Trainer(
            model=model, args=targs,
            train_dataset=ds["train"], eval_dataset=ds["val"],
            tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        t = time.time()
        train_out = trainer.train()
        mlflow.log_metric("fit_seconds", time.time() - t)

        # curva de entrenamiento (loss train vs eval)
        hist = pd.DataFrame(trainer.state.log_history)
        hist.to_csv(TMP / "v2_log_history.csv", index=False)
        mlflow.log_artifact(str(TMP / "v2_log_history.csv"), "reports")

        m = {}
        for s in ["train", "val", "test"]:
            out = trainer.predict(ds[s])
            pred = out.predictions.argmax(-1)
            y_true = [LABELS[i] for i in out.label_ids]
            y_pred = [LABELS[i] for i in pred]
            m.update(metric_block(y_true, y_pred, s))
            m[f"{s}_loss"] = float(out.metrics.get("test_loss", np.nan))
            if s in ("val", "test"):
                log_confusion(mlflow, y_true, y_pred, f"cm_v2_{s}")
                per_class_f1(mlflow, y_true, y_pred, f"v2_{s}")
        mlflow.log_metrics({k: v for k, v in m.items() if not np.isnan(v)})
        of = overfitting_summary(mlflow, m)
        mlflow.log_dict(of, "overfitting_v2.json")

        import mlflow.transformers
        _log_model_safe(lambda **kw: mlflow.transformers.log_model(
            {"model": trainer.model, "tokenizer": tok}, "model",
            task="text-classification", **kw))
        print("v2", json.dumps({k: round(float(v), 4) for k, v in m.items() if not np.isnan(v)}, indent=2))
        print("v2 overfitting:", of)
        return run.info.run_id, m, of


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["v1", "v2", "all"], default="all")
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--v2_method", choices=["embed_lr", "finetune"], default="embed_lr")
    ap.add_argument("--st_model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="sentence-encoder para --v2_method embed_lr")
    ap.add_argument("--model", default="distilbert-base-uncased",
                    help="modelo base para --v2_method finetune")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_train", type=int, default=None,
                    help="submuestrea train a N filas (estratificado). Para v2 en CPU 2 vCPU: 4000-6000.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(EXPERIMENT)
    data = load(Path(args.data_dir), max_train=args.max_train)

    summary = {}
    if args.stage in ("v1", "all"):
        rid, m, of = train_v1(data, args, mlflow)
        summary["v1"] = {"run_id": rid, "metrics": m, "overfitting": of}
    if args.stage in ("v2", "all"):
        rid, m, of = train_v2(data, args, mlflow)
        summary["v2"] = {"run_id": rid, "metrics": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                                    for k, v in m.items()}, "overfitting": of}

    Path("docs").mkdir(exist_ok=True)
    Path("docs/model_results.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nResumen ->", "docs/model_results.json")


if __name__ == "__main__":
    main()
