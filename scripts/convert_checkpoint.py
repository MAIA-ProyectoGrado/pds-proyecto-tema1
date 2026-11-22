"""Convierte el checkpoint del clasificador final a una carpeta estándar de Hugging Face.

El entrenamiento (notebooks/Entrenamiento_clasificadores_scibert_mlflow.ipynb) guarda
solo el `state_dict` del mejor modelo en un archivo .pt. Este script reconstruye la
arquitectura, carga esos pesos y deja una carpeta lista para servir:

    models/scif-scibert-chunks/
        config.json, model.safetensors     arquitectura y pesos
        tokenizer.json, vocab.txt, ...     tokenizer de SciBERT
        label_order.json                   orden de las 9 clases
        input_format.json                  cómo se construye la entrada (configuración B)

Uso:
    python scripts/convert_checkpoint.py \
        --checkpoint Run2_SciBERT_ConfigB_Top3Chunks_best_model.pt \
        --out models/scif-scibert-chunks

El .pt se carga con `weights_only=True`: solo se leen tensores, nunca se ejecuta
código contenido en el archivo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE_MODEL = "allenai/scibert_scivocab_uncased"
LABELS = sorted([
    "Application", "Background", "Basis", "Comparison", "Evidence",
    "Further_Reading", "Gap", "Identification_of_the_Originator",
    "Modification_Improvement",
])

# Formato de entrada de la configuración B, tal como lo construye
# `construir_entrada_config_b` en el notebook de entrenamiento.
INPUT_FORMAT = {
    "version": "v4-config-b",
    "base_model": BASE_MODEL,
    "input": "pair",
    "max_len": 512,
    "truncation": "longest_first",
    # El entrenamiento le pasa al modelo solo estas dos entradas, sin token_type_ids.
    "model_inputs": ["input_ids", "attention_mask"],
    "text_a": "citation_context",
    "text_b": {
        "parts_separator": " | ",
        "section": "Section: {section}",
        "section_rule": "se omite si la sección está vacía",
        "passages": "Cited passages: {passages}",
        "passages_join": " ",
        "fallback_title": "Cited work: {title}",
        "fallback_none": "Passages unavailable",
    },
    "passages": {
        "k": 3,
        "unit": "oración",
        "source_priority": ["abstract del artículo citado", "título"],
        "sentence_split_regex": r"(?<=[.!?])\s+",
        "min_sentence_chars": 13,
        "dedupe": "sin distinguir mayúsculas",
        "end_punctuation": "se agrega un punto si la oración no termina en . ! ?",
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity": "coseno",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, type=Path, help="archivo .pt con el state_dict")
    ap.add_argument("--out", required=True, type=Path, help="carpeta de salida")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise SystemExit(f"{args.checkpoint}: no contiene un state_dict")

    config = AutoConfig.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)},
    )
    model = AutoModelForSequenceClassification.from_config(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Los buffers no persistentes (p. ej. position_ids) pueden variar entre versiones.
    missing = [k for k in missing if not k.endswith("position_ids")]
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if missing or unexpected:
        raise SystemExit(f"Los pesos no coinciden con la arquitectura.\n"
                         f"  faltan: {missing[:10]}\n  sobran: {unexpected[:10]}")
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(args.out)
    (args.out / "label_order.json").write_text(json.dumps(LABELS, indent=2) + "\n")
    fmt = dict(INPUT_FORMAT, source_checkpoint={"file": args.checkpoint.name,
                                               "sha256": sha256(args.checkpoint)})
    (args.out / "input_format.json").write_text(json.dumps(fmt, indent=2, ensure_ascii=False) + "\n")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"OK: {len(state)} tensores cargados, {n_params / 1e6:.1f} M parámetros -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
