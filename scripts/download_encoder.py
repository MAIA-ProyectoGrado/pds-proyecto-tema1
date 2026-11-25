"""Descarga el codificador de oraciones del modelo v4 y lo deja listo para uso sin conexión.

La imagen de Docker de la API trabaja sin acceso a Hugging Face (HF_HUB_OFFLINE=1),
así que el codificador se guarda en `models/` y se versiona con DVC como el resto
de los artefactos.

Uso:
    python scripts/download_encoder.py --out models/minilm-l6-v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.passages import ENCODER_NAME, MAX_SEQ_LENGTH  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=REPO_ROOT / "models" / "minilm-l6-v2", type=Path)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    AutoModel.from_pretrained(ENCODER_NAME).save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(ENCODER_NAME).save_pretrained(args.out)
    (args.out / "pooling.json").write_text(json.dumps({
        "source": ENCODER_NAME,
        "pooling": "mean",
        "normalize": True,
        "max_seq_length": MAX_SEQ_LENGTH,
    }, indent=2) + "\n")
    print(f"OK -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
