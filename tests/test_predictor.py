"""Pruebas del cargador de modelos (`models/scif_predict.py`).

Comprueban que SciBERT recibe la entrada con las mismas piezas que en el
entrenamiento. Las que cargan los pesos se omiten si no están (vienen de DVC).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from convert_checkpoint import INPUT_FORMAT  # noqa: E402
from models.scif_predict import model_kwargs  # noqa: E402
from src.api import model_loader  # noqa: E402

TRAINING_INPUTS = ["input_ids", "attention_mask"]
SCIBERT_DIR = REPO_ROOT / "models" / "scif-scibert-chunks"
ENCODING = {"input_ids": 1, "attention_mask": 2, "token_type_ids": 3}


def test_model_receives_only_the_declared_inputs():
    assert model_kwargs(ENCODING, {"model_inputs": TRAINING_INPUTS}) == {
        "input_ids": 1, "attention_mask": 2}


def test_model_receives_everything_when_nothing_is_declared():
    assert model_kwargs(ENCODING, {}) == ENCODING


def test_converter_declares_the_training_inputs():
    assert INPUT_FORMAT["model_inputs"] == TRAINING_INPUTS


@pytest.mark.skipif(not (SCIBERT_DIR / "input_format.json").exists(),
                    reason="faltan los pesos de SciBERT (DVC)")
def test_served_model_uses_the_converter_format():
    served = json.loads((SCIBERT_DIR / "input_format.json").read_text())
    served.pop("source_checkpoint", None)
    assert served == INPUT_FORMAT


class _Spy:
    """Envuelve el modelo y guarda con qué entradas se llamó."""

    def __init__(self, model):
        self.model = model
        self.keys: set[str] = set()

    def __call__(self, **kwargs):
        self.keys = set(kwargs)
        return self.model(**kwargs)


@pytest.mark.skipif(not model_loader.is_available("scibert"),
                    reason="faltan los pesos de SciBERT (DVC)")
def test_scibert_does_not_receive_token_type_ids(monkeypatch):
    pytest.importorskip("torch")
    predictor = model_loader.get_predictor("scibert")
    spy = _Spy(predictor._model)
    monkeypatch.setattr(predictor, "_model", spy)
    predictor.predict("We fine-tune SciBERT [4].", "Methods", ["SciBERT is a BERT model."])
    assert spy.keys == set(TRAINING_INPUTS)
