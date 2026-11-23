"""Pruebas de `scripts/evaluate_test_humano.py` con un test sintético (sin el dataset)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from scripts.evaluate_test_humano import agreement, evaluate, subsets  # noqa: E402

TEST = pd.DataFrame({
    "source_dataset": ["scitail", "scicite", "scicite", "acl_arc"],
    "certeza_evaluacion": ["ALTA_CERTEZA", "ALTA_CERTEZA", "DUDA", "DUDA"],
    "top1_cited_chunk": ["premisa", None, "oración del abstract", None],
    "label": ["Background", "Gap", "Basis", "Evidence"],
    "human_label": ["Background", "Gap", "Evidence", "Basis"],
})


def test_subsets_split_by_certainty_source_and_passages():
    groups = subsets(TEST)
    assert groups["alta_certeza"] == [0, 1] and groups["duda"] == [2, 3]
    assert groups["solo_scitail"] == [0] and groups["sin_scitail"] == [1, 2, 3]
    assert groups["con_pasajes"] == [0, 2] and groups["sin_pasajes"] == [1, 3]


def test_agreement_counts_matches_overall_and_by_certainty():
    res = agreement(TEST)
    assert res["n"] == 4 and res["coincidencia"] == 0.5
    assert res["por_certeza"]["alta_certeza"]["coincidencia"] == 1.0
    assert res["por_certeza"]["duda"]["coincidencia"] == 0.0


def test_each_reference_is_scored_separately():
    pred = ["Background", "Gap", "Basis", "Evidence"]          # igual a `label`
    res = evaluate(TEST, pred, subsets(TEST))
    assert res["label"]["f1_micro"] == 1.0
    assert res["human_label"]["f1_micro"] == 0.5
    assert res["human_label"]["subconjuntos"]["duda"]["f1_micro"] == 0.0
