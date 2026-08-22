"""
Pruebas para la lógica de extracción de contextos de cita.

No requieren conexión a internet: solo validan la lógica de detección de
marcadores de cita y de recorte del contexto alrededor de cada marcador.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from extract_arxiv_data import extract_citation_contexts  # noqa: E402


def test_detects_numeric_citation_markers():
    text = "Este resultado ya se había reportado antes [12] en otro estudio."
    contexts = extract_citation_contexts("paper-1", text)
    assert len(contexts) == 1
    assert contexts[0]["citation_marker"] == "[12]"
    assert contexts[0]["paper_id"] == "paper-1"


def test_detects_multiple_numeric_markers_in_one_bracket():
    text = "Varios trabajos coinciden en este punto [3, 4, 5]."
    contexts = extract_citation_contexts("paper-2", text)
    assert len(contexts) == 1
    assert contexts[0]["citation_marker"] == "[3, 4, 5]"


def test_detects_author_year_citation_markers():
    text = "Este enfoque fue propuesto originalmente por (Smith et al., 2020)."
    contexts = extract_citation_contexts("paper-3", text)
    assert len(contexts) == 1
    assert "Smith et al., 2020" in contexts[0]["citation_marker"]


def test_no_false_positive_on_plain_text():
    text = "Este texto no contiene ninguna cita en su interior."
    contexts = extract_citation_contexts("paper-4", text)
    assert contexts == []


def test_context_window_includes_surrounding_text():
    text = "A" * 500 + " marcador [1] " + "B" * 500
    contexts = extract_citation_contexts("paper-5", text)
    assert len(contexts) == 1
    context = contexts[0]["citation_context"]
    assert "A" in context
    assert "B" in context
