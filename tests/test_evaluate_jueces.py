"""Pruebas de `scripts/evaluate_jueces.py` con un Bedrock simulado (sin red ni costo)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_jueces import (  # noqa: E402
    PROMPT_FILE, correr, fleiss_kappa, interpretar, leer_respuestas, mayoria, mensaje_usuario,
)


# ── interpretación de la respuesta ──────────────────────────────────────────

@pytest.mark.parametrize("texto, esperado", [
    ('```json\n{\n  "predicted_label": "Application"\n}\n```', ("Application", "json")),
    ('{"predicted_label": "Modification_Improvement"}', ("Modification_Improvement", "json")),
    ('{"predicted_label": "Identification of the Originator"}', ("Identification_of_the_Originator", "json")),
    ('The answer is Evidence.', ("Evidence", "texto")),
    ('Background or Basis', (None, "invalido")),
    ('{"predicted_label": "Method"}', (None, "invalido")),
    ('', (None, "invalido")),
])
def test_interpretar(texto, esperado):
    assert interpretar(texto) == esperado


def test_mensaje_usuario_replica_el_notebook():
    fila = {"citation_context": " We use BERT [4]. ", "rhetorical_section": "Methods",
            "cited_paper_title": "BERT: Pre-training", "cited_paper_abstract": "x" * 400}
    texto = mensaje_usuario(fila)
    assert texto.startswith('Rhetorical Section: Methods\nCitation Context: "We use BERT [4]."')
    assert '\nCited Paper Title: "BERT: Pre-training"' in texto
    assert texto.endswith('\nCited Paper Abstract: "' + "x" * 300 + '..."')


def test_mensaje_usuario_sin_titulo_ni_abstract():
    fila = {"citation_context": "Context.", "rhetorical_section": float("nan"),
            "cited_paper_title": float("nan"), "cited_paper_abstract": "corto"}
    assert mensaje_usuario(fila) == 'Rhetorical Section: nan\nCitation Context: "Context."'


def test_el_prompt_tiene_las_definiciones_del_notebook():
    nb = json.loads((REPO_ROOT / "notebooks" / "Obtener_dataset.ipynb").read_text())
    celda = next("".join(c["source"]) for c in nb["cells"] if "SYSTEM_PROMPT" in "".join(c["source"]))

    def definiciones(texto):
        return [" ".join(d.split()) for d in re.findall(r"(?:Definition|Criteria):(.*)", texto)]

    assert len(definiciones(celda)) == 18
    assert definiciones(PROMPT_FILE.read_text()) == definiciones(celda)


# ── acuerdo y mayoría ───────────────────────────────────────────────────────

def test_fleiss_acuerdo_perfecto_y_azar():
    assert fleiss_kappa([["A", "A", "A"], ["B", "B", "B"]]) == 1.0
    assert fleiss_kappa([["A", "B"], ["B", "A"]]) < 0


def test_fleiss_ejemplo_conocido():
    # 4 ítems, 3 evaluadores: P̄ = (1 + 1/3 + 1/3 + 1) / 4 = 2/3 y P_e = 1/2, así que kappa = 1/3.
    filas = [["A", "A", "A"], ["A", "A", "B"], ["A", "B", "B"], ["B", "B", "B"]]
    assert fleiss_kappa(filas) == pytest.approx(1 / 3)


def test_mayoria():
    assert mayoria(["Gap", "Gap", "Basis"]) == "Gap"
    assert mayoria(["Gap", "Basis", "Evidence"]) is None
    assert mayoria(["Gap", None, "Gap"]) == "Gap"
    assert mayoria([None, None, None]) is None


# ── llamadas, reintentos y reanudación ──────────────────────────────────────

@pytest.fixture
def bedrock():
    """Bedrock simulado; las pruebas que lo usan se omiten si falta botocore."""
    pytest.importorskip("botocore")
    return _Bedrock


class _Bedrock:
    """Responde siempre `etiqueta`; las primeras `fallos` llamadas lanzan `codigo`."""

    def __init__(self, etiqueta="Gap", fallos=0, codigo="ThrottlingException"):
        self.etiqueta, self.fallos, self.codigo, self.llamadas = etiqueta, fallos, codigo, 0

    def converse(self, **kw):
        self.llamadas += 1
        assert kw["inferenceConfig"] == {"temperature": 0.0, "maxTokens": 30}
        if self.llamadas <= self.fallos:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": self.codigo, "Message": "x"}}, "Converse")
        return {"output": {"message": {"content": [{"text": json.dumps({"predicted_label": self.etiqueta})}]}},
                "usage": {"inputTokens": 850, "outputTokens": 15}, "metrics": {"latencyMs": 300},
                "stopReason": "end_turn"}


def _datos(n=3):
    return [{"id": f"c{i}", "citation_context": f"Context {i}.", "rhetorical_section": "Methods",
             "cited_paper_title": None, "cited_paper_abstract": None} for i in range(n)]


def test_reintenta_errores_transitorios(tmp_path, monkeypatch, bedrock):
    monkeypatch.setattr("time.sleep", lambda s: None)
    cliente = bedrock(fallos=2)
    conteo = correr("gemma-3-4b", _datos(1), cliente, tmp_path / "r.jsonl", hilos=1)
    assert conteo == {"json": 1} and cliente.llamadas == 3
    assert leer_respuestas(tmp_path / "r.jsonl")["c0"]["intentos"] == 3


def test_no_reintenta_errores_permanentes(tmp_path, bedrock):
    cliente = bedrock(fallos=99, codigo="ValidationException")
    conteo = correr("gemma-3-4b", _datos(1), cliente, tmp_path / "r.jsonl", hilos=1)
    assert conteo == {"error": 1} and cliente.llamadas == 1


def test_reanuda_sin_repetir_lo_hecho(tmp_path, bedrock):
    salida = tmp_path / "r.jsonl"
    correr("gemma-3-4b", _datos(3), bedrock(), salida, hilos=2, limite=2)
    cliente = bedrock()
    correr("gemma-3-4b", _datos(3), cliente, salida, hilos=2)
    assert cliente.llamadas == 1
    assert set(leer_respuestas(salida)) == {"c0", "c1", "c2"}
