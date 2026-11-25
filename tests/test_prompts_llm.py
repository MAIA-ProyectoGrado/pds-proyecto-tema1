"""Pruebas de `src/api/prompts_llm.py`: plantilla, enfoques y lectura de la lista de puntajes."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.scif_predict import LABELS  # noqa: E402
from src.api.prompts_llm import (ENFOQUES, ORDEN, entrada_cita, instrucciones, interpretar,  # noqa: E402
                                 interpretar_ajustado, lista)


def test_orden_tiene_las_9_funciones():
    assert sorted(ORDEN) == sorted(LABELS) and len(ORDEN) == 9


def test_entrada_con_campos_faltantes():
    texto = entrada_cita({"citation_context": "We use X [1].", "rhetorical_section": float("nan"),
                          "cited_paper_title": None, "cited_paper_abstract": "  An abstract.  "})
    assert texto == ('Citation context: "We use X [1]."\nSection: not available\n'
                     "Cited paper title: not available\nCited paper abstract: An abstract.")


@pytest.mark.parametrize("enfoque, criterios, ejemplos", [
    ("E1", False, False), ("E2", True, False), ("E3", False, True), ("E4", True, True)])
def test_bloques_por_enfoque(enfoque, criterios, ejemplos):
    texto = instrucciones(enfoque)
    assert ("Definitions and criteria" in texto) is criterios
    assert ("Solved examples" in texto) is ejemplos
    assert ", ".join(ORDEN) in texto and texto.rstrip().endswith("Do not add any other text.")


def test_los_ejemplos_cubren_las_9_funciones_una_vez():
    texto = instrucciones("E3")
    assert texto.count("\nAnswer: [") == 9
    for f in ORDEN:
        assert f"Answer: {lista(f)}" in texto


def test_enfoques_definidos():
    assert list(ENFOQUES) == ["E1", "E2", "E3", "E4"]


@pytest.mark.parametrize("texto, estado, etiqueta, empate", [
    ("[0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]", "valido", "Gap", False),
    ("```\n[0, 0, 0, 0, 1, 0, 0, 0, 0]\n```", "valido", "Application", False),
    ("[0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]", "valido", "Background", True),
    ("Background", "sin_lista", None, False),
    ("[0.1, 0.2, high, 0, 0, 0, 0, 0, 0]", "no_numerico", None, False),
    ("[0.1, 0.2, 0.3]", "cantidad", None, False),
    ("[0.1, 0.2, 1.5, 0, 0, 0, 0, 0, 0]", "rango", None, False),
])
def test_interpretar(texto, estado, etiqueta, empate):
    r = interpretar(texto)
    assert (r["estado"], r["etiqueta"], r["empate"]) == (estado, etiqueta, empate)


def test_respuesta_cortada():
    assert interpretar("[0.1, 0.2, 0.3", cortada=True)["estado"] == "cortada"
    assert interpretar("[0.1, 0.2]", cortada=True)["estado"] == "cortada"


@pytest.mark.parametrize("texto,cortada,estado,etiqueta,ajuste", [
    ("[0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]", False, "valido", "Gap", None),
    ("0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1", False, "valido", "Identification_of_the_Originator", "sin_corchetes"),
    ("[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, **1.0**, 0.0, 0.0]", False, "valido", "Evidence", "markdown"),
    ("", True, "cortada", None, "no_recuperable"),
    ("[0.", True, "cortada", None, "no_recuperable"),
    ("0.1, 0.2, 0.3", False, "sin_lista", None, "no_recuperable"),
])
def test_interpretar_ajustado(texto, cortada, estado, etiqueta, ajuste):
    r = interpretar_ajustado(texto, cortada)
    assert (r["estado"], r["etiqueta"], r["ajuste"]) == (estado, etiqueta, ajuste)
