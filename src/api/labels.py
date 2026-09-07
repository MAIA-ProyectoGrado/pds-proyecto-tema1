"""Etiquetas de función de cita: nombre canónico ↔ nombre para mostrar.

El nombre **canónico** es el que usó el entrenamiento y el que devuelven los
artefactos (`models/scif-scibert/label_order.json`). Es la clave con la que viaja
todo por la API. El nombre *display* existe solo para la interfaz.

Fuente de verdad del orden: `models/<modelo>/label_order.json`. La lista de abajo
se usa como respaldo cuando el artefacto no la trae.
"""
from __future__ import annotations

# Orden alfabético usado en entrenamiento (label_order.json / dataset_manifest.json)
CANONICAL_LABELS: list[str] = [
    "Application",
    "Background",
    "Basis",
    "Comparison",
    "Evidence",
    "Further_Reading",
    "Gap",
    "Identification_of_the_Originator",
    "Modification_Improvement",
]

DISPLAY_NAMES: dict[str, str] = {
    "Application": "Application",
    "Background": "Background",
    "Basis": "Basis",
    "Comparison": "Comparison",
    "Evidence": "Evidence",
    "Further_Reading": "Further Reading",
    "Gap": "Gap",
    "Identification_of_the_Originator": "Identification of the Originator",
    "Modification_Improvement": "Modification / Improvement",
}

# Secciones retóricas canónicas que reconoce `models.scif_predict.canon_section`.
SECTION_OPTIONS: list[str] = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Method",
    "Experiments",
    "Results",
    "Discussion",
    "Conclusion",
    "Other",
]


def display_name(label: str) -> str:
    """Nombre legible de una etiqueta canónica; si no se conoce, se devuelve tal cual."""
    return DISPLAY_NAMES.get(label, label)
