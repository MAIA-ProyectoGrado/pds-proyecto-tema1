"""Etiquetas de función de cita: nombre canónico - nombre para mostrar.

"""
from __future__ import annotations

# Orden alfabético usado en entrenamiento (label_order.json del modelo)
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

# Secciones que ofrece el tablero. Son valores que aparecen en el entrenamiento
# (SciBERT es "uncased": "Methods" y "METHODS" le llegan igual). "Abstract" no
# aparece nunca en los datos de entrenamiento, por eso no se ofrece. Si la sección
# no se conoce, se envía null y el modelo recibe la entrada sin sección, como las
# citas de entrenamiento que no la tenían. Por la API se puede enviar cualquier
# texto: se pasa tal cual al modelo, igual que en el entrenamiento.
SECTION_OPTIONS: list[str] = [
    "Introduction",
    "Related Work",
    "Methods",
    "Experiments",
    "Results",
    "Discussion",
    "Conclusion",
]


def display_name(label: str) -> str:
    """Nombre legible de una etiqueta canónica; si no se conoce, se devuelve tal cual."""
    return DISPLAY_NAMES.get(label, label)
