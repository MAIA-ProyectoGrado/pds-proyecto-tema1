"""Prompts de la evaluación con modelos de lenguaje y lectura de sus respuestas.

Plantilla base del enunciado: contexto de la cita, título y resumen del documento
citado y la lista de las 9 funciones; el modelo devuelve un puntaje entre 0.0 y 1.0
por función, como lista `[0.0, …, 0.0]`, y la predicción es el puntaje más alto.
La plantilla agrega la sección retórica, que también recibe SciBERT B, para comparar
en igualdad de condiciones.

Cuatro enfoques, según los dos bloques opcionales del enunciado:

    E1  zero-shot genérico          sin criterios, sin ejemplos
    E2  zero-shot con criterios     con criterios, sin ejemplos
    E3  few-shot genérico           sin criterios, con ejemplos resueltos
    E4  few-shot con criterios      con criterios, con ejemplos resueltos

El mensaje de sistema (instrucciones y bloques) es fijo por enfoque y va primero, para
que los proveedores lo reutilicen en caché; la cita va en el mensaje del usuario.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EJEMPLOS_FILE = REPO_ROOT / "docs" / "evaluacion_llm" / "ejemplos_resueltos.json"

#: Orden de las funciones en el prompt y en la lista de puntajes (tabla del enunciado).
ORDEN = ["Background", "Gap", "Basis", "Comparison", "Application", "Modification_Improvement",
         "Evidence", "Identification_of_the_Originator", "Further_Reading"]

ENFOQUES = {
    "E1": {"nombre": "zero-shot genérico", "criterios": False, "ejemplos": False},
    "E2": {"nombre": "zero-shot con criterios", "criterios": True, "ejemplos": False},
    "E3": {"nombre": "few-shot genérico", "criterios": False, "ejemplos": True},
    "E4": {"nombre": "few-shot con criterios", "criterios": True, "ejemplos": True},
}

#: Definiciones y criterios de la tabla del enunciado (sección 2).
CRITERIOS = {
    "Background": ("Citations used to provide context, summarize the general background of a research topic, "
                   "or trace the history of a field or idea. They lay the foundation for understanding the focal study.",
                   "Reviews existing literature; cites review papers or previous studies depicting the current state "
                   "of the field; highlights important or prevailing findings; traces historical development or key ideas."),
    "Gap": ("Citations that help identify research gaps or unexplored areas, justifying the author's choice of research topic.",
            "Highlights what has or has not been done; identifies gaps in knowledge or areas for further study; "
            "justifies the relevance of the current research by contrasting it with existing work."),
    "Basis": ("Citations that provide a foundation for the current research, at a macro level (broad influence) "
              "or a micro level (specific key contributions).",
              "Serves as the intellectual starting point; significantly shapes the research idea or hypothesis; "
              "the focal research builds upon, continues, or expands on the cited work."),
    "Comparison": ("Citations used to compare the current work with cited studies or to draw comparisons between cited studies.",
                   "Highlights similarities or differences; compares methodologies, findings, algorithms, data, or "
                   "theoretical concepts; may point out advantages of the current study or link different cited works."),
    "Application": ("Citations that directly employ a method, technique, or tool from the cited work without modification.",
                    "Utilizes existing methods, algorithms, instruments, or data; employs the cited work as a practical "
                    "tool (e.g., equations, analysis methods) for calculation or experimentation; applies it without modification."),
    "Modification_Improvement": ("Citations in which methods or tools from the cited work are adapted, expanded, "
                                 "or modified for the current research.",
                                 "The cited method is adapted, improved, or extended to fit new experimental conditions "
                                 "or research objectives; uses the cited work as a foundation but modifies it to enhance "
                                 "precision or scope."),
    "Evidence": ("Citations used to support claims, hypotheses, or findings in the current research.",
                 "Supports arguments, hypotheses, or factual statements; justifies research design, methodologies, or "
                 "experimental procedures; substantiates or explains findings; provides evidence to mitigate limitations "
                 "or support further research suggestions."),
    "Identification_of_the_Originator": ("Citations used to acknowledge the original source of an idea, concept, method, or theory.",
                                         "Identifies the original publication where a key idea or method was first "
                                         "introduced; acknowledges pioneers in the field; gives credit to the priority "
                                         "of a cited work."),
    "Further_Reading": ("Citations that direct readers to additional or supplementary literature for more detailed "
                        "information or context.",
                        "Alerts readers to new, different, or relevant sources of information; provides more complete "
                        "details on data, methods, or background not fully covered in the current work."),
}

NO_DISPONIBLE = "not available"


def _texto(valor) -> str:
    if valor is None:
        return NO_DISPONIBLE
    texto = " ".join(str(valor).split())
    return NO_DISPONIBLE if not texto or texto.lower() == "nan" else texto


def entrada_cita(fila: dict) -> str:
    """Campos de una cita, en el mismo formato para la cita evaluada y para los ejemplos."""
    return (f"Citation context: \"{_texto(fila.get('citation_context'))}\"\n"
            f"Section: {_texto(fila.get('rhetorical_section'))}\n"
            f"Cited paper title: {_texto(fila.get('cited_paper_title'))}\n"
            f"Cited paper abstract: {_texto(fila.get('cited_paper_abstract'))}")


def lista(funcion: str) -> str:
    """Respuesta resuelta: 1.0 en la función y 0.0 en las demás."""
    return "[" + ", ".join("1.0" if f == funcion else "0.0" for f in ORDEN) + "]"


def bloque_criterios() -> str:
    lineas = ["Definitions and criteria of each citation function:"]
    for i, f in enumerate(ORDEN, 1):
        definicion, criterios = CRITERIOS[f]
        lineas.append(f"{i}. {f}\n   Definition: {definicion}\n   Criteria: {criterios}")
    return "\n".join(lineas)


def bloque_ejemplos(ruta: Path = EJEMPLOS_FILE) -> str:
    ejemplos = json.loads(ruta.read_text())["ejemplos"]
    partes = ["Solved examples:"]
    for i, e in enumerate(ejemplos, 1):
        partes.append(f"Example {i}\n{entrada_cita(e)}\nAnswer: {lista(e['funcion'])}")
    return "\n\n".join(partes)


def instrucciones(enfoque: str) -> str:
    """Mensaje de sistema del enfoque: fijo para todas las citas."""
    cfg = ENFOQUES[enfoque]
    partes = [
        "Classify the rhetorical function of a citation in a scientific paper.",
        "You receive the citation context, the section of the citing paper where it appears, and the title "
        "and abstract of the cited paper.",
        "Score each of the following 9 citation functions between 0.0 and 1.0, in this exact order:\n"
        + ", ".join(ORDEN) + ".",
    ]
    if cfg["criterios"]:
        partes.append(bloque_criterios())
    if cfg["ejemplos"]:
        partes.append(bloque_ejemplos())
    partes.append("Answer only with a list of 9 numbers, for example "
                  "[0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]. Do not add any other text.")
    return "\n\n".join(partes)


_LISTA = re.compile(r"\[([^\[\]]*)\]")
_NUMERO = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def interpretar(texto: str | None, cortada: bool = False) -> dict:
    """Lee la lista de 9 puntajes de una respuesta.

    estado: valido | sin_lista | no_numerico | cantidad | rango | cortada.
    `cortada` indica que el modelo se detuvo por el límite de tokens; solo se reporta así
    si además la lista no es válida. Con un empate en el puntaje mayor se elige la primera
    función en ORDEN y se marca `empate`.
    """
    res = {"puntajes": None, "etiqueta": None, "estado": None, "empate": False}
    m = _LISTA.search(texto or "")
    if not m:
        res["estado"] = "cortada" if cortada else "sin_lista"
        return res
    partes = [p.strip() for p in m.group(1).split(",") if p.strip()]
    if not all(_NUMERO.match(p) for p in partes):
        res["estado"] = "no_numerico"
        return res
    valores = [float(p) for p in partes]
    if len(valores) != len(ORDEN):
        res["estado"] = "cortada" if cortada else "cantidad"
        return res
    if any(v < 0.0 or v > 1.0 for v in valores):
        res["estado"] = "rango"
        return res
    mayor = max(valores)
    res.update(puntajes=valores, etiqueta=ORDEN[valores.index(mayor)],
               estado="valido", empate=valores.count(mayor) > 1)
    return res


_SECUENCIA = re.compile(r"(?<![\d.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*,\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)){8}(?![\d.])")


def interpretar_ajustado(texto: str | None, cortada: bool = False) -> dict:
    """Lectura tolerante: recupera el resultado real de respuestas con formato inválido.

    Primero aplica la lectura estricta de `interpretar`. Si no es válida, quita las marcas
    de Markdown (`*`, `_`, `` ` ``) y, si la respuesta no trae corchetes, busca una
    secuencia de exactamente 9 números separados por comas. `ajuste` indica qué se
    corrigió: None (válida sin ajuste), `markdown`, `sin_corchetes` o `no_recuperable`.
    """
    estricta = interpretar(texto, cortada)
    if estricta["estado"] == "valido":
        return {**estricta, "ajuste": None}
    limpio = re.sub(r"[*_`]", "", texto or "")
    if _LISTA.search(limpio):
        res, ajuste = interpretar(limpio, cortada), "markdown"
    else:
        m = _SECUENCIA.search(limpio)
        res, ajuste = (interpretar(f"[{m.group(0)}]", cortada), "sin_corchetes") if m else (estricta, None)
    if res["estado"] != "valido":
        return {**estricta, "ajuste": "no_recuperable"}
    return {**res, "ajuste": ajuste}
