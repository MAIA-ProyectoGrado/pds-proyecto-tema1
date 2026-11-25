"""Curvas de aprendizaje, F1 por clase y matrices de confusión de los clasificadores v4.

Lee las salidas guardadas de `notebooks/Entrenamiento_clasificadores_scibert_mlflow.ipynb`
sin ejecutarlo: para cada corrida, la pérdida de entrenamiento, la de validación y el
F1 macro de validación de cada época, y el reporte por clase en validación. Las
métricas del notebook son todas de validación; las del test vienen de
`docs/modelos_v4/test_referencias_metricas.json` (`scripts/evaluate_test_humano.py`).

Escribe en docs/modelos_v4/:
  - corridas_validacion.json: épocas y reporte por clase de las 5 corridas;
  - f1_por_clase.csv: F1 por clase de las 5 corridas en validación y de SciBERT B y
    TF-IDF en el test, contra la revisión humana y contra la etiqueta automática;
  - figuras/curvas_aprendizaje.png;
  - figuras/matriz_confusion_test_humana.png y matriz_confusion_test_automatica.png
    (SciBERT B en el test, normalizadas por fila).

Uso:
    python scripts/extraer_corridas_notebook.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.scif_predict import LABELS  # noqa: E402

NOTEBOOK = REPO_ROOT / "notebooks" / "Entrenamiento_clasificadores_scibert_mlflow.ipynb"
OUT_DIR = REPO_ROOT / "docs" / "modelos_v4"
TEST_METRICS = OUT_DIR / "test_referencias_metricas.json"

#: Nombre corto de cada corrida para tablas y figuras.
NOMBRES = {
    "Run1_SciBERT_ConfigA_Metadata": "Run 1: SciBERT, config. A",
    "Run2_SciBERT_ConfigB_Top3Chunks": "Run 2: SciBERT, config. B",
    "Run3_SPECTER_ConfigB": "Run 3: SPECTER, config. B",
    "Run4_SciBERT_Frozen_Control": "Run 4: SciBERT congelado, config. B",
    "Run5_SciBERT_ConfigB_UnweightedLoss": "Run 5: SciBERT sin pesos, config. B",
}

_EXPERIMENTO = re.compile(r"INICIANDO EXPERIMENTO:\s*(\S+)")
_EPOCA = re.compile(r"Epoca (\d+)/(\d+) \[([\d.]+)s\] - Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Val Macro-F1: ([\d.]+)")
_FILA = re.compile(r"^\s*([A-Za-z_]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$")
_EXACTITUD = re.compile(r"^\s*accuracy\s+([\d.]+)\s+(\d+)\s*$")
_PROMEDIO = re.compile(r"^\s*(macro|weighted) avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$")


def texto_de_celda(celda: dict) -> str:
    partes = []
    for salida in celda.get("outputs", []):
        if "text" in salida:
            partes.append("".join(salida["text"]))
        elif "text/plain" in salida.get("data", {}):
            partes.append("".join(salida["data"]["text/plain"]))
    return "\n".join(partes)


def leer_corrida(texto: str) -> dict | None:
    """Épocas y reporte de validación de una corrida; None si el texto no es una corrida."""
    m = _EXPERIMENTO.search(texto)
    if not m:
        return None
    epocas = [{"epoca": int(e), "segundos": float(s), "perdida_entrenamiento": float(tl),
               "perdida_validacion": float(vl), "f1_macro_validacion": float(f1)}
              for e, _, s, tl, vl, f1 in _EPOCA.findall(texto)]
    por_clase, resumen = {}, {}
    reporte = texto.split("REPORTE DE CLASIFICACION (VALIDATION):", 1)
    for linea in (reporte[1].splitlines() if len(reporte) == 2 else []):
        if (f := _FILA.match(linea)) and f.group(1) in LABELS:
            por_clase[f.group(1)] = {"precision": float(f.group(2)), "exhaustividad": float(f.group(3)),
                                     "f1": float(f.group(4)), "soporte": int(f.group(5))}
        elif e := _EXACTITUD.match(linea):
            resumen["exactitud"] = float(e.group(1))
        elif p := _PROMEDIO.match(linea):
            resumen[f"{p.group(1)}_avg"] = {"precision": float(p.group(2)), "exhaustividad": float(p.group(3)),
                                            "f1": float(p.group(4))}
    return {"corrida": m.group(1), "nombre": NOMBRES.get(m.group(1), m.group(1)),
            "epocas": epocas, "validacion": {"por_clase": por_clase, **resumen}}


def leer_notebook(ruta: Path = NOTEBOOK) -> list[dict]:
    nb = json.loads(ruta.read_text())
    corridas = [c for c in (leer_corrida(texto_de_celda(celda)) for celda in nb["cells"]) if c]
    return [c for c in corridas if c["epocas"] and c["validacion"]["por_clase"]]


def tabla_f1(corridas: list[dict], test: dict) -> list[dict]:
    """Una fila por clase: F1 de cada corrida en validación y de los modelos v4 en el test."""
    columnas_test = {
        "SciBERT B, test (revisión humana)": test["modelos"]["scibert"]["human_label"],
        "SciBERT B, test (etiqueta automática)": test["modelos"]["scibert"]["label"],
        "TF-IDF, test (revisión humana)": test["modelos"]["tfidf-logreg"]["human_label"],
        "TF-IDF, test (etiqueta automática)": test["modelos"]["tfidf-logreg"]["label"],
    }
    filas = []
    for clase in LABELS:
        fila = {"clase": clase}
        for c in corridas:
            fila[f"{c['nombre']}, validación"] = c["validacion"]["por_clase"][clase]["f1"]
        for nombre, res in columnas_test.items():
            fila[nombre] = round(res["f1_por_clase"].get(clase, 0.0), 3)
        filas.append(fila)
    macro = {"clase": "Macro F1"}
    for c in corridas:
        macro[f"{c['nombre']}, validación"] = c["validacion"]["macro_avg"]["f1"]
    for nombre, res in columnas_test.items():
        macro[nombre] = round(res["f1_macro"], 3)
    filas.append(macro)
    return filas


def figura_curvas(corridas: list[dict], ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, c in enumerate(corridas):
        color = f"C{i}"
        x = [e["epoca"] for e in c["epocas"]]
        ax1.plot(x, [e["perdida_entrenamiento"] for e in c["epocas"]], "-o", color=color, label=f"{c['nombre']} (entrenamiento)")
        ax1.plot(x, [e["perdida_validacion"] for e in c["epocas"]], "--s", color=color, label=f"{c['nombre']} (validación)")
        ax2.plot(x, [e["f1_macro_validacion"] for e in c["epocas"]], "-o", color=color, label=c["nombre"])
    ax1.set(title="Pérdida por época (línea continua: entrenamiento; punteada: validación)",
            xlabel="Época", ylabel="Entropía cruzada")
    ax2.set(title="F1 macro de validación por época", xlabel="Época", ylabel="F1 macro", ylim=(0, 0.75))
    for ax in (ax1, ax2):
        ax.set_xticks(range(1, max(len(c["epocas"]) for c in corridas) + 1))
        ax.grid(alpha=0.3)
    ax1.title.set_fontsize(9)
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def figura_matriz(matriz: dict, titulo: str, ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orden, valores = matriz["orden"], matriz["valores"]
    pct = [[100 * v / sum(fila) if sum(fila) else 0.0 for v in fila] for fila in valores]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i, fila in enumerate(pct):
        for j, v in enumerate(fila):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 55 else "black")
    soporte = [sum(f) for f in valores]
    ax.set_xticks(range(len(orden)), orden, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(orden)), [f"{c} (n={n})" for c, n in zip(orden, soporte)], fontsize=8)
    ax.set(xlabel="Clase predicha", ylabel="Clase real", title=titulo)
    ax.title.set_fontsize(10)
    fig.colorbar(im, ax=ax, label="% de la clase real")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def main() -> int:
    corridas = leer_notebook()
    test = json.loads(TEST_METRICS.read_text())
    (OUT_DIR / "corridas_validacion.json").write_text(json.dumps(
        {"fuente": str(NOTEBOOK.relative_to(REPO_ROOT)),
         "nota": "Métricas de validación (2.280 citas, etiqueta automática); el notebook no evalúa el test.",
         "corridas": corridas}, indent=2, ensure_ascii=False) + "\n")

    filas = tabla_f1(corridas, test)
    with open(OUT_DIR / "f1_por_clase.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    figuras = OUT_DIR / "figuras"
    figura_curvas(corridas, figuras / "curvas_aprendizaje.png")
    sb = test["modelos"]["scibert"]
    figura_matriz(sb["human_label"]["matriz_confusion"],
                  "SciBERT B en el test contra la revisión humana\n(2.241 citas; % de cada clase real)",
                  figuras / "matriz_confusion_test_humana.png")
    figura_matriz(sb["label"]["matriz_confusion"],
                  "SciBERT B en el test contra la etiqueta automática\n(2.241 citas; % de cada clase real)",
                  figuras / "matriz_confusion_test_automatica.png")

    for c in corridas:
        ult = c["epocas"][-1]
        print(f"{c['nombre']:38s} épocas={len(c['epocas'])}  F1 val final={ult['f1_macro_validacion']:.4f}  "
              f"macro reporte={c['validacion']['macro_avg']['f1']:.2f}")
    print(f"-> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
