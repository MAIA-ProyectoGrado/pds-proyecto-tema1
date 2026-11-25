"""Esquema de jueces del pre-etiquetado sobre el dataset completo.

Tres modelos abiertos de familias distintas clasifican las 14.937 citas del dataset,
antes de la división en particiones, con las definiciones y la entrada del
pre-etiquetado de `notebooks/Obtener_dataset.ipynb` (sección 4):
  - prompt de sistema: `docs/jueces/prompt_preetiquetado.txt`, con las mismas
    definiciones y criterios del notebook; pide solo la etiqueta, sin los puntajes;
  - entrada: sección retórica, contexto, título y los primeros 300 caracteres del abstract;
  - generación: temperatura 0 y máximo 30 tokens.

Llama 3.2 3B, el modelo del pre-etiquetado, ya no se ofrece en Bedrock.
La etiqueta automática del dataset (`label`, de Llama con el rescate) entra en el
análisis como cuarto evaluador.

Subcomandos:
  correr    Consulta un juez en Bedrock y guarda cada respuesta en
            dataset/jueces/respuestas_<juez>.jsonl. Se puede interrumpir y reanudar:
            solo consulta las citas que aún no tienen respuesta válida.
  analizar  Junta las respuestas, calcula el voto de mayoría y escribe
            dataset/jueces/jueces_preetiquetado.csv y docs/jueces/metricas_jueces.json.

Uso:
    pip install boto3 "botocore[crt]" pandas scikit-learn
    python scripts/evaluate_jueces.py correr --juez gemma-3-4b
    python scripts/evaluate_jueces.py analizar
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.scif_predict import LABELS  # noqa: E402

DATA_DIR = REPO_ROOT / "dataset"
OUT_DIR = DATA_DIR / "jueces"
DOCS_DIR = REPO_ROOT / "docs" / "jueces"
PROMPT_FILE = DOCS_DIR / "prompt_preetiquetado.txt"
REGION = "us-east-2"
MAX_TOKENS = 30

JUECES = {
    "ministral-3-8b": {"model_id": "mistral.ministral-3-8b-instruct", "familia": "Mistral"},
    "gemma-3-4b": {"model_id": "google.gemma-3-4b-it", "familia": "Google"},
    "qwen3-32b": {"model_id": "qwen.qwen3-32b-v1:0", "familia": "Alibaba (Qwen)"},
}

#: Precio de lista en us-east-2, USD por millón de tokens (API de precios de AWS, 22 nov 2026).
PRECIOS = {
    "ministral-3-8b": (0.15, 0.15),
    "gemma-3-4b": (0.04, 0.08),
    "qwen3-32b": (0.15, 0.60),
}

#: Errores de Bedrock que se reintentan con espera exponencial.
REINTENTABLES = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException",
                 "InternalServerException", "ModelTimeoutException"}
MAX_INTENTOS = 6


# ── entrada y salida del modelo ─────────────────────────────────────────────

def cita_id(pair_id, contexto) -> str:
    """Identificador estable de una cita: el par (pair_id, contexto) es único en el dataset."""
    return hashlib.sha1(f"{pair_id}\x1f{contexto}".encode()).hexdigest()[:16]


def mensaje_usuario(fila: dict) -> str:
    """Entrada por cita, igual que en el notebook. Los valores faltantes se convierten a
    texto como allí: una sección faltante llega al modelo como «nan»."""
    ctx = str(fila["citation_context"]).strip()
    sec = str(fila.get("rhetorical_section", "Scientific Body"))
    tit = str(fila.get("cited_paper_title", "")).strip()
    abs_txt = str(fila.get("cited_paper_abstract", "")).strip()
    texto = f"Rhetorical Section: {sec}\nCitation Context: \"{ctx}\""
    if tit and len(tit) > 5 and tit.lower() != "nan":
        texto += f"\nCited Paper Title: \"{tit}\""
    if abs_txt and len(abs_txt) > 25 and abs_txt.lower() != "nan":
        texto += f"\nCited Paper Abstract: \"{abs_txt[:300]}...\""
    return texto


_CANONICA = {re.sub(r"[^a-z]", "", l.lower()): l for l in LABELS}


def _canonica(nombre: str) -> str | None:
    return _CANONICA.get(re.sub(r"[^a-z]", "", str(nombre).lower()))


def interpretar(texto: str) -> tuple[str | None, str]:
    """Etiqueta de la respuesta y cómo se obtuvo.

    json: JSON válido con `predicted_label` y una de las 9 clases (con o sin bloque ```).
    texto: no hay JSON válido, pero el texto nombra exactamente una de las 9 clases.
    invalido: ninguna de las dos.
    """
    limpio = re.sub(r"```(?:json)?", "", texto or "").strip()
    m = re.search(r"\{.*?\}", limpio, re.S)
    if m:
        try:
            etiqueta = _canonica(json.loads(m.group(0)).get("predicted_label", ""))
            if etiqueta:
                return etiqueta, "json"
        except (json.JSONDecodeError, AttributeError):
            pass
    plano = re.sub(r"[^a-z]", "", limpio.lower())
    encontradas = {l for k, l in _CANONICA.items() if k in plano}
    if len(encontradas) == 1:
        return encontradas.pop(), "texto"
    return None, "invalido"


# ── llamadas a Bedrock ──────────────────────────────────────────────────────

def consultar(cliente, model_id: str, sistema: str, usuario: str, dormir=time.sleep) -> dict:
    """Una llamada a la API Converse, con reintentos solo para errores transitorios."""
    from botocore.exceptions import ClientError

    for intento in range(1, MAX_INTENTOS + 1):
        try:
            r = cliente.converse(
                modelId=model_id,
                system=[{"text": sistema}],
                messages=[{"role": "user", "content": [{"text": usuario}]}],
                inferenceConfig={"temperature": 0.0, "maxTokens": MAX_TOKENS},
            )
            texto = "".join(p.get("text", "") for p in r["output"]["message"]["content"])
            etiqueta, estado = interpretar(texto)
            return {"respuesta": texto, "etiqueta": etiqueta, "estado": estado,
                    "tokens_entrada": r["usage"]["inputTokens"], "tokens_salida": r["usage"]["outputTokens"],
                    "latencia_ms": r["metrics"]["latencyMs"], "motivo_fin": r.get("stopReason"),
                    "intentos": intento}
        except ClientError as exc:
            codigo = exc.response.get("Error", {}).get("Code", "")
            if codigo not in REINTENTABLES or intento == MAX_INTENTOS:
                return {"respuesta": None, "etiqueta": None, "estado": "error",
                        "error": f"{codigo}: {exc}", "intentos": intento}
            dormir(min(60, 2 ** intento) + random.random())
    raise AssertionError("inalcanzable")


def leer_respuestas(ruta: Path) -> dict[str, dict]:
    """Última respuesta guardada por cita."""
    hechas: dict[str, dict] = {}
    if ruta.exists():
        for linea in ruta.read_text().splitlines():
            if linea.strip():
                r = json.loads(linea)
                hechas[r["id"]] = r
    return hechas


def correr(juez: str, datos, cliente, salida: Path, hilos: int = 8, limite: int | None = None) -> dict:
    """Consulta al juez las citas sin respuesta válida y agrega cada resultado a `salida`."""
    sistema = PROMPT_FILE.read_text()
    model_id = JUECES[juez]["model_id"]
    hechas = leer_respuestas(salida)
    pendientes = [f for f in datos if hechas.get(f["id"], {}).get("estado") not in ("json", "texto", "invalido")]
    if limite:
        pendientes = pendientes[:limite]
    salida.parent.mkdir(parents=True, exist_ok=True)
    candado = threading.Lock()
    conteo = Counter()

    def tarea(fila):
        res = consultar(cliente, model_id, sistema, mensaje_usuario(fila))
        res.update(id=fila["id"], juez=juez, model_id=model_id,
                   fecha_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with candado, open(salida, "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        return res["estado"]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=hilos) as pool:
        for i, fut in enumerate(as_completed([pool.submit(tarea, f) for f in pendientes]), 1):
            conteo[fut.result()] += 1
            if i % 500 == 0 or i == len(pendientes):
                print(f"  {juez}: {i}/{len(pendientes)} en {time.perf_counter() - t0:.0f} s  {dict(conteo)}", flush=True)
    return dict(conteo)


# ── análisis ────────────────────────────────────────────────────────────────

def fleiss_kappa(filas: list[list[str]]) -> float:
    """Kappa de Fleiss para N ítems calificados por el mismo número de evaluadores."""
    n = len(filas[0])
    categorias = sorted({c for f in filas for c in f})
    conteos = [[f.count(c) for c in categorias] for f in filas]
    p_i = [(sum(x * x for x in fila) - n) / (n * (n - 1)) for fila in conteos]
    p_j = [sum(fila[j] for fila in conteos) / (len(filas) * n) for j in range(len(categorias))]
    p_bar, p_e = sum(p_i) / len(p_i), sum(p * p for p in p_j)
    return 1.0 if math.isclose(p_e, 1.0) else (p_bar - p_e) / (1 - p_e)


def mayoria(votos: list) -> str | None:
    """La etiqueta con al menos dos votos de tres; None si no hay mayoría."""
    validos = [v for v in votos if isinstance(v, str)]
    if not validos:
        return None
    etiqueta, n = Counter(validos).most_common(1)[0]
    return etiqueta if n >= 2 else None


def _acierto(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    return {"n": len(y_true),
            "exactitud": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 4),
            "kappa_cohen": round(float(cohen_kappa_score(y_true, y_pred)), 4)}


def cargar_dataset():
    import pandas as pd
    total = pd.read_csv(DATA_DIR / "dataset_definitivo_total_chunks.csv")
    total["id"] = [cita_id(p, c) for p, c in zip(total["pair_id"], total["citation_context"])]
    for split in ("train", "val", "test"):
        part = pd.read_csv(DATA_DIR / f"{split}_enriquecido_top3_chunks.csv")
        ids = {cita_id(p, c) for p, c in zip(part["pair_id"], part["citation_context"])}
        total.loc[total["id"].isin(ids), "particion"] = split
        if split == "test":
            humano = {cita_id(p, c): (h, s) for p, c, h, s in zip(part["pair_id"], part["citation_context"],
                                                               part["human_label"], part["certeza_evaluacion"])}
    total["human_label"] = total["id"].map(lambda i: humano.get(i, (None, None))[0])
    total["certeza_evaluacion"] = total["id"].map(lambda i: humano.get(i, (None, None))[1])
    return total


def analizar(total, respuestas: dict[str, dict[str, dict]]) -> tuple[dict, "object"]:
    jueces = list(respuestas)
    tabla = total[["id", "particion", "source_dataset", "label", "human_label", "certeza_evaluacion"]].copy()
    for j in jueces:
        tabla[f"pred_{j}"] = tabla["id"].map(lambda i, j=j: respuestas[j].get(i, {}).get("etiqueta"))
        tabla[f"estado_{j}"] = tabla["id"].map(lambda i, j=j: respuestas[j].get(i, {}).get("estado", "sin_respuesta"))
    cols = [f"pred_{j}" for j in jueces]
    tabla["mayoria"] = [mayoria(list(v)) for v in tabla[cols].itertuples(index=False)]
    tabla["n_coinciden"] = [Counter(x for x in v if isinstance(x, str)).most_common(1)[0][1]
                            if any(isinstance(x, str) for x in v) else 0
                            for v in tabla[cols].itertuples(index=False)]

    completas = tabla.dropna(subset=cols)
    m: dict = {
        "descripcion": "Esquema de jueces del pre-etiquetado sobre las 14.937 citas, antes de la división.",
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jueces": {j: JUECES[j] for j in jueces},
        "parametros": {"temperatura": 0.0, "max_tokens": MAX_TOKENS, "region": REGION,
                       "prompt": str(PROMPT_FILE.relative_to(REPO_ROOT))},
        "citas": len(tabla),
        "por_juez": {},
    }
    for j in jueces:
        estados = Counter(tabla[f"estado_{j}"])
        regs = [r for r in respuestas[j].values() if r.get("tokens_entrada") is not None]
        t_in, t_out = sum(r["tokens_entrada"] for r in regs), sum(r["tokens_salida"] for r in regs)
        lat = sorted(r["latencia_ms"] for r in regs)
        m["por_juez"][j] = {
            "formato": dict(estados),
            "formato_valido_pct": round(100 * (estados["json"] + estados["texto"]) / len(tabla), 2),
            "distribucion": tabla[f"pred_{j}"].value_counts().to_dict(),
            "tokens_entrada": t_in, "tokens_salida": t_out,
            "costo_usd": round(t_in / 1e6 * PRECIOS[j][0] + t_out / 1e6 * PRECIOS[j][1], 3),
            "latencia_mediana_ms": lat[len(lat) // 2] if lat else None,
            "acuerdo_con_label": _acierto(completas["label"].tolist(), completas[f"pred_{j}"].tolist()),
        }
    m["acuerdo_entre_jueces"] = {
        "n": len(completas),
        "fleiss_3_jueces": round(fleiss_kappa(completas[cols].values.tolist()), 4),
        "fleiss_3_jueces_y_label": round(fleiss_kappa(completas[cols + ["label"]].values.tolist()), 4),
        "unanimes_pct": round(100 * (completas["n_coinciden"] == len(jueces)).mean(), 2),
        "con_mayoria_pct": round(100 * completas["mayoria"].notna().mean(), 2),
        "por_pares": {f"{a}|{b}": _acierto(completas[f"pred_{a}"].tolist(), completas[f"pred_{b}"].tolist())["kappa_cohen"]
                      for i, a in enumerate(jueces) for b in jueces[i + 1:]},
    }
    con_may = tabla[tabla["mayoria"].notna()]
    cambia = con_may["mayoria"] != con_may["label"]
    m["cambios_si_se_usara_la_mayoria"] = {
        "n_con_mayoria": len(con_may),
        "cambiarian_pct": round(100 * cambia.mean(), 2),
        "por_clase_de_label": (100 * cambia.groupby(con_may["label"]).mean()).round(2).to_dict(),
        "por_fuente": (100 * cambia.groupby(con_may["source_dataset"]).mean()).round(2).to_dict(),
    }

    test = tabla[tabla["human_label"].notna()]
    contra_humano = {"label (Llama con rescate)": _acierto(test["human_label"].tolist(), test["label"].tolist())}
    for j in jueces:
        t = test.dropna(subset=[f"pred_{j}"])
        contra_humano[j] = _acierto(t["human_label"].tolist(), t[f"pred_{j}"].tolist())
    t = test[test["mayoria"].notna()]
    contra_humano["mayoria_de_jueces"] = _acierto(t["human_label"].tolist(), t["mayoria"].tolist())
    t = test[test["n_coinciden"] == len(jueces)]
    contra_humano["solo_unanimes"] = _acierto(t["human_label"].tolist(), t["mayoria"].tolist())
    t = test[test["mayoria"].notna() & (test["mayoria"] == test["label"])]
    contra_humano["mayoria_coincide_con_label"] = _acierto(t["human_label"].tolist(), t["label"].tolist())
    m["test_contra_revision_humana"] = contra_humano
    return m, tabla


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("correr")
    c.add_argument("--juez", required=True, choices=sorted(JUECES))
    c.add_argument("--hilos", type=int, default=8)
    c.add_argument("--limite", type=int, help="consultar solo las primeras N citas pendientes")
    sub.add_parser("analizar")
    args = ap.parse_args()

    total = cargar_dataset()
    if args.cmd == "correr":
        import boto3
        from botocore.config import Config
        cliente = boto3.client("bedrock-runtime", region_name=REGION,
                               config=Config(retries={"max_attempts": 1}, max_pool_connections=args.hilos * 2))
        datos = total.to_dict("records")
        conteo = correr(args.juez, datos, cliente, OUT_DIR / f"respuestas_{args.juez}.jsonl", args.hilos, args.limite)
        print(f"{args.juez}: {conteo}")
        return 0

    respuestas = {j: leer_respuestas(OUT_DIR / f"respuestas_{j}.jsonl") for j in JUECES
                  if (OUT_DIR / f"respuestas_{j}.jsonl").exists()}
    metricas, tabla = analizar(total, respuestas)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "metricas_jueces.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False) + "\n")
    tabla.to_csv(OUT_DIR / "jueces_preetiquetado.csv", index=False)
    print(json.dumps({k: metricas[k] for k in ("acuerdo_entre_jueces", "cambios_si_se_usara_la_mayoria",
                                               "test_contra_revision_humana")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
