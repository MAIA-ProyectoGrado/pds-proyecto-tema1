"""Evaluación de modelos de lenguaje sobre el test humano (enfoques de prompt, comparación
con los encoders y eficiencia).

Siete modelos responden con los prompts de `src/api/prompts_llm.py` (cuatro enfoques):
GPT-6.2 Sol, GPT-6 Luna y GPT-6.2 Astra por la API de OpenAI (Responses), y Nova Pro,
Nova 2 Lite, Ministral 3 8B y Gemma 3 4B por Amazon Bedrock (Converse, us-east-2).

Etapas (todas sobre las 2.241 citas del test):
  1   Los 4 enfoques de cada modelo, salvo Astra, en una muestra de 50 citas por clase de
      `human_label` (Identification_of_the_Originator tiene 45): 445 citas, semilla 42.
  1b  Astra en la misma muestra, con el mejor enfoque de Sol.
  2   El mejor enfoque de cada modelo, salvo Astra, en las demás 1.796 citas.
  3   Estabilidad: 50 citas de la muestra, 3 repeticiones, mejor enfoque, los 6 modelos
      que no son Astra.
  R   Efecto del razonamiento: Sol con esfuerzo `high` (variante gpt-6.2-sol-high, en una
      submuestra de 11 citas por clase de la muestra, semilla 42) y `xhigh`
      (gpt-6.2-sol-xhigh, en la muestra completa), con el mejor enfoque de Sol; cada una se
      compara con Sol en `low` en las mismas citas.
El mejor enfoque es el de mayor F1 macro contra `human_label` en la muestra; en empate,
el más barato (E1, E2, E3, E4).

Cada llamada se guarda en dataset/evaluacion_llm/respuestas_<modelo>.jsonl. Una corrida se
puede interrumpir y reanudar: solo se repiten las llamadas que terminaron en error. El
gasto se calcula con los tokens reales; antes de cada llamada se comprueba que el gasto
acumulado, más lo que está en curso, no supere los topes por modelo y por servicio.

Uso:
    pip install -r requirements-llm.txt        # credenciales: aws login y OPENAI_API_KEY
    python scripts/evaluate_llm.py estimar --etapa 1
    python scripts/evaluate_llm.py correr --etapa 1 --modelo gemma-3-4b
    python scripts/evaluate_llm.py analizar
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_jueces import cita_id  # noqa: E402
from src.api.prompts_llm import ENFOQUES, ORDEN, entrada_cita, instrucciones, interpretar, interpretar_ajustado  # noqa: E402

DATA_DIR = REPO_ROOT / "dataset"
OUT_DIR = DATA_DIR / "evaluacion_llm"
DOCS_DIR = REPO_ROOT / "docs" / "evaluacion_llm"
MUESTRA_FILE = DOCS_DIR / "muestra_test.json"
SEMILLA = 42
POR_CLASE = 50
N_ESTABILIDAD = 50
REPETICIONES = 3
REGION = "us-east-2"

#: Configuración por modelo, fijada con la fase 0 (23 nov 2026). Precios en USD por millón
#: de tokens (entrada, salida, entrada en caché), de las tarifas estándar publicadas por cada
#: proveedor. En Bedrock no se usa la caché, así que su precio en caché es el de entrada.
#: `salida_tipica` estima el gasto antes de cada llamada.
MODELOS = {
    "gemma-3-4b": dict(servicio="bedrock", id="google.gemma-3-4b-it", precio=(0.04, 0.08, 0.04),
                       max_tokens=80, temperatura=0.0, razonamiento=None, salida_tipica=50),
    "ministral-3-8b": dict(servicio="bedrock", id="mistral.ministral-3-8b-instruct", precio=(0.15, 0.15, 0.15),
                           max_tokens=80, temperatura=0.0, razonamiento=None, salida_tipica=50),
    "nova-pro": dict(servicio="bedrock", id="us.amazon.nova-pro-v1:0", precio=(0.80, 3.20, 0.80),
                     max_tokens=80, temperatura=0.0, razonamiento=None, salida_tipica=50),
    "nova-2-lite": dict(servicio="bedrock", id="us.amazon.nova-2-lite-v1:0", precio=(0.33, 2.75, 0.33),
                        max_tokens=3000, temperatura=None, razonamiento="low", salida_tipica=1000),
    "gpt-6-luna": dict(servicio="openai", id="gpt-6-luna", precio=(0.10, 0.50, 0.01),
                       max_tokens=1000, temperatura=0.0, razonamiento="low", salida_tipica=200),
    "gpt-6.2-sol": dict(servicio="openai", id="gpt-6.2-sol", precio=(2.00, 10.00, 0.10),
                        max_tokens=1000, temperatura=None, razonamiento="low", salida_tipica=60),
    "gpt-6.2-astra": dict(servicio="openai", id="gpt-6.2-astra", precio=(8.00, 40.00, 0.40),
                          max_tokens=1000, temperatura=None, razonamiento="low", salida_tipica=60),
}
PRINCIPALES = list(MODELOS)
#: Variantes de Sol con más esfuerzo de razonamiento (etapa R). El máximo de salida sube
#: para que el razonamiento no corte la lista de puntajes.
#: `alcance`: submuestra de 11 citas por clase o muestra completa.
VARIANTES = {"gpt-6.2-sol-high": {"esfuerzo": "high", "alcance": "submuestra"},
             "gpt-6.2-sol-xhigh": {"esfuerzo": "xhigh", "alcance": "muestra"}}
for _variante, _cfg in VARIANTES.items():
    MODELOS[_variante] = {**MODELOS["gpt-6.2-sol"], "razonamiento": _cfg["esfuerzo"], "max_tokens": 16000,
                          "salida_tipica": 1000}
POR_CLASE_R = 11

TOPE_MODELO = {"gpt-6.2-astra": 10.0, "gpt-6.2-sol": 17.0, "gpt-6-luna": 2.0,
               "gpt-6.2-sol-high": 4.0, "gpt-6.2-sol-xhigh": 4.0}
TOPE_SERVICIO = {"openai": 30.0, "bedrock": 26.0}
SIN_ASTRA = [m for m in PRINCIPALES if m != "gpt-6.2-astra"]
INVALIDO = "__invalido__"

_TRANSITORIOS = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException",
                 "InternalServerException", "ModelTimeoutException", "RateLimitError",
                 "APITimeoutError", "APIConnectionError", "InternalServerError"}
MAX_INTENTOS = 6


# ── datos ───────────────────────────────────────────────────────────────────

def cargar_test():
    """Test con sus tres referencias y las predicciones de SciBERT B y TF-IDF, por ID de cita."""
    import pandas as pd

    test = pd.read_csv(DATA_DIR / "test_enriquecido_top3_chunks.csv")
    test["id"] = [cita_id(p, c) for p, c in zip(test["pair_id"], test["citation_context"])]
    pred = pd.read_csv(REPO_ROOT / "docs" / "modelos_v4" / "predicciones_test.csv").set_index("fila").sort_index()
    if len(pred) != len(test) or not (pred["label"].to_numpy() == test["label"].to_numpy()).all():
        raise SystemExit("predicciones_test.csv no corresponde al test.")
    test["pred_scibert"] = pred["pred_scibert"].to_numpy()
    test["pred_tfidf"] = pred["pred_tfidf"].to_numpy()
    hoja = pd.read_excel(DATA_DIR / "anotacion" / "hoja_anotacion_test.xlsx", sheet_name="Anotacion")
    david = dict(zip(hoja["id"], hoja["etiqueta"]))
    test["david_label"] = test["id"].map(david)
    if test["david_label"].isna().any():
        raise SystemExit("Faltan citas en la anotación de David.")
    return test.set_index("id", drop=False)


def construir_muestra(test, semilla: int = SEMILLA) -> dict:
    """Muestra estratificada por `human_label` y subconjunto de estabilidad, reproducibles."""
    ids = []
    for clase in ORDEN:
        grupo = test[test["human_label"] == clase]
        ids += grupo.sample(n=min(POR_CLASE, len(grupo)), random_state=semilla)["id"].tolist()
    orden = {i: n for n, i in enumerate(test["id"])}
    ids = sorted(ids, key=orden.get)
    estabilidad = sorted(random.Random(semilla).sample(ids, N_ESTABILIDAD), key=orden.get)
    return {"semilla": semilla, "por_clase": POR_CLASE, "muestra": ids, "estabilidad": estabilidad}


def cargar_muestra(test) -> dict:
    muestra = construir_muestra(test)
    if MUESTRA_FILE.exists():
        guardada = json.loads(MUESTRA_FILE.read_text())
        if guardada["muestra"] != muestra["muestra"] or guardada["estabilidad"] != muestra["estabilidad"]:
            raise SystemExit(f"{MUESTRA_FILE} no coincide con la muestra calculada.")
    else:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        MUESTRA_FILE.write_text(json.dumps(muestra, indent=2) + "\n")
    return muestra


def submuestra_razonamiento(test, muestra: dict, semilla: int = SEMILLA) -> list[str]:
    """Hasta POR_CLASE_R citas de la muestra por clase de `human_label`, en el orden del test."""
    clase = dict(zip(test["id"], test["human_label"]))
    orden = {i: n for n, i in enumerate(test["id"])}
    rng = random.Random(semilla)
    ids = []
    for c in ORDEN:
        grupo = [i for i in muestra["muestra"] if clase[i] == c]
        ids += rng.sample(grupo, min(POR_CLASE_R, len(grupo)))
    return sorted(ids, key=orden.get)


# ── registros, costo y mejor enfoque ───────────────────────────────────────

def clave(r: dict) -> tuple:
    return (r["id"], r["enfoque"], r["etapa"], r.get("repeticion", 1))


def leer_registros(ruta: Path) -> dict[tuple, dict]:
    """Último registro de cada llamada (cita, enfoque, etapa, repetición)."""
    regs: dict[tuple, dict] = {}
    if ruta.exists():
        for linea in ruta.read_text().splitlines():
            if linea.strip():
                r = json.loads(linea)
                regs[clave(r)] = r
    return regs


def costo(r: dict, precio: tuple[float, float, float]) -> float:
    """USD de una llamada. `tokens_entrada` incluye los tokens en caché, que se cobran aparte."""
    cache = r.get("tokens_cache") or 0
    entrada = (r.get("tokens_entrada") or 0) - cache
    return (entrada * precio[0] + cache * precio[2] + (r.get("tokens_salida") or 0) * precio[1]) / 1e6


def gasto(regs: dict[tuple, dict], modelo: str) -> float:
    return sum(costo(r, MODELOS[modelo]["precio"]) for r in regs.values())


def f1_macro(y_true: list, y_pred: list) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=ORDEN, average="macro", zero_division=0))


def etiqueta_ajustada(r: dict) -> str:
    """Etiqueta con la lectura tolerante del texto literal de la respuesta."""
    if r["estado"] == "valido" or "respuesta" not in r:
        return r["etiqueta"] or INVALIDO
    return interpretar_ajustado(r["respuesta"], r["estado"] == "cortada")["etiqueta"] or INVALIDO


def predicciones(regs: dict[tuple, dict], etapa: str, enfoque: str, repeticion: int = 1,
                 ajustada: bool = False) -> dict[str, str]:
    """Etiqueta por cita (las respuestas inválidas cuentan como error).

    Con `ajustada`, las respuestas de formato inválido se leen con `interpretar_ajustado`.
    """
    return {r["id"]: (etiqueta_ajustada(r) if ajustada else (r["etiqueta"] or INVALIDO)) for r in regs.values()
            if r["etapa"] == etapa and r["enfoque"] == enfoque and r.get("repeticion", 1) == repeticion
            and r["estado"] != "error"}


def mejor_enfoque(regs: dict[tuple, dict], referencia: dict[str, str], ids: list[str]) -> str:
    """Enfoque de mayor F1 macro en la muestra; en empate, el primero en E1..E4."""
    puntajes = {}
    for e in ENFOQUES:
        pred = predicciones(regs, "1", e)
        faltan = [i for i in ids if i not in pred]
        if faltan:
            raise SystemExit(f"El enfoque {e} tiene {len(faltan)} citas sin respuesta en la etapa 1.")
        puntajes[e] = round(f1_macro([referencia[i] for i in ids], [pred[i] for i in ids]), 6)
    return max(ENFOQUES, key=lambda e: (puntajes[e], -list(ENFOQUES).index(e)))


# ── llamadas a los servicios ────────────────────────────────────────────────

class Bedrock:
    def __init__(self, cliente):
        self.cliente = cliente

    def llamar(self, cfg: dict, sistema: str, usuario: str) -> dict:
        kw = dict(modelId=cfg["id"], system=[{"text": sistema}],
                  messages=[{"role": "user", "content": [{"text": usuario}]}],
                  inferenceConfig={"maxTokens": cfg["max_tokens"]})
        if cfg["temperatura"] is not None:
            kw["inferenceConfig"]["temperature"] = cfg["temperatura"]
        if cfg["razonamiento"]:
            kw["additionalModelRequestFields"] = {
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": cfg["razonamiento"]}}
        r = self.cliente.converse(**kw)
        u = r["usage"]
        return {"texto": "".join(p.get("text", "") for p in r["output"]["message"]["content"]),
                "tokens_entrada": u["inputTokens"], "tokens_cache": u.get("cacheReadInputTokens", 0),
                "tokens_salida": u["outputTokens"], "tokens_razonamiento": None,
                "latencia_servicio_ms": r.get("metrics", {}).get("latencyMs"),
                "motivo_fin": r.get("stopReason"), "cortada": r.get("stopReason") == "max_tokens",
                "parametros": {k: v for k, v in kw.items() if k in ("inferenceConfig", "additionalModelRequestFields")}}


class OpenAIResponses:
    def __init__(self, cliente):
        self.cliente = cliente
        self.sin_temperatura: set[str] = set()
        self._candado = threading.Lock()

    def llamar(self, cfg: dict, sistema: str, usuario: str) -> dict:
        kw = dict(model=cfg["id"], instructions=sistema, input=usuario, max_output_tokens=cfg["max_tokens"],
                  prompt_cache_key="scif-" + hashlib.sha256(sistema.encode()).hexdigest()[:16])
        if cfg["razonamiento"]:
            kw["reasoning"] = {"effort": cfg["razonamiento"]}
        if cfg["temperatura"] is not None and cfg["id"] not in self.sin_temperatura:
            try:
                return self._llamar({**kw, "temperature": cfg["temperatura"]})
            except Exception as exc:
                if type(exc).__name__ != "BadRequestError" or "temperature" not in str(exc):
                    raise
                with self._candado:
                    self.sin_temperatura.add(cfg["id"])
        return self._llamar(kw)

    def _llamar(self, kw: dict) -> dict:
        r = self.cliente.responses.create(**kw)
        u = r.usage
        return {"texto": r.output_text or "", "tokens_entrada": u.input_tokens,
                "tokens_cache": getattr(u.input_tokens_details, "cached_tokens", 0) or 0,
                "tokens_salida": u.output_tokens,
                "tokens_razonamiento": getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0,
                "latencia_servicio_ms": None, "motivo_fin": r.status, "cortada": r.status == "incomplete",
                "parametros": {k: v for k, v in kw.items()
                               if k in ("max_output_tokens", "reasoning", "temperature", "prompt_cache_key")}}


def es_transitorio(exc: Exception) -> bool:
    codigo = getattr(exc, "response", {}).get("Error", {}).get("Code", "") if isinstance(
        getattr(exc, "response", None), dict) else ""
    return codigo in _TRANSITORIOS or type(exc).__name__ in _TRANSITORIOS


def consultar(servicio, cfg: dict, enfoque: str, fila: dict, dormir=time.sleep) -> dict:
    """Una llamada con reintentos para errores transitorios; devuelve el registro sin claves."""
    sistema, usuario = instrucciones(enfoque), entrada_cita(fila)
    for intento in range(1, MAX_INTENTOS + 1):
        t0 = time.perf_counter()
        try:
            res = servicio.llamar(cfg, sistema, usuario)
        except Exception as exc:
            if es_transitorio(exc) and intento < MAX_INTENTOS:
                dormir(min(60, 2 ** intento) + random.random())
                continue
            return {"estado": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "intentos": intento,
                    "etiqueta": None, "puntajes": None, "empate": False}
        texto = res.pop("texto")
        lectura = interpretar(texto, res.pop("cortada"))
        return {"respuesta": texto, **res, **lectura,
                "latencia_cliente_ms": round((time.perf_counter() - t0) * 1000, 1), "intentos": intento}
    raise AssertionError("inalcanzable")


def estimar_llamada(modelo: str, enfoque: str, fila: dict) -> float:
    cfg = MODELOS[modelo]
    tokens = len(instrucciones(enfoque) + entrada_cita(fila)) / 4
    return tokens / 1e6 * cfg["precio"][0] + cfg["salida_tipica"] / 1e6 * cfg["precio"][1]


# ── etapas ──────────────────────────────────────────────────────────────────

def tareas(etapa: str, modelo: str, test, muestra: dict, mejores: dict[str, str]) -> list[dict]:
    if etapa == "1":
        if modelo == "gpt-6.2-astra":
            raise SystemExit("Astra no corre la etapa 1: usa la 1b.")
        return [{"id": i, "enfoque": e, "etapa": "1", "repeticion": 1} for e in ENFOQUES for i in muestra["muestra"]]
    if etapa == "1b":
        if modelo != "gpt-6.2-astra":
            raise SystemExit("La etapa 1b es solo para Astra.")
        if "gpt-6.2-sol" not in mejores:
            raise SystemExit("Falta completar la etapa 1 de Sol para elegir el enfoque de Astra.")
        return [{"id": i, "enfoque": mejores["gpt-6.2-sol"], "etapa": "1b", "repeticion": 1} for i in muestra["muestra"]]
    if etapa == "2":
        if modelo == "gpt-6.2-astra":
            raise SystemExit("Astra no corre la etapa 2.")
        en_muestra = set(muestra["muestra"])
        if modelo not in mejores:
            raise SystemExit(f"Falta completar la etapa 1 de {modelo}.")
        return [{"id": i, "enfoque": mejores[modelo], "etapa": "2", "repeticion": 1}
                for i in test["id"] if i not in en_muestra]
    if etapa == "3":
        base = "gpt-6.2-sol" if modelo == "gpt-6.2-astra" else modelo
        if base not in mejores:
            raise SystemExit(f"Falta completar la etapa 1 de {base}.")
        enfoque = mejores[base]
        return [{"id": i, "enfoque": enfoque, "etapa": "3", "repeticion": k}
                for k in range(1, REPETICIONES + 1) for i in muestra["estabilidad"]]
    if etapa == "R":
        if modelo not in VARIANTES:
            raise SystemExit("La etapa R es solo para las variantes de razonamiento de Sol.")
        if "gpt-6.2-sol" not in mejores:
            raise SystemExit("Falta completar la etapa 1 de Sol para elegir el enfoque.")
        ids = muestra["muestra"] if VARIANTES[modelo]["alcance"] == "muestra" else submuestra_razonamiento(test, muestra)
        return [{"id": i, "enfoque": mejores["gpt-6.2-sol"], "etapa": "R", "repeticion": 1} for i in ids]
    raise SystemExit(f"Etapa desconocida: {etapa}")


def correr(modelo: str, lista: list[dict], filas: dict[str, dict], servicio, salida: Path,
           gastado_servicio: float, hilos: int = 8, limite: int | None = None, dormir=time.sleep) -> dict:
    """Ejecuta las tareas pendientes respetando los topes; devuelve un resumen."""
    regs = leer_registros(salida)
    pendientes = [t for t in lista if regs.get(clave(t), {}).get("estado", "error") == "error"]
    if limite:
        pendientes = pendientes[:limite]
    cfg = MODELOS[modelo]
    tope_m = TOPE_MODELO.get(modelo, float("inf"))
    tope_s = TOPE_SERVICIO[cfg["servicio"]]
    gastado_m = gasto(regs, modelo)
    reales = [costo(r, cfg["precio"]) for r in regs.values() if r.get("tokens_entrada")]
    en_curso = 0.0
    candado = threading.Lock()
    conteo = Counter()
    salida.parent.mkdir(parents=True, exist_ok=True)

    def tarea(t: dict, est: float):
        res = consultar(servicio, cfg, t["enfoque"], filas[t["id"]], dormir=dormir)
        res.update(t, modelo=modelo, model_id=cfg["id"],
                   fecha_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        return res, est

    detenido = None
    with ThreadPoolExecutor(max_workers=hilos) as pool:
        activos = set()
        for t in pendientes:
            # La estimación por largo del prompt se corrige con el costo real observado.
            est = max(estimar_llamada(modelo, t["enfoque"], filas[t["id"]]),
                      sum(reales) / len(reales) if reales else 0.0)
            while len(activos) >= hilos:
                hechos, activos = wait(activos, return_when=FIRST_COMPLETED)
                for f in hechos:
                    gastado_m, gastado_servicio, en_curso = _cerrar(f, salida, candado, conteo, modelo,
                                                                     gastado_m, gastado_servicio, en_curso, reales)
            if gastado_m + en_curso + est > tope_m or gastado_servicio + en_curso + est > tope_s:
                detenido = (f"tope alcanzado: modelo USD {gastado_m:.3f} de {tope_m}, "
                            f"servicio USD {gastado_servicio:.3f} de {tope_s}")
                break
            en_curso += est
            activos.add(pool.submit(tarea, t, est))
        for f in wait(activos).done:
            gastado_m, gastado_servicio, en_curso = _cerrar(f, salida, candado, conteo, modelo,
                                                             gastado_m, gastado_servicio, en_curso, reales)
    return {"estados": dict(conteo), "gasto_modelo_usd": round(gastado_m, 4),
            "gasto_servicio_usd": round(gastado_servicio, 4), "detenido": detenido}


def _cerrar(fut, salida, candado, conteo, modelo, gastado_m, gastado_s, en_curso, reales):
    res, est = fut.result()
    c = costo(res, MODELOS[modelo]["precio"])
    if res.get("tokens_entrada"):
        reales.append(c)
    with candado, open(salida, "a") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    conteo[res["estado"]] += 1
    return gastado_m + c, gastado_s + c, en_curso - est


# ── análisis ────────────────────────────────────────────────────────────────

def _metricas(y_true: list, y_pred: list) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score
    kw = dict(labels=ORDEN, zero_division=0)
    return {"n": len(y_true),
            "exactitud": round(sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true), 4) if y_true else None,
            "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", **kw)), 4),
            "exhaustividad_macro": round(float(recall_score(y_true, y_pred, average="macro", **kw)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", **kw)), 4),
            "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", **kw)), 4)}


REFERENCIAS = {"juan_david": "human_label", "david": "david_label", "automatica": "label"}


def _contra_referencias(test, pred: dict[str, str], ids: list[str]) -> dict:
    out = {}
    for nombre, col in REFERENCIAS.items():
        y = [test.at[i, col] for i in ids]
        p = [pred[i] for i in ids]
        out[nombre] = _metricas(y, p)
        validos = [k for k, v in enumerate(p) if v != INVALIDO]
        out[nombre]["f1_macro_solo_validas"] = round(f1_macro([y[k] for k in validos], [p[k] for k in validos]), 4) if validos else None
    return out


def _razonamiento(test, muestra: dict, registros: dict[str, dict[tuple, dict]], mejores: dict[str, str]) -> dict:
    """Cada variante de razonamiento frente a Sol en `low` (etapa 1), en las citas que respondieron ambos."""
    if "gpt-6.2-sol" not in mejores:
        return {}
    enfoque = mejores["gpt-6.2-sol"]

    def hechas(regs, etapa):
        return {r["id"]: r for r in regs.values()
                if r["etapa"] == etapa and r["enfoque"] == enfoque and r["estado"] != "error"}

    def resumen(m, h, ids):
        regs = [h[i] for i in ids]
        lat = sorted(r["latencia_cliente_ms"] for r in regs)
        return {"esfuerzo": MODELOS[m]["razonamiento"],
                "ajustado": _contra_referencias(test, {i: etiqueta_ajustada(h[i]) for i in ids}, ids),
                "tasa_error_formato": round(sum(r["estado"] != "valido" for r in regs) / len(regs), 4),
                "cortadas": sum(r["estado"] == "cortada" for r in regs),
                "tokens_razonamiento_promedio": round(sum(r.get("tokens_razonamiento") or 0 for r in regs) / len(regs), 1),
                "tokens_salida_promedio": round(sum(r.get("tokens_salida") or 0 for r in regs) / len(regs), 1),
                "costo_por_cita_usd": round(sum(costo(r, MODELOS[m]["precio"]) for r in regs) / len(regs), 5),
                "latencia_mediana_ms": lat[len(lat) // 2]}

    base = hechas(registros.get("gpt-6.2-sol", {}), "1")
    out = {"enfoque": enfoque, "variantes": {}}
    for v in VARIANTES:
        if v not in registros:
            continue
        h = hechas(registros[v], "R")
        ids = [i for i in test["id"] if i in h and i in base]
        if ids:
            out["variantes"][v] = {"citas": len(ids), "alcance": VARIANTES[v]["alcance"],
                                   "misma_etiqueta_que_low": round(sum(etiqueta_ajustada(h[i]) == etiqueta_ajustada(base[i])
                                                                       for i in ids) / len(ids), 4),
                                   "low": resumen("gpt-6.2-sol", base, ids), "variante": resumen(v, h, ids)}
    return out


def _ambiguedad(test, finales: dict[str, dict[str, str]]) -> list[dict]:
    """Proporción de cada clase real (revisión humana) que cada modelo asigna a otra clase, en sus respuestas válidas."""
    tasas: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for m, pred in finales.items():
        ids = [i for i in pred if pred[i] != INVALIDO]
        reales = Counter(test.at[i, "human_label"] for i in ids)
        pares = Counter((test.at[i, "human_label"], pred[i]) for i in ids)
        for real in ORDEN:
            for predicha in ORDEN:
                if real != predicha and reales[real]:
                    tasas[real, predicha][m] = round(pares[real, predicha] / reales[real], 4)
    filas = [{"clase_real": r, "clase_predicha": p, "promedio_modelos": round(sum(v.values()) / len(v), 4),
              "maximo_modelos": max(v.values()), "por_modelo": v} for (r, p), v in tasas.items()]
    return sorted(filas, key=lambda f: (-f["promedio_modelos"], f["clase_real"], f["clase_predicha"]))


def analizar(test, muestra: dict, registros: dict[str, dict[tuple, dict]]) -> dict:
    todos_regs = registros
    registros = {m: r for m, r in registros.items() if m in PRINCIPALES}
    ids_m, todos = muestra["muestra"], list(test["id"])
    ref_jd = dict(zip(test["id"], test["human_label"]))
    mejores = {m: mejor_enfoque(registros[m], ref_jd, ids_m) for m in SIN_ASTRA if m in registros}
    res = {"descripcion": "Modelos de lenguaje sobre el test humano: enfoques en la muestra, comparación final y eficiencia.",
           "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "muestra": {"citas": len(ids_m), "estabilidad": len(muestra["estabilidad"]), "semilla": muestra["semilla"]},
           "mejor_enfoque": dict(mejores), "enfoques": {}, "comparacion": {}, "eficiencia": {}}

    # Enfoques en la muestra
    for m, regs in registros.items():
        etapa = "1b" if m == "gpt-6.2-astra" else "1"
        for e in ENFOQUES:
            pred = predicciones(regs, etapa, e)
            if not all(i in pred for i in ids_m):
                continue
            estados = Counter(r["estado"] for r in regs.values() if r["etapa"] == etapa and r["enfoque"] == e)
            pred_aj = predicciones(regs, etapa, e, ajustada=True)
            res["enfoques"].setdefault(m, {})[e] = {
                **_contra_referencias(test, pred, ids_m),
                "ajustado": _contra_referencias(test, pred_aj, ids_m),
                "llamadas": sum(estados.values()),
                "errores_formato": {k: v for k, v in estados.items() if k != "valido"},
                "tasa_error_formato": round(1 - estados["valido"] / sum(estados.values()), 4),
                "tasa_error_formato_ajustada": round(sum(pred_aj[i] == INVALIDO for i in ids_m) / len(ids_m), 4)}

    # Comparación final: mejor enfoque en las 2.241 (Astra: solo la muestra)
    finales: dict[str, dict[str, str]] = {}
    ajustadas: dict[str, dict[str, str]] = {}
    for m, regs in registros.items():
        if m == "gpt-6.2-astra":
            if "gpt-6.2-sol" in mejores:
                pred = predicciones(regs, "1b", mejores["gpt-6.2-sol"])
                if all(i in pred for i in ids_m):
                    finales[m] = pred
                    ajustadas[m] = predicciones(regs, "1b", mejores["gpt-6.2-sol"], ajustada=True)
            continue
        if m not in mejores:
            continue
        pred = {**predicciones(regs, "1", mejores[m]), **predicciones(regs, "2", mejores[m])}
        if all(i in pred for i in todos):
            finales[m] = pred
            ajustadas[m] = {**predicciones(regs, "1", mejores[m], ajustada=True),
                            **predicciones(regs, "2", mejores[m], ajustada=True)}
    for nombre, col in (("scibert-b", "pred_scibert"), ("tfidf", "pred_tfidf")):
        finales[nombre] = ajustadas[nombre] = dict(zip(test["id"], test[col]))
    for m, pred in finales.items():
        ids = ids_m if m == "gpt-6.2-astra" else todos
        res["comparacion"][m] = {"citas": len(ids), "test_completo": _contra_referencias(test, pred, ids),
                                 "muestra": _contra_referencias(test, pred, ids_m),
                                 "test_completo_ajustado": _contra_referencias(test, ajustadas[m], ids),
                                 "muestra_ajustada": _contra_referencias(test, ajustadas[m], ids_m)}

    # Eficiencia
    for m, regs in registros.items():
        validos = [r for r in regs.values() if r["estado"] != "error"]
        enfoque_final = mejores.get("gpt-6.2-sol") if m == "gpt-6.2-astra" else mejores.get(m)
        finales_r = [r for r in validos if r["enfoque"] == enfoque_final and r["etapa"] in ("1", "1b", "2")]
        lat = sorted(r["latencia_cliente_ms"] for r in finales_r)
        estab = defaultdict(list)
        for r in validos:
            if r["etapa"] == "3":
                estab[r["id"]].append(r["etiqueta"] or INVALIDO)
        completas = [v for v in estab.values() if len(v) == REPETICIONES]
        n_final = len(finales_r)
        res["eficiencia"][m] = {
            "servicio": MODELOS[m]["servicio"], "llamadas": len(regs),
            "errores_de_llamada": sum(1 for r in regs.values() if r["estado"] == "error"),
            "gasto_total_usd": round(gasto(regs, m), 4),
            "costo_por_1000_citas_usd": round(1000 * sum(costo(r, MODELOS[m]["precio"]) for r in finales_r) / n_final, 4) if n_final else None,
            "latencia_media_ms": round(sum(lat) / len(lat), 1) if lat else None,
            "latencia_mediana_ms": lat[len(lat) // 2] if lat else None,
            "latencia_p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
            "tasa_error_formato": round(sum(1 for r in validos if r["estado"] != "valido") / len(validos), 4) if validos else None,
            "tasa_error_formato_ajustada": round(sum(1 for r in validos if etiqueta_ajustada(r) == INVALIDO) / len(validos), 4) if validos else None,
            "tasa_empates": round(sum(1 for r in validos if r.get("empate")) / len(validos), 4) if validos else None,
            "estabilidad_misma_prediccion": round(sum(len(set(v)) == 1 for v in completas) / len(completas), 4) if completas else None,
            "estabilidad_formato_valido": round(sum(all(x != INVALIDO for x in v) for v in completas) / len(completas), 4) if completas else None,
            "parametros_efectivos": next((r.get("parametros") for r in validos), None),
        }
    res["ambiguedad"] = _ambiguedad(test, finales)
    res["razonamiento"] = _razonamiento(test, muestra, todos_regs, mejores)
    res["_finales"] = finales
    res["_ajustadas"] = ajustadas
    res["_ids_muestra"] = ids_m
    return res


def escribir_resultados(test, res: dict) -> None:
    import csv
    finales = res.pop("_finales")
    ajustadas = res.pop("_ajustadas")
    en_muestra = set(res.pop("_ids_muestra"))
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "resultados_llm.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")

    with open(DOCS_DIR / "enfoques.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "enfoque", "f1_macro_juan_david", "exactitud_juan_david", "f1_macro_david",
                    "f1_macro_automatica", "tasa_error_formato", "f1_macro_juan_david_ajustado",
                    "f1_macro_david_ajustado", "f1_macro_automatica_ajustado", "tasa_error_formato_ajustada"])
        for m, por_e in res["enfoques"].items():
            for e, v in por_e.items():
                a = v["ajustado"]
                w.writerow([m, e, v["juan_david"]["f1_macro"], v["juan_david"]["exactitud"], v["david"]["f1_macro"],
                            v["automatica"]["f1_macro"], v["tasa_error_formato"], a["juan_david"]["f1_macro"],
                            a["david"]["f1_macro"], a["automatica"]["f1_macro"], v["tasa_error_formato_ajustada"]])

    estados = sorted({k for por_e in res["enfoques"].values() for v in por_e.values() for k in v["errores_formato"]})
    with open(DOCS_DIR / "formato_por_enfoque.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "enfoque", "llamadas", "validas", *estados, "tasa_error_formato",
                    "tasa_error_formato_ajustada"])
        for m, por_e in res["enfoques"].items():
            for e, v in por_e.items():
                err = v["errores_formato"]
                w.writerow([m, e, v["llamadas"], v["llamadas"] - sum(err.values()), *(err.get(k, 0) for k in estados),
                            v["tasa_error_formato"], v["tasa_error_formato_ajustada"]])

    with open(DOCS_DIR / "comparacion_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "citas", "lectura", "referencia", "exactitud", "precision_macro",
                    "exhaustividad_macro", "f1_macro", "f1_micro"])
        for m, v in res["comparacion"].items():
            for lectura, clave_res in (("estricta", "test_completo"), ("ajustada", "test_completo_ajustado")):
                for ref, met in v[clave_res].items():
                    w.writerow([m, v["citas"], lectura, ref, met["exactitud"], met["precision_macro"],
                                met["exhaustividad_macro"], met["f1_macro"], met["f1_micro"]])

    modelos_llm = [m for m in finales if m not in ("scibert-b", "tfidf")]
    with open(DOCS_DIR / "predicciones_llm.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "en_muestra", "human_label", "david_label", "label", "pred_scibert", "pred_tfidf"]
                   + [c for m in modelos_llm for c in (m, f"{m}_ajustada")])
        for i in test["id"]:
            w.writerow([i, i in en_muestra, test.at[i, "human_label"], test.at[i, "david_label"], test.at[i, "label"],
                        test.at[i, "pred_scibert"], test.at[i, "pred_tfidf"]]
                       + [x for m in modelos_llm for x in (finales[m].get(i, ""), ajustadas[m].get(i, ""))])

    from sklearn.metrics import confusion_matrix, f1_score
    filas = []
    for clase in ORDEN:
        fila = {"clase": clase}
        for m, pred in finales.items():
            ids = list(pred) if m == "gpt-6.2-astra" else list(test["id"])
            y = [test.at[i, "human_label"] for i in ids]
            fila[m] = round(float(f1_score(y, [pred[i] for i in ids], labels=[clase], average="macro", zero_division=0)), 3)
        filas.append(fila)
    with open(DOCS_DIR / "f1_por_clase_llm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    with open(DOCS_DIR / "eficiencia.csv", "w", newline="") as f:
        campos = ["modelo", "servicio", "llamadas", "gasto_total_usd", "costo_por_1000_citas_usd",
                  "latencia_media_ms", "latencia_mediana_ms", "latencia_p95_ms", "tasa_error_formato", "tasa_empates",
                  "estabilidad_misma_prediccion", "estabilidad_formato_valido"]
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for m, v in res["eficiencia"].items():
            w.writerow({"modelo": m, **v})

    modelos_amb = list(finales)
    with open(DOCS_DIR / "ambiguedad.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clase_real", "clase_predicha", "promedio_modelos", "maximo_modelos", *modelos_amb])
        for fila in res["ambiguedad"]:
            w.writerow([fila["clase_real"], fila["clase_predicha"], fila["promedio_modelos"], fila["maximo_modelos"],
                        *(fila["por_modelo"].get(m, "") for m in modelos_amb)])

    if res.get("razonamiento", {}).get("variantes"):
        with open(DOCS_DIR / "razonamiento.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variante", "citas", "esfuerzo", "f1_macro_juan_david", "exactitud_juan_david", "f1_macro_david",
                        "exactitud_david", "f1_macro_automatica", "tokens_razonamiento_promedio", "costo_por_cita_usd",
                        "latencia_mediana_ms", "tasa_error_formato", "misma_etiqueta_que_low"])
            for v, comp in res["razonamiento"]["variantes"].items():
                for lado in ("low", "variante"):
                    x, a = comp[lado], comp[lado]["ajustado"]
                    w.writerow([v, comp["citas"], x["esfuerzo"], a["juan_david"]["f1_macro"], a["juan_david"]["exactitud"],
                                a["david"]["f1_macro"], a["david"]["exactitud"], a["automatica"]["f1_macro"],
                                x["tokens_razonamiento_promedio"], x["costo_por_cita_usd"], x["latencia_mediana_ms"],
                                x["tasa_error_formato"], comp["misma_etiqueta_que_low"]])

    from scripts.extraer_corridas_notebook import figura_matriz
    for m, pred in finales.items():
        if m == "scibert-b":
            continue
        ids = [i for i in pred if pred[i] != INVALIDO]
        mat = confusion_matrix([test.at[i, "human_label"] for i in ids], [pred[i] for i in ids], labels=ORDEN)
        detalle = f"{len(ids)} citas" if m == "tfidf" else f"{len(ids)} respuestas válidas"
        figura_matriz({"orden": ORDEN, "valores": mat.tolist()},
                      f"{m} en el test contra la revisión humana\n({detalle}; % de cada clase real)",
                      DOCS_DIR / "figuras" / f"matriz_confusion_{m}.png")


# ── programa ────────────────────────────────────────────────────────────────

def _registros_todos() -> dict[str, dict[tuple, dict]]:
    return {m: leer_registros(OUT_DIR / f"respuestas_{m}.jsonl") for m in MODELOS
            if (OUT_DIR / f"respuestas_{m}.jsonl").exists()}


def _mejores(test, muestra, registros) -> dict[str, str]:
    ref = dict(zip(test["id"], test["human_label"]))
    out = {}
    for m in SIN_ASTRA:
        if m in registros:
            try:
                out[m] = mejor_enfoque(registros[m], ref, muestra["muestra"])
            except SystemExit:
                pass
    return out


def _servicio(nombre: str, hilos: int):
    if nombre == "bedrock":
        import boto3
        from botocore.config import Config
        return Bedrock(boto3.client("bedrock-runtime", region_name=REGION,
                                    config=Config(retries={"max_attempts": 1}, max_pool_connections=hilos * 2)))
    from openai import OpenAI
    return OpenAIResponses(OpenAI(max_retries=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre in ("estimar", "correr"):
        p = sub.add_parser(nombre)
        p.add_argument("--etapa", required=True, choices=["1", "1b", "2", "3", "R"])
        p.add_argument("--modelo", choices=sorted(MODELOS))
        if nombre == "correr":
            p.add_argument("--hilos", type=int, default=8)
            p.add_argument("--limite", type=int)
    sub.add_parser("analizar")
    args = ap.parse_args()

    test = cargar_test()
    muestra = cargar_muestra(test)
    filas = test.to_dict("index")
    registros = _registros_todos()
    mejores = _mejores(test, muestra, registros)

    if args.cmd == "analizar":
        res = analizar(test, muestra, registros)
        escribir_resultados(test, res)
        print(json.dumps({"mejor_enfoque": res["mejor_enfoque"],
                          "f1_macro_juan_david": {m: v["test_completo"]["juan_david"]["f1_macro"]
                                                  for m, v in res["comparacion"].items()}}, indent=2))
        return 0

    if args.modelo:
        modelos = [args.modelo]
    elif args.etapa == "R":
        modelos = list(VARIANTES)
    else:
        modelos = [m for m in PRINCIPALES if not (
            (args.etapa in ("1", "2") and m == "gpt-6.2-astra") or (args.etapa == "1b" and m != "gpt-6.2-astra"))]
    if args.cmd == "estimar":
        total = defaultdict(float)
        for m in modelos:
            try:
                lista = tareas(args.etapa, m, test, muestra, mejores)
            except (SystemExit, KeyError) as exc:
                print(f"{m:15s} no se puede estimar todavía ({exc})")
                continue
            hechas = leer_registros(OUT_DIR / f"respuestas_{m}.jsonl")
            pend = [t for t in lista if hechas.get(clave(t), {}).get("estado", "error") == "error"]
            usd = sum(estimar_llamada(m, t["enfoque"], filas[t["id"]]) for t in pend)
            total[MODELOS[m]["servicio"]] += usd
            print(f"{m:15s} llamadas pendientes {len(pend):5d}  USD {usd:7.3f}  (gastado {gasto(hechas, m):.3f})")
        print({k: round(v, 3) for k, v in total.items()})
        return 0

    for m in modelos:
        lista = tareas(args.etapa, m, test, muestra, mejores)
        servicio = MODELOS[m]["servicio"]
        gastado_s = sum(gasto(regs, otro) for otro, regs in registros.items() if MODELOS[otro]["servicio"] == servicio)
        resumen = correr(m, lista, filas, _servicio(servicio, args.hilos), OUT_DIR / f"respuestas_{m}.jsonl",
                         gastado_s, hilos=args.hilos, limite=args.limite)
        print(f"{m}: {json.dumps(resumen, ensure_ascii=False)}", flush=True)
        registros[m] = leer_registros(OUT_DIR / f"respuestas_{m}.jsonl")
        if resumen["detenido"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
