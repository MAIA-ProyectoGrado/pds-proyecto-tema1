"""Modelos de lenguaje en Amazon Bedrock: Nova Pro (comercial) y Ministral 3 8B
(pesos abiertos, 8 B parámetros).

Son los mejores de cada tipo en la evaluación del test (`scripts/evaluate_llm.py`):
Nova Pro es el de mayor F1-macro entre los 7 LLM y Ministral 3 8B el mejor de pesos
abiertos entre 1 y 8 B. Se consultan igual que en la evaluación: el prompt de
`prompts_llm.py` con el enfoque E4 (few-shot con criterios), el mejor de ambos,
temperatura 0 y como máximo 80 tokens de salida. El modelo devuelve un puntaje entre
0 y 1 por función; la predicción es el puntaje más alto.

La llamada usa las credenciales de AWS del entorno (en EC2, el rol de la instancia):
en el servidor no hay llaves. Dos topes acotan el gasto:

* un presupuesto diario (SCIF_LLM_DAILY_BUDGET_USD, USD 5 por defecto) que suma el
  costo real de cada respuesta, calculado con los tokens que informa Bedrock. Antes
  de cada llamada se reserva el costo máximo posible y después se ajusta al real, así
  que el tope se respeta aunque lleguen consultas simultáneas. El acumulado se guarda
  en un archivo (SCIF_LLM_SPEND_FILE) que sobrevive a los reinicios y que comparten
  los procesos de uvicorn;
* un límite de consultas por minuto y por IP (SCIF_LLM_RATE_LIMIT_PER_MIN), que
  aplica `main.py`.
"""
from __future__ import annotations

import fcntl
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from src.api import config
from src.api.labels import CANONICAL_LABELS
from src.api.prompts_llm import ENFOQUES, ORDEN, entrada_cita, instrucciones, interpretar

APPROACH = "E4"
MAX_TOKENS = 80
TEMPERATURE = 0.0
#: tope del abstract que se envía; los abstracts de arXiv rara vez pasan de 2.000 caracteres
MAX_ABSTRACT_CHARS = 4000
#: caracteres por token para estimar la entrada antes de la llamada; por lo bajo, para
#: no subestimar el costo
CHARS_PER_TOKEN = 3.0

#: Precios en USD por millón de tokens (entrada, salida), los de la evaluación.
#: `cost_per_query_usd` y `typical_latency_ms` son el costo medio y la latencia mediana
#: medidos en el test con este mismo enfoque (docs/evaluacion_llm/eficiencia.csv).
MODELS: dict[str, dict[str, Any]] = {
    "nova-pro": {
        "bedrock_id": "us.amazon.nova-pro-v1:0",
        "price": (0.80, 3.20),
        "cost_per_query_usd": 0.0025,
        "typical_latency_ms": 624,
    },
    "ministral-3-8b": {
        "bedrock_id": "mistral.ministral-3-8b-instruct",
        "price": (0.15, 0.15),
        "cost_per_query_usd": 0.00045,
        "typical_latency_ms": 365,
    },
}


class LLMUnavailable(RuntimeError):
    """Los modelos de lenguaje están desactivados en este servidor."""


class BudgetExceeded(RuntimeError):
    """Se alcanzó el presupuesto del día."""

    def __init__(self, message: str, retry_after_s: int):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class BedrockError(RuntimeError):
    """Bedrock no respondió o rechazó la consulta."""


class InvalidAnswer(RuntimeError):
    """El modelo respondió, pero no con la lista de 9 puntajes."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


# ── disponibilidad ───────────────────────────────────────────────────────────

def _boto3_installed() -> bool:
    try:
        import boto3  # noqa: F401
    except ImportError:
        return False
    return True


def available() -> bool:
    return config.LLM_ENABLED and _boto3_installed()


def unavailable_reason() -> str:
    if not config.LLM_ENABLED:
        return ("Desactivado en este servidor: los modelos de lenguaje requieren acceso a Amazon Bedrock "
                "(SCIF_LLM_ENABLED=true y credenciales de AWS con permiso bedrock:InvokeModel). "
                "Ver la sección «Modelos de lenguaje» del README.")
    return "Falta la librería boto3: instala src/api/requirements.txt."


# ── presupuesto diario ───────────────────────────────────────────────────────

def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def seconds_to_utc_midnight(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


class DailyBudget:
    """Gasto acumulado del día (UTC) en un archivo JSON, con bloqueo entre procesos."""

    def __init__(self, path: Path, budget_usd: float, today: Callable[[], str] = _utc_today):
        self.path = Path(path)
        self.budget_usd = budget_usd
        self.today = today
        self._lock = threading.Lock()

    def _update(self, change: Callable[[float], float | None]) -> float:
        """Lee el gasto del día, aplica `change` y lo guarda. `change` devuelve el nuevo
        gasto o None para dejarlo igual. Devuelve el gasto resultante."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                day = self.today()
                try:
                    data = json.loads(self.path.read_text())
                except (FileNotFoundError, ValueError):
                    data = {}
                spent = float(data.get("spent_usd", 0.0)) if data.get("date") == day else 0.0
                new = change(spent)
                if new is not None:
                    spent = max(0.0, new)
                    tmp = self.path.with_suffix(".tmp")
                    tmp.write_text(json.dumps({"date": day, "spent_usd": round(spent, 8)}))
                    tmp.replace(self.path)
                return spent
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def spent(self) -> float:
        return self._update(lambda s: None)

    def reserve(self, amount: float) -> None:
        """Suma `amount` al gasto del día si cabe en el presupuesto; si no, BudgetExceeded."""
        def change(spent: float) -> float:
            if spent + amount > self.budget_usd:
                raise BudgetExceeded(
                    f"Se alcanzó el tope diario de USD {self.budget_usd:g} para los modelos de lenguaje. "
                    "SciBERT y TF-IDF siguen disponibles; el cupo se renueva a las 00:00 UTC.",
                    seconds_to_utc_midnight())
            return spent + amount
        self._update(change)

    def settle(self, reserved: float, actual: float) -> float:
        """Reemplaza la reserva por el costo real. Devuelve el gasto del día."""
        return self._update(lambda s: s - reserved + actual)

    def status(self) -> dict[str, Any]:
        """Para /health. Si el archivo no se puede leer, lo informa en lugar de fallar."""
        try:
            spent: float | None = round(self.spent(), 4)
        except OSError:
            spent = None
        return {"date": self.today(), "spent_usd": spent, "budget_usd": self.budget_usd}


budget = DailyBudget(config.LLM_SPEND_FILE, config.LLM_DAILY_BUDGET_USD)


# ── costo ────────────────────────────────────────────────────────────────────

def cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = MODELS[model_id]["price"]
    return (input_tokens * p_in + output_tokens * p_out) / 1e6


def max_cost(model_id: str, system: str, user: str) -> float:
    """Cota superior del costo de una llamada, para reservarla antes de hacerla."""
    tokens_in = int((len(system) + len(user)) / CHARS_PER_TOKEN) + 50
    return cost(model_id, tokens_in, MAX_TOKENS)


# ── llamada a Bedrock ────────────────────────────────────────────────────────

_client = None
_client_lock = threading.Lock()


def client():
    """Cliente de Bedrock Runtime, creado una sola vez."""
    global _client
    with _client_lock:
        if _client is None:
            import boto3
            from botocore.config import Config

            _client = boto3.client(
                "bedrock-runtime", region_name=config.BEDROCK_REGION,
                config=Config(connect_timeout=5, read_timeout=config.LLM_TIMEOUT,
                              retries={"max_attempts": 2, "mode": "standard"}))
    return _client


@lru_cache(maxsize=None)
def system_prompt() -> str:
    return instrucciones(APPROACH)


def user_prompt(context: str, section: str | None, title: str | None, abstract: str | None) -> str:
    return entrada_cita({"citation_context": context, "rhetorical_section": section,
                         "cited_paper_title": title,
                         "cited_paper_abstract": abstract[:MAX_ABSTRACT_CHARS] if abstract else None})


def _explain(exc: Exception) -> str:
    """Mensaje para el usuario. No incluye el texto del error de AWS, que puede traer
    identificadores de la cuenta."""
    name = type(exc).__name__
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "")
    if name in {"NoCredentialsError", "PartialCredentialsError"}:
        return "El servidor no tiene credenciales de AWS para consultar Bedrock."
    if code == "AccessDeniedException":
        return "El servidor no tiene permiso para usar este modelo en Bedrock (bedrock:InvokeModel)."
    if code in {"ThrottlingException", "ServiceQuotaExceededException"}:
        return "Bedrock está limitando las consultas en este momento. Intenta de nuevo en unos segundos."
    if name in {"ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError"} or code == "ModelTimeoutException":
        return f"Bedrock no respondió en {config.LLM_TIMEOUT} s. Intenta de nuevo."
    return f"Bedrock no pudo responder ({code or name}). Intenta de nuevo."


def classify(model_id: str, context: str, section: str | None,
             title: str | None, abstract: str | None) -> dict[str, Any]:
    """Clasifica una cita con el modelo de lenguaje. Devuelve el mismo formato que el
    predictor local más el detalle de la llamada: la confianza es el puntaje mayor, las
    probabilidades son los puntajes normalizados a suma 1 y la ambigüedad se marca si
    hay empate o si los dos puntajes mayores difieren menos que SCIF_AMBIGUITY_THRESHOLD."""
    if not available():
        raise LLMUnavailable(unavailable_reason())
    meta = MODELS[model_id]
    system, user = system_prompt(), user_prompt(context, section, title, abstract)
    reserved = max_cost(model_id, system, user)
    try:
        budget.reserve(reserved)
    except OSError as exc:
        # sin registro de gasto no se puede respetar el tope: no se consulta
        raise LLMUnavailable("No se pudo leer el registro de gasto diario (SCIF_LLM_SPEND_FILE).") from exc

    started = time.perf_counter()
    try:
        r = client().converse(
            modelId=meta["bedrock_id"], system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": TEMPERATURE})
    except Exception as exc:
        budget.settle(reserved, 0.0)
        raise BedrockError(_explain(exc)) from exc
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    usage = r.get("usage", {})
    tokens_in, tokens_out = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
    spent_usd = cost(model_id, tokens_in, tokens_out)
    spent_today = budget.settle(reserved, spent_usd)

    raw = "".join(p.get("text", "") for p in r["output"]["message"]["content"])
    parsed = interpretar(raw, r.get("stopReason") == "max_tokens")
    if parsed["estado"] != "valido":
        raise InvalidAnswer("El modelo no respondió con la lista de 9 puntajes que pide el prompt. "
                            "Vuelve a intentarlo o usa otro modelo.", raw)

    scores = dict(zip(ORDEN, parsed["puntajes"]))
    total = sum(scores.values())
    probabilities = {lab: (scores[lab] / total if total > 0 else 1 / len(CANONICAL_LABELS))
                     for lab in CANONICAL_LABELS}
    label = parsed["etiqueta"]
    # Confianza y ambigüedad con los puntajes tal como los dio el modelo: cada uno ya
    # está entre 0 y 1. La distribución normalizada solo sirve para comparar las barras.
    top = sorted(scores.values(), reverse=True)
    return {
        "predicted_label": label,
        "confidence": scores[label],
        "probabilities": probabilities,
        # los puntajes vienen con uno o dos decimales: se redondea para que 0,7 - 0,6 sea 0,1
        "ambiguous": parsed["empate"] or round(top[0] - top[1], 6) < config.AMBIGUITY_THRESHOLD,
        "model_input": user,
        "latency_ms": latency_ms,
        "llm": {
            "provider": "Amazon Bedrock",
            "bedrock_model_id": meta["bedrock_id"],
            "approach": APPROACH,
            "approach_name": ENFOQUES[APPROACH]["nombre"],
            "scores": scores,
            "tie": parsed["empate"],
            "raw_response": raw,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost_usd": round(spent_usd, 6),
            "spent_today_usd": round(spent_today, 4),
            "daily_budget_usd": budget.budget_usd,
        },
    }
