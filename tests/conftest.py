"""Configuración común de las pruebas.

Se fija antes de importar la API, que lee las variables al arrancar:
  * sin límite de peticiones: la suite hace más POST por minuto que el límite
    por defecto (las pruebas del límite crean su propio RateLimiter);
  * sin precarga en segundo plano, para que las pruebas no compitan con ella.
"""
import os

os.environ.setdefault("SCIF_RATE_LIMIT_PER_MIN", "0")
os.environ.setdefault("SCIF_WARMUP", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
