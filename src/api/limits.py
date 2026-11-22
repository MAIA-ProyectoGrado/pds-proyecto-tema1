"""Límite de peticiones por IP (ventana deslizante de 60 s, en memoria).

Cada proceso de uvicorn lleva su propia cuenta: con SCIF_WORKERS=n el límite
efectivo por IP es hasta n veces SCIF_RATE_LIMIT_PER_MIN. En el despliegue con
Docker la IP del cliente llega desde nginx (cabecera X-Forwarded-For, que uvicorn
lee con --proxy-headers).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

WINDOW_S = 60.0
_MAX_KEYS = 10_000     # por encima, se purgan las IPs sin peticiones recientes


class RateLimiter:
    def __init__(self, per_minute: int, window_s: float = WINDOW_S):
        self.per_minute = per_minute
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, now: float | None = None) -> int:
        """Registra una petición. Devuelve 0 si se admite, o los segundos que
        faltan para que se libere un cupo."""
        if self.per_minute <= 0:
            return 0
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.per_minute:
                return max(1, math.ceil(self.window_s - (now - q[0])))
            q.append(now)
            if len(self._hits) > _MAX_KEYS:
                self._purge(now)
            return 0

    def _purge(self, now: float) -> None:
        for k in [k for k, q in self._hits.items() if not q or now - q[-1] >= self.window_s]:
            del self._hits[k]
