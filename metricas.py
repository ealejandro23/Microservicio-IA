"""
=============================================================================
METRICAS Y MONITOREO - MICROSERVICIO DE IA
============================================================================
Almacena en memoria las metricas del servicio:
  - Latencia (quantiles p50, p95, p99)
  - Tasa de error
  - Saturacion del modelo
  - Uso de recursos (CPU, memoria)
============================================================================
"""

import os
import threading
from collections import deque
from datetime import datetime
from typing import Dict

MAX_HISTORIAL = 1000
UMBRAL_P95_MS = 200
UMBRAL_P99_MS = 500
UMBRAL_ERROR = 1.0
UMBRAL_SATURACION = 80.0


class MetricsStore:
    """Almacen en memoria de metricas del microservicio de IA."""

    def __init__(self):
        self.latencias = deque(maxlen=MAX_HISTORIAL)
        self.errores = 0
        self.total_peticiones = 0
        self.predicciones = 0
        self.inicio = datetime.now()
        self._lock = threading.Lock()

    def record_latency(self, ms: float):
        with self._lock:
            self.latencias.append(ms)

    def record_error(self):
        with self._lock:
            self.errores += 1
            self.total_peticiones += 1

    def record_prediction(self):
        with self._lock:
            self.predicciones += 1

    def record_request(self):
        with self._lock:
            self.total_peticiones += 1

    def _percentil(self, p: float) -> float:
        if not self.latencias:
            return 0.0
        ordenadas = sorted(self.latencias)
        idx = int(len(ordenadas) * p / 100)
        return float(ordenadas[min(idx, len(ordenadas) - 1)])

    def get_metrics(self) -> Dict:
        with self._lock:
            uptime = (datetime.now() - self.inicio).total_seconds()
            p50 = self._percentil(50)
            p95 = self._percentil(95)
            p99 = self._percentil(99)
            tasa_error = (self.errores / self.total_peticiones * 100) if self.total_peticiones else 0.0
            saturacion = (self.predicciones / max(self.total_peticiones, 1)) * 100
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": round(uptime, 2),
                "latencia": {
                    "p50_ms": round(p50, 2),
                    "p95_ms": round(p95, 2),
                    "p99_ms": round(p99, 2),
                    "umbral_p95_ms": UMBRAL_P95_MS,
                    "umbral_p99_ms": UMBRAL_P99_MS,
                    "status": "ok" if p95 < UMBRAL_P95_MS else "warning",
                },
                "errores": {
                    "cantidad": self.errores,
                    "tasa_pct": round(tasa_error, 2),
                    "umbral_pct": UMBRAL_ERROR,
                    "status": "ok" if tasa_error < UMBRAL_ERROR else "warning",
                },
                "saturacion": {
                    "pct": round(saturacion, 2),
                    "umbral_pct": UMBRAL_SATURACION,
                    "status": "ok" if saturacion < UMBRAL_SATURACION else "warning",
                },
                "peticiones": {
                    "total": self.total_peticiones,
                    "predicciones": self.predicciones,
                    "errores": self.errores,
                },
            }


metrics_store = MetricsStore()


def get_metrics() -> Dict:
    return metrics_store.get_metrics()


def health_check() -> Dict:
    m = get_metrics()
    return {
        "status": "ok" if m["latencia"]["status"] == "ok" else "warning",
        "latencia_p95_ms": m["latencia"]["p95_ms"],
    }
