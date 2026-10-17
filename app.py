"""
=============================================================================
MICROSERVICIO DE IA - API REST
============================================================================
Sirve la inferencia del modelo de propension de compra (PyTorch).

Endpoints:
  GET  /health          - estado del servicio
  POST /predict         - prediccion individual
  POST /predict/batch   - prediccion por lotes
  GET  /modelo/info     - informacion del modelo
  GET  /metrics         - metricas (quantiles de latencia, errores)
============================================================================
"""

import logging
import os
import sys
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

sys.path.insert(0, os.path.dirname(__file__))
from modelo import cargar_modelo, predecir
from metricas import get_metrics, health_check, metrics_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nexo Recommender - Microservicio de IA",
    description="API de inferencia del modelo de propension de compra (PyTorch)",
    version="1.0.0",
)

modelo = None
feature_cols = None
metricas_modelo = None
version_modelo = "1.0.0"
inicio_servicio = time.time()


@app.on_event("startup")
async def cargar():
    global modelo, feature_cols, metricas_modelo
    logger.info("Cargando modelo PyTorch...")
    modelo, feature_cols, metricas_modelo = cargar_modelo()
    logger.info(f"Modelo cargado. Features: {feature_cols}")


# =============================================================================
# ESQUEMAS
# =============================================================================
class PredictRequest(BaseModel):
    num_compras_categoria: int = Field(..., ge=0)
    num_vistas_categoria: int = Field(..., ge=0)
    num_clics_categoria: int = Field(..., ge=0)
    total_gastado_categoria: float = Field(..., ge=0)
    dias_ultima_compra: int = Field(..., ge=0)
    frecuencia_compra: int = Field(..., ge=0)
    ticket_promedio: float = Field(..., ge=0)
    edad: int = Field(..., ge=0, le=120)
    acepta_marketing: int = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    prediccion: float
    propension: str
    version_modelo: str
    tiempo_ms: float


class BatchPredictRequest(BaseModel):
    registros: List[PredictRequest]


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/health")
async def health():
    uptime = time.time() - inicio_servicio
    m = get_metrics()
    ok = m["latency"]["status"] == "ok" and m["errors"]["status"] == "ok"
    return {
        "status": "ok" if (modelo is not None and ok) else "degraded",
        "modelo_cargado": modelo is not None,
        "version_modelo": version_modelo if modelo is not None else None,
        "uptime_seconds": round(uptime, 2),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if modelo is None:
        metrics_store.record_error()
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    inicio = time.time()
    try:
        datos = pd.DataFrame([request.dict()])[feature_cols]
        prediccion = float(predecir(modelo, datos)[0])
        tiempo_ms = (time.time() - inicio) * 1000
        metrics_store.record_latency(tiempo_ms)
        metrics_store.record_prediction()
    except Exception as e:
        metrics_store.record_error()
        logger.error(f"Error en prediccion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if prediccion >= 0.7:
        propension = "alta"
    elif prediccion >= 0.4:
        propension = "media"
    else:
        propension = "baja"

    logger.info(f"Prediccion: {prediccion:.4f} ({propension}) en {tiempo_ms:.2f}ms")
    return PredictResponse(
        prediccion=round(prediccion, 4),
        propension=propension,
        version_modelo=version_modelo,
        tiempo_ms=round(tiempo_ms, 2),
    )


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    if modelo is None:
        metrics_store.record_error()
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    inicio = time.time()
    datos = pd.DataFrame([r.dict() for r in request.registros])[feature_cols]
    predicciones = predecir(modelo, datos)
    metrics_store.record_latency((time.time() - inicio) * 1000 / max(len(predicciones), 1))
    return {
        "predicciones": [round(float(p), 4) for p in predicciones],
        "total_registros": len(predicciones),
        "version_modelo": version_modelo,
    }


@app.get("/modelo/info")
async def modelo_info():
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {
        "version": version_modelo,
        "features": feature_cols,
        "metricas": metricas_modelo,
        "arquitectura": {
            "tipo": "Red Neuronal PyTorch",
            "capas_ocultas": [64, 32, 16],
            "dropout": 0.3,
            "salida": "Sigmoid",
        },
    }


@app.get("/metrics")
async def metrics():
    return get_metrics()
