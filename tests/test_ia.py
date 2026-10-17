"""
=============================================================================
TESTS - MICROSERVICIO DE IA
============================================================================
"""
import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient


def _crear_modelo_fake(monkeypatch):
    """Crea un modelo dummy para los tests sin necesitar el .pkl real."""
    import torch.nn as nn

    class Dummy(nn.Module):
        def forward(self, x):
            return torch.sigmoid(torch.ones(x.size(0), 1) * 0.75)

    dummy = Dummy()
    dummy.eval()

    import app as app_module
    monkeypatch.setattr(app_module, "modelo", dummy)
    monkeypatch.setattr(app_module, "feature_cols", [
        "num_vistas_categoria", "num_clics_categoria",
        "dias_ultima_compra", "frecuencia_compra",
        "ticket_promedio", "edad", "acepta_marketing",
    ])
    return dummy


def test_health(monkeypatch):
    import app as app_module
    _crear_modelo_fake(monkeypatch)
    client = TestClient(app_module.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["modelo_cargado"] is True


def test_predict_valido(monkeypatch):
    import app as app_module
    _crear_modelo_fake(monkeypatch)
    client = TestClient(app_module.app)
    payload = {
        "num_compras_categoria": 5, "num_vistas_categoria": 20,
        "num_clics_categoria": 10, "total_gastado_categoria": 150000,
        "dias_ultima_compra": 15, "frecuencia_compra": 10,
        "ticket_promedio": 75000, "edad": 35, "acepta_marketing": 1,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["prediccion"] <= 1
    assert data["propension"] in ["alta", "media", "baja"]


def test_predict_invalido(monkeypatch):
    import app as app_module
    _crear_modelo_fake(monkeypatch)
    client = TestClient(app_module.app)
    payload = {"num_compras_categoria": -1}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_batch(monkeypatch):
    import app as app_module
    _crear_modelo_fake(monkeypatch)
    client = TestClient(app_module.app)
    payload = {"registros": [{
        "num_compras_categoria": 5, "num_vistas_categoria": 20,
        "num_clics_categoria": 10, "total_gastado_categoria": 150000,
        "dias_ultima_compra": 15, "frecuencia_compra": 10,
        "ticket_promedio": 75000, "edad": 35, "acepta_marketing": 1,
    }]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    assert r.json()["total_registros"] == 1


def test_modelo_info(monkeypatch):
    import app as app_module
    _crear_modelo_fake(monkeypatch)
    client = TestClient(app_module.app)
    r = client.get("/modelo/info")
    assert r.status_code == 200
    assert "features" in r.json()
