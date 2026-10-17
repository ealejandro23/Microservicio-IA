"""
=============================================================================
MODELO DE RECOMENDACION - PROPENSION DE COMPRA (PYTORCH)
============================================================================
Red neuronal para predecir la propension de compra de un usuario en una
categoria de productos.

Arquitectura:
  Input (9 features) -> Dense(64) + ReLU + Dropout(0.3)
                      -> Dense(32) + ReLU + Dropout(0.3)
                      -> Dense(16) + ReLU
                      -> Dense(1) + Sigmoid
============================================================================
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATOS_DIR = os.getenv(
    "DATOS_DIR", os.path.join(os.path.dirname(__file__), "datos", "datos_tratados")
)
MODELO_PATH = os.getenv(
    "MODEL_PATH", os.path.join(os.path.dirname(__file__), "modelo_entrenado.pkl")
)

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_DIMS = [64, 32, 16]
DROPOUT_RATE = 0.3
PATIENCE = 10
UMBRAL_METRICA = 0.40  # F1 minimo para publicar el modelo


class RecommenderNet(nn.Module):
    """Red neuronal de recomendacion."""

    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev = hidden
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.sigmoid(self.output_layer(self.hidden_layers(x)))


def entrenar_modelo(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO DEL MODELO")
    print("=" * 60)

    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val.values)
    y_val_t = torch.FloatTensor(y_val.values).unsqueeze(1)

    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)])

    modelo = RecommenderNet(X_train.shape[1], HIDDEN_DIMS, DROPOUT_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    mejor_f1 = 0.0
    sin_mejora = 0
    mejor_estado = None

    for epoch in range(EPOCHS):
        modelo.train()
        permutation = torch.randperm(X_train_t.size(0))
        for i in range(0, X_train_t.size(0), BATCH_SIZE):
            idx = permutation[i:i + BATCH_SIZE]
            salidas = modelo(X_train_t[idx])
            perdida = criterion(salidas, y_train_t[idx])
            optimizer.zero_grad()
            perdida.backward()
            optimizer.step()

        modelo.eval()
        with torch.no_grad():
            val_salidas = modelo(X_val_t)
            val_perdida = criterion(val_salidas, y_val_t).item()
            pred = (val_salidas >= 0.5).int()
            verd = y_val_t.int()
            tp = ((pred == 1) & (verd == 1)).sum().item()
            fp = ((pred == 1) & (verd == 0)).sum().item()
            fn = ((pred == 0) & (verd == 1)).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        scheduler.step(val_perdida)
        if f1 > mejor_f1:
            mejor_f1 = f1
            sin_mejora = 0
            mejor_estado = modelo.state_dict().copy()
        else:
            sin_mejora += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Val Loss: {val_perdida:.4f} - F1: {f1:.4f}")

        if sin_mejora >= PATIENCE:
            print(f"  Early stopping en epoch {epoch+1}")
            break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    print(f"  Mejor F1 en validacion: {mejor_f1:.4f}")
    return modelo


def evaluar_modelo(modelo, X_test, y_test):
    print("\n" + "=" * 60)
    print("EVALUACION DEL MODELO")
    print("=" * 60)
    modelo.eval()
    with torch.no_grad():
        prob = modelo(torch.FloatTensor(X_test.values)).numpy().flatten()
    pred = (prob >= 0.5).astype(int)
    verd = y_test.values

    metricas = {
        "accuracy": float(accuracy_score(verd, pred)),
        "precision": float(precision_score(verd, pred, zero_division=0)),
        "recall": float(recall_score(verd, pred, zero_division=0)),
        "f1_score": float(f1_score(verd, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(verd, prob)),
    }
    for k, v in metricas.items():
        print(f"  {k}: {v:.4f}")
    return metricas


def guardar_modelo(modelo, feature_cols, metricas, scaler=None):
    joblib.dump(
        {
            "state_dict": modelo.state_dict(),
            "feature_cols": feature_cols,
            "metricas": metricas,
            "arquitectura": {
                "input_dim": modelo.input_dim,
                "hidden_dims": modelo.hidden_dims,
                "dropout_rate": modelo.dropout_rate,
            },
            "scaler_mean": scaler.mean_.tolist() if scaler is not None else None,
            "scaler_scale": scaler.scale_.tolist() if scaler is not None else None,
            "version": "1.0.0",
        },
        MODELO_PATH,
    )
    print(f"\n  Modelo guardado en: {MODELO_PATH}")


def cargar_modelo():
    if not os.path.exists(MODELO_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en {MODELO_PATH}")
    data = joblib.load(MODELO_PATH)
    arq = data["arquitectura"]
    modelo = RecommenderNet(arq["input_dim"], arq["hidden_dims"], arq["dropout_rate"])
    modelo.load_state_dict(data["state_dict"])
    modelo.eval()
    scaler = None
    if data.get("scaler_mean") is not None:
        scaler = StandardScaler()
        scaler.mean_ = np.array(data["scaler_mean"])
        scaler.scale_ = np.array(data["scaler_scale"])
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(data["feature_cols"])
        scaler.feature_names_in_ = np.array(data["feature_cols"])
    return modelo, data["feature_cols"], data["metricas"], scaler


TEMPERATURA = 5.0  # suaviza probabilidades moderadamente para variación real


def predecir(modelo, datos, scaler=None):
    modelo.eval()
    if scaler is not None:
        datos = pd.DataFrame(
            scaler.transform(datos), columns=datos.columns, index=datos.index
        )
    with torch.no_grad():
        logits = modelo.hidden_layers(torch.FloatTensor(datos.values))
        logits = modelo.output_layer(logits)
        probs = torch.sigmoid(logits / TEMPERATURA).numpy().flatten()
    return probs


def main():
    print("=" * 60)
    print("ENTRENAMIENTO - MICROSERVICIO DE IA")
    print("=" * 60)

    X_train = pd.read_csv(os.path.join(DATOS_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATOS_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATOS_DIR, "y_train.csv")).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(DATOS_DIR, "y_test.csv")).iloc[:, 0]
    with open(os.path.join(DATOS_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    # Estandarizar features (media 0, varianza 1)
    scaler = StandardScaler()
    scaler.fit(X_train[feature_cols])
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train[feature_cols]), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[feature_cols]), columns=feature_cols, index=X_test.index
    )

    modelo = entrenar_modelo(X_train_scaled, y_train, X_test_scaled, y_test)
    metricas = evaluar_modelo(modelo, X_test_scaled, y_test)

    if metricas["f1_score"] < UMBRAL_METRICA:
        raise SystemExit(
            f"F1 {metricas['f1_score']:.4f} menor al umbral {UMBRAL_METRICA}"
        )

    guardar_modelo(modelo, feature_cols, metricas, scaler)
    print("\nENTRENAMIENTO COMPLETADO")


if __name__ == "__main__":
    main()
