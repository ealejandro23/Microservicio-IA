"""
=============================================================================
TRATAMIENTO DE DATOS - MICROSERVICIO DE IA
============================================================================
Limpia los datos crudos del caso Nexus Retail y genera las variables de
entrada del modelo de recomendacion (propension de compra).

Rutas (relativas al script):
  - Entrada:  datos/crudos/*.csv
  - Salida:   datos/datos_tratados/*.csv
============================================================================
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "crudos"))
OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "datos_tratados")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. PERFILADO
# =============================================================================
def perfilado(df, nombre):
    return {
        "archivo": nombre,
        "filas": len(df),
        "columnas": len(df.columns),
        "nulos": df.isnull().sum().to_dict(),
        "duplicados": int(df.duplicated().sum()),
    }


def imprimir_perfilado(reporte, etapa="ANTES"):
    print(f"--- PERFILADO {etapa}: {reporte['archivo']} ---")
    print(f"  Filas: {reporte['filas']}  Columnas: {reporte['columnas']}")
    print(f"  Duplicados: {reporte['duplicados']}")
    print(f"  Nulos: {reporte['nulos']}")


# =============================================================================
# 2. LIMPIEZA
# =============================================================================
def limpiar_productos(df):
    df = df.copy()
    filas_antes = len(df)
    df["categoria"] = df["categoria"].str.lower().str.strip()
    df["precio_lista"] = (
        df["precio_lista"].astype(str).str.replace("$", "", regex=False)
    )
    df["precio_lista"] = pd.to_numeric(df["precio_lista"], errors="coerce")
    df["activo"] = (
        df["activo"].astype(str).str.lower()
        .map({"si": True, "true": True, "1": True, "0": False, "no": False})
        .fillna(False)
    )
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)
    df.loc[df["stock"] < 0, "stock"] = 0
    df["costo"] = pd.to_numeric(df["costo"], errors="coerce")
    df["costo"] = df["costo"].fillna(df["costo"].median())
    df = df.drop_duplicates(subset=["sku"], keep="last")
    print(f"  [LIMPIEZA] productos: {filas_antes} -> {len(df)} filas")
    return df


def limpiar_transacciones(df):
    df = df.copy()
    filas_antes = len(df)
    df["precio_unitario"] = pd.to_numeric(df["precio_unitario"], errors="coerce")
    df["descuento"] = df["descuento"].fillna(0)
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce").fillna(1)
    df["es_devolucion"] = df["cantidad"] < 0
    df["cantidad"] = df["cantidad"].abs()
    df["total"] = df["cantidad"] * df["precio_unitario"] - df["descuento"]
    df["es_invitado"] = df["usuario_id"] == "invitado"
    estados_validos = ["pagado", "completado", "entregado"]
    df = df[df["estado"].isin(estados_validos)]
    print(f"  [LIMPIEZA] transacciones: {filas_antes} -> {len(df)} filas")
    return df


def limpiar_usuarios(df):
    df = df.copy()
    filas_antes = len(df)
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df.loc[(df["edad"] < 0) | (df["edad"] > 120), "edad"] = np.nan
    df["edad"] = df["edad"].fillna(df["edad"].median()).astype(int)
    df["region"] = df["region"].str.lower().str.strip()
    df["email_valido"] = (
        df["email_valido"].astype(str).str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )
    df["acepta_marketing"] = df["acepta_marketing"].fillna(0).astype(int)
    df = df.drop_duplicates(subset=["usuario_id"], keep="first")
    print(f"  [LIMPIEZA] usuarios: {filas_antes} -> {len(df)} filas")
    return df


def limpiar_clickstream(df):
    df = df.copy()
    filas_antes = len(df)

    def parse_timestamp(x):
        try:
            return pd.to_datetime(x, utc=True)
        except Exception:
            return pd.NaT

    df["timestamp"] = df["timestamp"].map(parse_timestamp)
    df = df.dropna(subset=["timestamp"])
    df["es_visitante"] = df["usuario_id"].isnull()
    df["tipo_evento"] = df["tipo_evento"].str.lower().str.strip()
    df = df.drop_duplicates()
    print(f"  [LIMPIEZA] clickstream: {filas_antes} -> {len(df)} filas")
    return df


# =============================================================================
# 3. INTEGRACION Y VARIABLES DEL MODELO
# =============================================================================
def crear_variables_modelo(transacciones, clickstream, productos, usuarios):
    """Propension de compra por usuario y categoria (variable objetivo binaria)."""
    print("--- Creacion de variables del modelo ---")

    transacciones = transacciones.merge(
        productos[["sku", "categoria"]], on="sku", how="left"
    ).dropna(subset=["categoria"])

    clickstream = clickstream.merge(
        productos[["sku", "categoria"]], on="sku", how="left"
    ).dropna(subset=["categoria"])

    agg_trans = transacciones.groupby(["usuario_id", "categoria"]).agg(
        num_compras_categoria=("pedido_id", "count"),
        total_gastado_categoria=("total", "sum"),
        fecha_ultima_compra=("fecha", "max"),
    ).reset_index()

    agg_click = clickstream.groupby(["usuario_id", "categoria"]).agg(
        num_vistas_categoria=("tipo_evento", lambda x: (x == "vista").sum()),
        num_clics_categoria=("tipo_evento", lambda x: (x == "clic").sum()),
    ).reset_index()

    variables = agg_trans.merge(
        agg_click, on=["usuario_id", "categoria"], how="outer"
    ).fillna(0)

    def parse_fecha(x):
        try:
            dt = pd.to_datetime(x, dayfirst=True, utc=True)
            return dt.tz_localize(None) if dt.tz else dt
        except Exception:
            return pd.NaT

    variables["fecha_ultima_compra"] = variables["fecha_ultima_compra"].apply(parse_fecha)
    variables["dias_ultima_compra"] = (
        pd.Timestamp.now(tz=None) - variables["fecha_ultima_compra"]
    ).dt.days
    variables["dias_ultima_compra"] = variables["dias_ultima_compra"].fillna(999).astype(int)

    user_stats = transacciones.groupby("usuario_id").agg(
        frecuencia_compra=("pedido_id", "count"),
        ticket_promedio=("total", "mean"),
    ).reset_index()
    variables = variables.merge(user_stats, on="usuario_id", how="left").fillna(0)

    variables = variables.merge(
        usuarios[["usuario_id", "edad", "acepta_marketing"]], on="usuario_id", how="left"
    )
    variables["edad"] = variables["edad"].fillna(variables["edad"].median())
    variables["acepta_marketing"] = variables["acepta_marketing"].fillna(0)
    variables["propension_compra"] = (variables["num_compras_categoria"] > 0).astype(int)

    print(f"  Registros: {len(variables)}")
    print(f"  Objetivo: {variables['propension_compra'].value_counts().to_dict()}")
    return variables


# =============================================================================
# 4. SEPARACION TRAIN/TEST
# =============================================================================
FEATURE_COLS = [
    "num_compras_categoria", "num_vistas_categoria", "num_clics_categoria",
    "total_gastado_categoria", "dias_ultima_compra", "frecuencia_compra",
    "ticket_promedio", "edad", "acepta_marketing",
]


def separar_train_test(variables):
    X = variables[FEATURE_COLS]
    y = variables["propension_compra"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# 5. SALIDAS
# =============================================================================
def guardar_resultados(productos, transacciones, usuarios, clickstream, variables,
                       X_train, X_test, y_train, y_test):
    productos.to_csv(os.path.join(OUTPUT_DIR, "productos_limpios.csv"), index=False)
    transacciones.to_csv(os.path.join(OUTPUT_DIR, "transacciones_limpias.csv"), index=False)
    usuarios.to_csv(os.path.join(OUTPUT_DIR, "usuarios_limpios.csv"), index=False)
    clickstream.to_csv(os.path.join(OUTPUT_DIR, "clickstream_limpio.csv"), index=False)
    variables.to_csv(os.path.join(OUTPUT_DIR, "variables_modelo.csv"), index=False)
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "feature_cols.json"), "w") as f:
        json.dump(FEATURE_COLS, f)
    print(f"  Datos tratados en: {OUTPUT_DIR}")


# =============================================================================
# PIPELINE
# =============================================================================
def main():
    print("=" * 60)
    print("PIPELINE DE DATOS - MICROSERVICIO DE IA")
    print("=" * 60)

    print("\nETAPA 1: PERFILADO")
    productos = pd.read_csv(os.path.join(DATA_DIR, "productos.csv"))
    transacciones = pd.read_csv(os.path.join(DATA_DIR, "transacciones.csv"))
    usuarios = pd.read_csv(os.path.join(DATA_DIR, "usuarios.csv"))
    clickstream = pd.read_csv(os.path.join(DATA_DIR, "clickstream.csv"))
    for df, nombre in [(productos, "productos"), (transacciones, "transacciones"),
                       (usuarios, "usuarios"), (clickstream, "clickstream")]:
        imprimir_perfilado(perfilado(df, nombre), "ANTES")

    print("\nETAPA 2: LIMPIEZA")
    productos = limpiar_productos(productos)
    transacciones = limpiar_transacciones(transacciones)
    usuarios = limpiar_usuarios(usuarios)
    clickstream = limpiar_clickstream(clickstream)

    print("\nETAPA 3: VARIABLES DEL MODELO")
    variables = crear_variables_modelo(transacciones, clickstream, productos, usuarios)

    print("\nETAPA 4: SEPARACION TRAIN/TEST")
    X_train, X_test, y_train, y_test = separar_train_test(variables)

    print("\nETAPA 5: SALIDAS")
    guardar_resultados(productos, transacciones, usuarios, clickstream, variables,
                       X_train, X_test, y_train, y_test)
    print("\nTRATAMIENTO COMPLETADO")


if __name__ == "__main__":
    main()
