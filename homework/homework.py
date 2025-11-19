# flake8: noqa: E501
import os
import json
import gzip
import pickle
from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preparar_datos(archivo_zip: str) -> pd.DataFrame:
    """Carga y prepara el dataset eliminando datos inválidos."""
    datos = pd.read_csv(archivo_zip, compression="zip").copy()
    datos.rename(columns={"default payment next month": "default"}, inplace=True)
    
    if "ID" in datos.columns:
        datos.drop(columns=["ID"], inplace=True)

    # Filtrar registros con información no disponible
    datos = datos[(datos["MARRIAGE"] != 0) & (datos["EDUCATION"] != 0)].copy()
    
    # Agrupar niveles educativos superiores a 4 en "others"
    datos["EDUCATION"] = datos["EDUCATION"].apply(lambda val: 4 if val >= 4 else val)
    
    return datos.dropna()


def calcular_metricas_rendimiento(nombre_conjunto: str, valores_reales, predicciones) -> dict:
    """Calcula métricas de rendimiento del modelo."""
    resultado = {
        "type": "metrics",
        "dataset": nombre_conjunto,
        "precision": precision_score(valores_reales, predicciones, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(valores_reales, predicciones),
        "recall": recall_score(valores_reales, predicciones, zero_division=0),
        "f1_score": f1_score(valores_reales, predicciones, zero_division=0),
    }
    return resultado


def generar_matriz_confusion(nombre_conjunto: str, valores_reales, predicciones) -> dict:
    """Genera la matriz de confusión en formato de diccionario."""
    verdaderos_neg, falsos_pos, falsos_neg, verdaderos_pos = confusion_matrix(
        valores_reales, predicciones
    ).ravel()
    
    estructura = {
        "type": "cm_matrix",
        "dataset": nombre_conjunto,
        "true_0": {"predicted_0": int(verdaderos_neg), "predicted_1": int(falsos_pos)},
        "true_1": {"predicted_0": int(falsos_neg), "predicted_1": int(verdaderos_pos)},
    }
    return estructura


def construir_pipeline_optimizacion(columnas_categoricas, columnas_numericas) -> GridSearchCV:
    """Construye el pipeline con búsqueda de hiperparámetros."""
    
    # Transformador para preprocesar columnas categóricas y numéricas
    transformador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), columnas_categoricas),
            ("num", StandardScaler(), columnas_numericas),
        ]
    )

    # Pipeline completo
    flujo = Pipeline(
        steps=[
            ("pre", transformador),
            ("selector", SelectKBest(score_func=f_classif)),
            ("pca", PCA()),
            ("mlp", MLPClassifier(max_iter=15000, random_state=21)),
        ]
    )

    # Espacio de hiperparámetros
    parametros = {
        "selector__k": [20],
        "pca__n_components": [None],
        "mlp__hidden_layer_sizes": [(50, 30, 40, 60)],
        "mlp__alpha": [0.26],
        "mlp__learning_rate_init": [0.001],
    }

    # Búsqueda con validación cruzada
    busqueda = GridSearchCV(
        estimator=flujo,
        param_grid=parametros,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    return busqueda


def main() -> None:
    """Función principal que ejecuta todo el flujo."""
    
    # Rutas de entrada
    archivo_entrenamiento = "files/input/train_data.csv.zip"
    archivo_prueba = "files/input/test_data.csv.zip"

    # Cargar y limpiar datos
    df_entrenamiento = preparar_datos(archivo_entrenamiento)
    df_prueba = preparar_datos(archivo_prueba)

    # Separar características y objetivo
    X_entrenamiento = df_entrenamiento.drop(columns=["default"])
    y_entrenamiento = df_entrenamiento["default"]
    X_prueba = df_prueba.drop(columns=["default"])
    y_prueba = df_prueba["default"]

    # Identificar columnas categóricas y numéricas
    cols_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    cols_numericas = [col for col in X_entrenamiento.columns if col not in cols_categoricas]

    # Construir y entrenar modelo
    optimizador = construir_pipeline_optimizacion(cols_categoricas, cols_numericas)
    optimizador.fit(X_entrenamiento, y_entrenamiento)

    # Limpiar directorio de modelos
    directorio_modelos = Path("files/models")
    if directorio_modelos.exists():
        for archivo in glob(str(directorio_modelos / "*")):
            os.remove(archivo)
        try:
            os.rmdir(directorio_modelos)
        except OSError:
            pass
    directorio_modelos.mkdir(parents=True, exist_ok=True)

    # Guardar modelo comprimido
    with gzip.open(directorio_modelos / "model.pkl.gz", "wb") as archivo_modelo:
        pickle.dump(optimizador, archivo_modelo)

    # Realizar predicciones
    pred_entrenamiento = optimizador.predict(X_entrenamiento)
    pred_prueba = optimizador.predict(X_prueba)

    # Calcular métricas
    metricas_train = calcular_metricas_rendimiento("train", y_entrenamiento, pred_entrenamiento)
    metricas_test = calcular_metricas_rendimiento("test", y_prueba, pred_prueba)
    cm_train = generar_matriz_confusion("train", y_entrenamiento, pred_entrenamiento)
    cm_test = generar_matriz_confusion("test", y_prueba, pred_prueba)

    # Preparar resultados
    resultados = [metricas_train, metricas_test, cm_train, cm_test]

    # Guardar métricas
    directorio_salida = Path("files/output")
    directorio_salida.mkdir(parents=True, exist_ok=True)
    with open(directorio_salida / "metrics.json", "w", encoding="utf-8") as archivo_salida:
        for registro in resultados:
            archivo_salida.write(json.dumps(registro) + "\n")


if __name__ == "__main__":
    main()