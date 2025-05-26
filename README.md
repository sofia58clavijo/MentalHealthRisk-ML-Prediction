

---

# Predicción del Riesgo de Salud Mental

Este proyecto tiene como objetivo desarrollar y comparar modelos de machine learning supervisado para predecir el riesgo de salud mental ('mental_health_risk') basándose en diversos factores relacionados con el estilo de vida, el entorno laboral y la historia de salud.

---

## Propósito

El objetivo principal de este proyecto es:

* **Explorar y preprocesar** un dataset de factores que influyen en el riesgo de salud mental.
* **Implementar y optimizar** dos algoritmos de clasificación, **Random Forest Classifier** y **Logistic Regression**, para predecir el riesgo de salud mental (una variable categórica).
* Utilizar técnicas de **búsqueda de hiperparámetros** (`GridSearchCV`) para encontrar la configuración óptima para cada modelo.
* Analizar las **curvas de aprendizaje** de ambos modelos para entender su rendimiento con diferentes tamaños de datos.
* Realizar un **análisis de concordancia** entre las predicciones de los dos modelos para identificar similitudes y diferencias en sus comportamientos, y evaluar su complementariedad.

---

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```
.
├── data/
│   ├── raw/
│   │   └── mental_health_dataset.csv     # Dataset original
│   └── processed/
│       ├── X_processed.csv                 # Características (X) preprocesadas
│       └── y_processed.csv                 # Variable objetivo (y) preprocesada
├── models/
│   ├── best_logistic_regression_model.joblib # Modelo de Regresión Logística entrenado y guardado
│   ├── best_random_forest_model.joblib   # Modelo de Random Forest entrenado y guardado
│   ├── label_encoder_target.joblib       # LabelEncoder para la variable objetivo
│   └── preprocessor.joblib               # Objeto ColumnTransformer para preprocesamiento de datos
├── notebooks/
│   ├── 01- exploración.ipynb             # Cuaderno para el Análisis Exploratorio de Datos (EDA)
│   ├── 02- preprocesado.ipynb            # Cuaderno para la limpieza y preprocesamiento de datos
│   ├── 03- modelo RandomForest.ipynb     # Cuaderno para la implementación y optimización del modelo Random Forest
│   ├── 04_Model_LogisticRegression.ipynb # Cuaderno para la implementación y optimización del modelo Regresión Logística
│   └── 05- comparacion.ipynb             # Cuaderno para la comparación y análisis de concordancia de los modelos
└── requirements.txt                      # Listado de librerías Python necesarias
```

---

## Flujo de Trabajo

1.  **`01- exploración.ipynb`**: Realiza un análisis exploratorio de datos (EDA) para entender la estructura, distribución y relaciones de las variables en el dataset original.
2.  **`02- preprocesado.ipynb`**: Se encarga de la limpieza de datos, imputación de valores faltantes y la creación del `ColumnTransformer` para el preprocesamiento de características numéricas y categóricas. Aquí se generan y guardan los archivos `X_processed.csv`, `y_processed.csv`, `preprocessor.joblib` y `label_encoder_target.joblib`.
3.  **`03- modelo RandomForest.ipynb`**: Carga los datos y el preprocesador. Implementa y optimiza un modelo de **Random Forest Classifier** usando `GridSearchCV` para encontrar los mejores hiperparámetros. Genera curvas de aprendizaje y guarda el modelo optimizado.
4.  **`04_Model_LogisticRegression.ipynb`**: Similar al notebook anterior, pero enfocado en la **Regresión Logística**. Carga los mismos datos y preprocesador, optimiza el modelo, genera curvas de aprendizaje y guarda el modelo final.
5.  **`05- comparacion.ipynb`**: Carga los dos modelos optimizados. Realiza predicciones en un conjunto de prueba independiente y compara el rendimiento de ambos modelos. Incluye un **análisis de concordancia** para entender dónde coinciden y difieren sus predicciones.

---

## Información del Dataset

El dataset utilizado en este proyecto es `mental_health_dataset.csv`. Contiene información sobre diversos factores que pueden influir en el riesgo de salud mental.

**Origen de la Información:**

https://www.kaggle.com/datasets/mahdimashayekhi/mental-health/data 

---

## Cómo Ejecutar el Proyecto

1.  **Clona este repositorio:**

2.  **Crea un entorno virtual e instala las dependencias:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # En Windows
    # source venv/bin/activate  # En Linux/macOS
    pip install -r requirements.txt
    ```
3.  **Asegúrate de que el archivo `mental_health_dataset.csv` esté en la carpeta `data/raw/`.**
4.  **Ejecuta los notebooks en orden:** Abre los cuadernos Jupyter en tu entorno (`jupyter notebook`) y ejecuta las celdas de cada notebook en el siguiente orden:
    * `01- exploración.ipynb`
    * `02- preprocesado.ipynb`
    * `03- modelo RandomForest.ipynb`
    * `04_Model_LogisticRegression.ipynb`
    * `05- comparacion.ipynb`

---