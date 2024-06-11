# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os # Operaciones de sistema

"""# BASE DE DATOS"""

import pandas as pd
df = pd.read_csv('/content/drive/My Drive/bd/linkedin_job_postings.csv')

# Verificamos el nuevo DataFrame
print(df.head())

df.info()

print(df.isnull().sum())

"""# **DEPURACION DE DATOS**"""

total_registros = len(df)
print(f"El total de registros en el DataFrame es: {total_registros}")

df.columns

"""Existen demasiadas columnas que no aportan a nuestra prediccion por lo que se procede a eliminar."""

# Eliminar las columnas que no vamos a usar
columnas_eliminar = ['company', 'job_location', 'job_link', 'last_processed_time', 'got_summary', 'got_ner','is_being_worked', 'first_seen']
df = df.drop(columnas_eliminar, axis=1)

# Verificar que las columnas se hayan eliminado correctamente
print(df.columns.tolist())

df[df.duplicated()]

duplicates_exist = df.duplicated().any()
print("¿Existen valores duplicados en general?", duplicates_exist)

"""Confirmamos los duplicados realizando el conteo"""

duplicates_count = df.duplicated().sum()
print("Recuento de valores duplicados en el dataset:", duplicates_count)

# Eliminar filas duplicadas del DataFrame original
df.drop_duplicates(inplace=True)

duplicates_count = df.duplicated().sum()
print("Recuento de valores duplicados en el dataset:", duplicates_count)

df.describe()

"""Tomando los primeros 10 puestos de trabajo de los datos"""

s = df['job_title'].value_counts().head(10)
print (s)

t = df['job_type'].value_counts().head(10)
print (t)

"""# Preprocesamiento de los datos

Descargamos recursos adicionales y bibliotecas que son necesarias para el preprocesamiento y la optimización del modelo.
"""

import pandas as pd  # Importar pandas para el manejo de datos en forma de DataFrames
import numpy as np   # Importar numpy para operaciones numéricas y manejo de arrays
import time          # Importar time para medir el tiempo de ejecución

# Importar herramientas para la transformación de texto a vectores
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# Importar funciones para dividir el conjunto de datos y validación cruzada
from sklearn.model_selection import train_test_split, cross_val_score
# Importar herramientas para escalar características numéricas
from sklearn.preprocessing import MinMaxScaler
# Importar el modelo de regresión logística para clasificación
from sklearn.linear_model import LogisticRegression
# Importar métricas para evaluación del modelo
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
# Importar modelo Naive Bayes para clasificación
from sklearn.naive_bayes import MultinomialNB
# Importar modelos de ensamble: Random Forest y Gradient Boosting
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Importar modelo K-Nearest Neighbors para clasificación
from sklearn.neighbors import KNeighborsClassifier
# Importar herramientas para el procesamiento de texto
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
# Importar herramientas para el procesamiento de texto usando expresiones regulares
import re
# Importar tqdm para mostrar una barra de progreso durante operaciones largas
from tqdm import tqdm
# Importar matplotlib y seaborn para visualizaciones
import matplotlib.pyplot as plt
import seaborn as sns

"""nltk.download('punkt'): Descarga el tokenizer de Punkt de NLTK, que se usa para la tokenización de texto.

nltk.download('stopwords'): Descarga la lista de palabras de parada (stopwords) de NLTK, que se usará para eliminar palabras comunes que no aportan significado en el análisis de texto.

!pip install optuna: Instala la biblioteca Optuna, que es una biblioteca de optimización de hiperparámetros.

import optuna: Importa la biblioteca Optuna para usarla posteriormente en la optimización de hiperparámetros.
"""

nltk.download('punkt')
nltk.download('stopwords')
!pip install optuna
import optuna

"""Aquí importamos la función word_tokenize de la biblioteca nltk. Esta función se utiliza para dividir el texto en tokens o palabras individuales.

Importamos la lista de stopwords en inglés de nltk y la almacenamos en un conjunto (set). Las stopwords son palabras comunes que suelen ser filtradas antes o después del procesamiento de texto porque generalmente no aportan significado relevante para el análisis.

Creamos una instancia del Stemmer de Porter. El stemming es un proceso que reduce las palabras a su raíz o stem. Por ejemplo, "running" se convierte en "run".

Luego definimos una función preprocess_text que toma un texto como entrada y realiza las siguientes operaciones:

Tokenización: Divide el texto en palabras individuales o tokens.

Eliminación de Stopwords y Stemming:
Convierte cada palabra a minúsculas.

*   Elemento de lista
*   Elemento de lista


Verifica que la palabra sea alfabética.
Elimina las stopwords.
Realiza el stemming en las palabras restantes.

**Analizamos graficamente los datos actuales del año 2023 a 2024**
"""

fig, ax = plt.subplots(figsize=(15, 10))
c = ['#004764','#0072A0','#00ABF0', '#19BDFF', '#37C6FF','#55CEFF','#73D7FF','#91E0FF','#A5E5FF','#B9EBFF']

plt.pie(s, autopct = '%1.0f%%',colors = c)

ax.legend(df.job_title,
          title='Title',
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1)
)
ax.set_title('Títulos de trabajo más comunes en Linkedin', fontfamily='serif', fontsize=15,fontweight='bold')

# Show Plot
plt.show()

# Configurar la visualización
sns.set(style="whitegrid")
plt.style.use('ggplot')

# Establecer el tamaño de la figura
plt.figure(figsize=(18, 8))

# Elegir una paleta de colores diferente (¡la elegancia es clave!)
colors = sns.color_palette('husl', len(df['job_title'].unique()))

# Obtener los 10 principales títulos de trabajo
top_job_titles = df['job_title'].value_counts().head(10)

# Crear un gráfico de barras con los nuevos colores
top_job_titles.plot(kind='bar', color=colors, edgecolor='black')

# Personalizar el gráfico
plt.title('Top 10 Títulos de Trabajo', fontsize=16)
plt.xlabel('Títulos de Trabajo', fontsize=14)
plt.ylabel('Número de Listados de Trabajo', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

"""# **Predicción de habilidades en demanda**

Queremos predecir en qué industria se ubicarán los trabajos en función de ciertas características.

La predicción de habilidades en demanda es un área crucial en el ámbito del reclutamiento y la gestión del talento, ya que permite a empresas y profesionales anticipar las habilidades necesarias para satisfacer las demandas del mercado laboral en constante evolución.

## Comparar varios modelos los mas adecuados utilizando scikit-learn

'Random Forest' RandomForestRegressor
>
'Linear Regression' LinearRegression
>
'Support Vector Machine' SVR

Seleccionamos estos tres modelos porque representan una variedad de enfoques de modelado que pueden ser efectivos para problemas de predicción en diferentes situaciones:

Random Forest (RandomForestRegressor):

RandomForestRegressor es un modelo de aprendizaje automático basado en árboles de decisión, que combina múltiples árboles de decisión para realizar predicciones. Es robusto, capaz de manejar conjuntos de datos grandes con muchas características, y es menos propenso al sobreajuste en comparación con un solo árbol de decisión.
Es útil cuando se esperan relaciones no lineales entre las características y la variable objetivo, y cuando se tienen características categóricas o numéricas.
RandomForestRegressor también puede manejar datos faltantes sin necesidad de imputación adicional.

Linear Regression (LinearRegression):

LinearRegression es un modelo lineal que busca establecer una relación lineal entre las características y la variable objetivo. Es simple y fácil de interpretar, y puede proporcionar buenos resultados cuando la relación entre las características y la variable objetivo es aproximadamente lineal.
Es útil para problemas donde se espera una relación lineal entre las características y la variable objetivo, y cuando se desea interpretabilidad del modelo.

Support Vector Machine (SVR):

SVR es una variante de las Máquinas de Soporte Vectorial adaptada para problemas de regresión. Busca encontrar el hiperplano que mejor se ajusta a los datos, maximizando el margen entre las instancias más cercanas.
Es útil cuando se tienen conjuntos de datos pequeños a medianos y se espera que haya una relación no lineal entre las características y la variable objetivo. SVR también puede ser eficaz en espacios de alta dimensión.
SVR es robusto frente a la presencia de datos atípicos debido a su enfoque en maximizar el margen.

**por las que podríamos elegir RandomForestRegressor y GradientBoostingClassifier:**

Flexibilidad en la relación entre características y respuesta: Random Forest y Gradient Boosting son modelos no paramétricos, lo que significa que no hacen suposiciones específicas sobre la forma funcional de la relación entre las características y la respuesta. Esto les permite manejar relaciones más complejas que los modelos lineales como Regresión Lineal.

Manejo de características no lineales y de interacciones: Estos modelos pueden capturar relaciones no lineales y de interacción entre las características de manera efectiva, lo que los hace más adecuados cuando la relación entre las características y la respuesta no es lineal.

Robustez frente a overfitting: Random Forest y Gradient Boosting tienen la capacidad de manejar el overfitting mejor que algunos modelos más simples como Regresión Lineal. Esto se debe a la técnica de ensemble learning que utilizan, donde combinan múltiples árboles de decisión débiles para formar un modelo más robusto y generalizable.

Adaptabilidad a diferentes tipos de datos: Estos modelos pueden manejar tanto características numéricas como categóricas, así como datos desbalanceados y presencia de valores atípicos, lo que los hace más versátiles en una variedad de situaciones.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

"""**train_test_split de sklearn.model_selection:** Esta función se utiliza para dividir los datos en conjuntos de entrenamiento y prueba.

**RandomForestRegressor de sklearn.ensemble:** Este es un algoritmo de aprendizaje automático que se utiliza para realizar regresión utilizando un conjunto de árboles de decisión.

**LinearRegression de sklearn.linear_model:** Este es un modelo de regresión lineal que se utiliza para ajustar un modelo lineal a los datos de entrenamiento.

**SVR de sklearn.svm:** Este es un modelo de regresión de vectores de soporte que se utiliza para ajustar una función de regresión a los datos de entrenamiento.

**mean_squared_error y r2_score de sklearn.metrics:** Estas funciones se utilizan para evaluar el rendimiento de los modelos de regresión. La primera calcula el error cuadrático medio entre las predicciones y los valores reales, mientras que la segunda calcula el coeficiente de determinación (R^2), que es una medida de la proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes.
"""

from sklearn.preprocessing import OneHotEncoder

# Codificar la columna 'job_title'
job_title_encoder = OneHotEncoder(handle_unknown='ignore')
X = job_title_encoder.fit_transform(df[['job_title']])

# Verificar la forma de la matriz codificada
print("Forma de la matriz codificada:", X.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

# Inicializar el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Especifica el número máximo de características deseado

# Aplicar el vectorizador TF-IDF a la columna 'job_title'
job_title_tfidf = tfidf_vectorizer.fit_transform(df['job_title'])

# Verificar la forma de la matriz TF-IDF
print("Forma de la matriz TF-IDF:", job_title_tfidf.shape)

"""Matriz TF-IDF (Term Frequency-Inverse Document Frequency): Esta matriz representa la frecuencia de cada palabra (término) en cada documento (en este caso, en cada título de trabajo). TF-IDF es una medida estadística que evalúa la importancia de una palabra en un documento dentro de un corpus más grande. Esta matriz se utiliza para capturar las características textuales de los títulos de trabajo de una manera que sea más representativa de su contenido."""

from sklearn.decomposition import TruncatedSVD

# Reducción de dimensionalidad con TruncatedSVD
svd = TruncatedSVD(n_components=100)  # Especifica el número deseado de componentes principales
X_svd = svd.fit_transform(job_title_tfidf)

# Verificar la forma de la matriz reducida
print("Forma de la matriz reducida:", X_svd.shape)

"""Matriz Reducida: Después de aplicar la técnica de reducción de dimensionalidad (en este caso, TF-IDF), obtenemos una matriz reducida que conserva la mayor parte de la variabilidad de los datos originales pero con un menor número de dimensiones. En este caso, redujimos la dimensionalidad de la matriz TF-IDF de 1000 características a solo 100 características. Esto se hace para simplificar el modelo y reducir el tiempo de entrenamiento, manteniendo al mismo tiempo la información más relevante de los datos originales.

normalizacion de la data
"""

from sklearn.preprocessing import StandardScaler

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_svd)

# Definir las características (X) y las etiquetas (y)
X = X_svd  # Utilizamos las características reducidas
y = df['job_type'].values  # Etiquetas

# Dividir el conjunto de datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dividir el conjunto de prueba en prueba y validación (20% prueba, 10% validación)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=1/3, random_state=42)

print("Forma del conjunto de entrenamiento:", X_train.shape, y_train.shape)
print("Forma del conjunto de prueba:", X_test.shape, y_test.shape)
print("Forma del conjunto de validación:", X_val.shape, y_val.shape)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Inicializar el clasificador Random Forest
rf_classifier = RandomForestClassifier()

# Entrenar el modelo Random Forest en el conjunto de entrenamiento
rf_classifier.fit(X_train, y_train)

# Evaluar el rendimiento del modelo Random Forest
print("Evaluación del modelo Random Forest:")
print("Precisión en el conjunto de entrenamiento:", rf_classifier.score(X_train, y_train))
print("Precisión en el conjunto de prueba:", rf_classifier.score(X_test, y_test))
print("Precisión en el conjunto de validación:", rf_classifier.score(X_val, y_val))
print()

# Inicializar el clasificador SVM
svm_classifier = SVC()

# Entrenar el modelo SVM en el conjunto de entrenamiento
svm_classifier.fit(X_train, y_train)

# Evaluar el rendimiento del modelo SVM
print("Evaluación del modelo SVM:")
print("Precisión en el conjunto de entrenamiento:", svm_classifier.score(X_train, y_train))
print("Precisión en el conjunto de prueba:", svm_classifier.score(X_test, y_test))
print("Precisión en el conjunto de validación:", svm_classifier.score(X_val, y_val))
print()

# Función para evaluar el rendimiento del modelo
def evaluate_model(model, X, y):
    # Predicciones en el conjunto de prueba
    y_pred = model.predict(X)

    # Exactitud del modelo
    accuracy = accuracy_score(y, y_pred)
    print("Exactitud del modelo:", accuracy)

    # Informe de clasificación
    print("\nInforme de clasificación:")
    print(classification_report(y, y_pred))

# Evaluación del modelo GradientBoosting
#print("Evaluación del modelo GradientBoosting:")
#evaluate_model(gb_classifier, X_test, y_test)

# Evaluación del modelo Random Forest
print("\nEvaluación del modelo Random Forest:")
evaluate_model(rf_classifier, X_test, y_test)

# Evaluación del modelo SVM
print("\nEvaluación del modelo SVM:")
evaluate_model(svm_classifier, X_test, y_test)

import matplotlib.pyplot as plt

# Función para plotear las métricas de evaluación
def plot_metrics(models, metrics, metric_names):
    num_models = len(models)
    num_metrics = len(metrics)

    plt.figure(figsize=(10, 6))

    for i in range(num_models):
        model = models[i]
        model_name = model.__class__.__name__
        x = range(num_metrics)
        y = [metric(model, X_test, y_test) for metric in metrics]
        plt.bar([m + i * 0.2 for m in x], y, width=0.2, align='center', label=model_name)

    plt.xlabel('Métricas')
    plt.ylabel('Valor')
    plt.title('Métricas de Evaluación de los Modelos')
    plt.xticks([m + 0.2 for m in x], metric_names)
    plt.legend()
    plt.show()

# Definir los modelos a comparar
models = [gb_classifier, rf_classifier, svm_classifier]

# Definir las métricas de evaluación
metrics = [accuracy_score, precision_score, recall_score, f1_score]
metric_names = ['Exactitud', 'Precisión', 'Recuperación', 'Puntaje F1']

# Plotear las métricas de evaluación
plot_metrics(models, metrics, metric_names)

# Obtener las predicciones del modelo en el conjunto de prueba
y_pred = gb_classifier.predict(X_test)

# Comparar las predicciones con las etiquetas verdaderas
incorrect_indices = y_test != y_pred
incorrect_predictions = X_test[incorrect_indices]
true_labels = y_test[incorrect_indices]
predicted_labels = y_pred[incorrect_indices]

# Analizar las características de las instancias mal clasificadas
for i, (true_label, predicted_label) in enumerate(zip(true_labels, predicted_labels)):
    print("Instancia mal clasificada #{}:".format(i + 1))
    print("  - Características:", incorrect_predictions[i])
    print("  - Etiqueta verdadera:", true_label)
    print("  - Etiqueta predicha:", predicted_label)
    print()

from sklearn.model_selection import GridSearchCV

# Definir la cuadrícula de hiperparámetros a explorar
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}

# Inicializar el clasificador GradientBoosting
gb_classifier = GradientBoostingClassifier()

# Realizar la búsqueda grid
grid_search = GridSearchCV(estimator=gb_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros:", grid_search.best_params_)

# Evaluar el modelo con los mejores hiperparámetros en el conjunto de prueba
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Precisión en el conjunto de prueba con los mejores hiperparámetros:", test_accuracy)

# Utilizar el mejor modelo para hacer predicciones sobre los empleos para el año 2025
predictions_2025 = best_model.predict(X_2025)