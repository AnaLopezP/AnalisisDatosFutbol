import pandas as pd
from pyspark.sql.functions import col, desc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import os

def calcular_probabilidades(d_uefa):
    # Calcular las probabilidades de ganar, empatar y perder
    d_uefa['Prob_ganar'] = d_uefa['Partidos_Gan'] / d_uefa['Partidos_Jug']
    d_uefa['Prob_empatar'] = d_uefa['Partidos_Empat'] / d_uefa['Partidos_Jug']
    d_uefa['Prob_perder'] = d_uefa['Partidos_perd'] / d_uefa['Partidos_Jug']
    
    return d_uefa

def calcular_probabilidad_goles(data):
    # Calcular la probabilidad de marcar un gol
    data['Probabilidad_marcar_gol'] = data['Goles_favor'] / data['Partidos_Jug']

    # Calcular la probabilidad de recibir un gol
    data['Probabilidad_recibir_gol'] = data['Goles_contra'] / data['Partidos_Jug']

    return data


# cargo el csv
d_uefa_ruta = os.path.join(os.path.dirname(__file__), 'datos_uefa_mejorados.csv')
d_uefa = pd.read_csv(d_uefa_ruta, delimiter=',')

# Calculo las probabilidades
d_uefa_prob = calcular_probabilidades(d_uefa)
d_uefa_prob = calcular_probabilidad_goles(d_uefa_prob)

# Guardo el csv con las nuevas columnas
d_uefa_prob.to_csv(os.path.join(os.path.dirname(__file__), 'datos_uefa_mejorados.csv'), index=False)

# Mostrar resumen estadístico de las columnas numéricas
print(d_uefa.describe())

# Calcular el total de títulos por país
d_uefa_grouped = d_uefa.groupby("Pais").sum("Titulos")
print(d_uefa_grouped)

# Calcular el promedio de goles a favor y en contra por país
d_uefa_avg_goals = d_uefa.groupby("Pais").agg({"Goles_favor": "sum", "Goles_contra": "sum"})
print(d_uefa_avg_goals)

# Calcular la correlación entre las características numéricas
numeric_columns = d_uefa.select_dtypes(include=['float64', 'int64'])
correlation = numeric_columns.corr()
print(correlation)

# Creo una sesión de Spark
spark = SparkSession.builder.appName('uefa').getOrCreate()

# leo el csv con Spark
d_uefa_rute = os.path.join(os.path.dirname(__file__), 'datos_uefa_mejorados.csv')
d_uefasp = spark.read.csv(d_uefa_rute, header=True, inferSchema=True)

# Crear un ensamblador de características para convertir las características en un solo vector
assembler = VectorAssembler(
    inputCols=["Posicion","Participaciones","Partidos_Gan","Partidos_Jug","Partidos_Empat","Partidos_perd","Goles_favor","Goles_contra","Puntos","Diferencia_goles","Prob_ganar","Prob_empatar","Prob_perder","Probabilidad_marcar_gol","Probabilidad_recibir_gol"],
    outputCol="features")

# Transformar los datos para incluir la columna de características
data_with_features = assembler.transform(d_uefasp)

# Dividir los datos en conjuntos de entrenamiento y prueba
(train_data, test_data) = data_with_features.randomSplit([0.8, 0.2], seed=1234)

# Entrenar un modelo de árbol de decisiones
dt = DecisionTreeClassifier(labelCol= "Titulos", featuresCol="features")
model = dt.fit(train_data)

# Realizar predicciones en el conjunto de prueba
predictions = model.transform(test_data)

# Evaluar el rendimiento del modelo
evaluator = MulticlassClassificationEvaluator(labelCol= "Titulos", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("\n")
print("\n")
print("Exactitud del modelo:", accuracy)
print("\n")
print("\n")
