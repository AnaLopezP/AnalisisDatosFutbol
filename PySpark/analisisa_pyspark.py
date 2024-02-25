from pyspark.sql import SparkSession
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Creo una sesión de Spark
spark = SparkSession.builder.appName('Predicción Champions League').getOrCreate()

# Rutas de los archivos
uefa_d_ruta = os.path.join(os.path.dirname(__file__), 'datos_uefa.csv')
partidos_ruta = os.path.join(os.path.dirname(__file__), 'partidos_definitivos.csv')

# Cargo el csv de la UEFA
uefa_df = spark.read.csv(uefa_d_ruta, header=True, inferSchema=True)

# Cargo el csv de los partidos
partidos_df = spark.read.csv(partidos_ruta, header=True, inferSchema=True)

# Unir ambos conjuntos de datos en función del campo 'Pais' para obtener un conjunto de datos combinado
combined_df = uefa_df.join(partidos_df, uefa_df.Pais == partidos_df.Pais)

# REGRESIÓN LOGÍSTICA
# Creo un vector assembler
assembler = VectorAssembler(inputCols=["Participaciones", "Titulos", "Partidos_Jug", "Partidos_Gan", "Partidos_Empat",
                                       "Partidos_perd", "Goles_favor", "Goles_contra", "Puntos", "Diferencia_goles",
                                       "Prob_ganar_local", "Prob_empate", "Prob_ganar_visitante"],
                            outputCol="features")
data_assembled = assembler.transform(combined_df)

# Inicializo el modelo de regresión logística
lr = LogisticRegression(featuresCol="features", labelCol="Posicion", family="multinomial")

# Entreno el modelo
lr_model = lr.fit(data_assembled)

# Predicciones
predictions = lr_model.transform(data_assembled)

# Resultados
predictions.select("Pais", "Club", "prediction").show()

# Detengo la sesión de Spark
spark.stop()
