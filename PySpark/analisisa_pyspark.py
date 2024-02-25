from pyspark.sql import SparkSession
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Creo una sesión de spark
spark = SparkSession.builder.appName('Analisis de Datos').getOrCreate()

# Cargo el csv
d_uefa_ruta = os.path.join(os.path.dirname(__file__), 'datos_uefa.csv')
d_uefa = spark.read.csv(d_uefa_ruta, header=True, inferSchema=True)

# LIMPIEZA DE DATOS
# Quito valores nulos
d_uefa = d_uefa.dropna()

# Elimino datos duplicados
d_uefa = d_uefa.dropDuplicates()

# ANÁLISIS EXPLOTATORIO DE DATOS
# Calculo estadísticas descriptivas
d_uefa.describe().show()

# Hago agregaciones
d_uefa.groupBy("Pais").agg({"Partidos_Jug": "sum", "Titulos": "sum"}).show()
pd_d_uefa = d_uefa.toPandas()
pd_d_uefa.plot(x="Posicion", y="Participaciones", kind="bar")
plt.show()

# REGRESIÓN LOGÍSTICA
# Creo un vector assembler
assembler = VectorAssembler(inputCols=["Participaciones", "Titulos", "Partidos_Jug", "Partidos_Gan", "Partidos_Empat", 
                                       "Partidos_perd", "Goles_favor", "Goles_contra", "Puntos", "Diferencia_goles"],
                            outputCol="features")
data_assembled = assembler.transform(d_uefa)

# Divido los datos en conjuntos de entrenamiento y prueba
train, test = data_assembled.randomSplit([0.7, 0.3], seed=12345)

# Inicializo el modelo de regresión logística
lr = LogisticRegression(featuresCol="features", labelCol="Posicion")

# Entreno el modelo
lr_model = lr.fit(train)

# predicciones en el conjunto de prueba
predictions = lr_model.transform(test)

# Evaluo el rendimiento del modelo
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Posicion")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

# Detener la sesión de Spark
spark.stop()