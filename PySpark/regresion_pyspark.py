from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

# Crear una sesión de Spark
spark = SparkSession.builder \
    .appName("Análisis de Datos de la Casa de Apuestas") \
    .getOrCreate()

# Cargar los datos desde un archivo CSV
data_rute = os.path.join(os.path.dirname(__file__), 'partidos_definitivos.csv')
data = spark.read.csv(data_rute, header=True, inferSchema=True)

# Crear un ensamblador de características para convertir las características en un solo vector
assembler = VectorAssembler(
    inputCols=["Prob_ganar_local", "Prob_empate", "Prob_ganar_visitante"],
    outputCol="features")

# Transformar los datos para incluir la columna de características
data_with_features = assembler.transform(data)

# Dividir los datos en conjuntos de entrenamiento y prueba
(train_data, test_data) = data_with_features.randomSplit([0.8, 0.2], seed=1234)

# Entrenar un modelo de regresión logística
lr = LogisticRegression(labelCol="Id_local", featuresCol="features", maxIter=10)
model = lr.fit(train_data)

# Realizar predicciones en el conjunto de prueba
predictions = model.transform(test_data)

# Evaluar el rendimiento del modelo
evaluator = MulticlassClassificationEvaluator(labelCol="Id_local", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Exactitud del modelo:", accuracy)

# Detener la sesión de Spark
spark.stop()