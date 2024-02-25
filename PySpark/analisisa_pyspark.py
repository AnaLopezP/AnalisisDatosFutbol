from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Crear una sesión de Spark
spark = SparkSession.builder.appName("Regresión Logística de Partidos").getOrCreate()

# Cargar el CSV de los partidos
partidos_df = spark.read.csv("partidos.csv", header=True, inferSchema=True)

# Crear un VectorAssembler para combinar las características
feature_cols = ["Prob_ganar_local", "Prob_empate", "Prob_ganar_visitante"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(partidos_df)

# Dividir los datos en conjuntos de entrenamiento y prueba
(train_data, test_data) = data.randomSplit([0.7, 0.3], seed=123)

# Inicializar y entrenar el modelo de regresión logística
lr = LogisticRegression(labelCol="???", featuresCol="features", maxIter=10)
lr_model = lr.fit(train_data)

# Realizar predicciones en el conjunto de prueba
predictions = lr_model.transform(test_data)

# Evaluar el modelo
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="???")
accuracy = evaluator.evaluate(predictions)

# Mostrar el rendimiento del modelo
print("Accuracy:", accuracy)

# Detener la sesión de Spark
spark.stop()
