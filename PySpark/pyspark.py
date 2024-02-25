from pyspark.sql import SparkSession
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

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

# REGRESIÓN LINEAL
# Creo un vector assembler
assembler = VectorAssembler(inputCols=["Partidos_Jug", "Partidos_Gan", "Partidos_Empat", "Partidos_perd", "Goles_favor", "Goles_contra", "Puntos", "Diferencia_goles"], outputCol="features")
d_uefa = assembler.transform(d_uefa)

lr = LinearRegression(featuresCol="features", labelCol="Partidos_Gan")
lr_model = lr.fit(d_uefa)

# Muestro los coeficientes
print("Coeficientes: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

# Realizo predicciones
predicciones = lr_model.transform(d_uefa)
predicciones.select("Partidos_Gan", "prediction").show()