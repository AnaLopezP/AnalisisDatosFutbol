# Importar las bibliotecas necesarias
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, median, stddev, desc
import matplotlib.pyplot as plt
import os

# Crear una sesión de Spark
spark = SparkSession.builder \
    .appName("Análisis de Datos de la Casa de Apuestas") \
    .getOrCreate()

# Cargar los datos desde un archivo CSV
data_rute = os.path.join(os.path.dirname(__file__), 'partidos_definitivos.csv')
data = spark.read.csv(data_rute, header=True, inferSchema=True)

# Mostrar el esquema de los datos
print("Esquema de los datos:")
data.printSchema()

# Calcular estadísticas descriptivas para las probabilidades de ganar local y visitante
print("Estadísticas descriptivas para las probabilidades de ganar:")
stats_local = data.select(mean("Prob_ganar_local").alias("Media_local"),
                          median("Prob_ganar_local").alias("Mediana_local"),
                          stddev("Prob_ganar_local").alias("Desviacion_local")).show()

stats_visitante = data.select(mean("Prob_ganar_visitante").alias("Media_visitante"),
                              median("Prob_ganar_visitante").alias("Mediana_visitante"),
                              stddev("Prob_ganar_visitante").alias("Desviacion_visitante")).show()

# Identificar los equipos con las probabilidades más altas de ganar en general
print("Equipos con las probabilidades más altas de ganar:")
equipos_mas_probables = data.select("Id_local", "Id_visitante") \
                            .orderBy(desc("Prob_ganar_local"), desc("Prob_ganar_visitante")) \
                            .limit(5).show()

# Visualizar la distribución de las probabilidades de ganar para los equipos locales y visitantes
plt.figure(figsize=(10, 6))
data.select("Prob_ganar_local", "Prob_ganar_visitante").toPandas().plot(kind="hist", bins=20, alpha=0.5)
plt.xlabel("Probabilidad de Ganar")
plt.ylabel("Frecuencia")
plt.title("Distribución de Probabilidades de Ganar")
plt.legend(["Local", "Visitante"])
plt.grid(True)
plt.show()

# Detener la sesión de Spark
spark.stop()
