import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os

# Cargar los dataset
equipo_raw_rute = os.path.join(os.path.dirname(__file__), 'equipo_raw.csv')
equipo = pd.read_csv(equipo_raw_rute)
partidos_rute = os.path.join(os.path.dirname(__file__), 'partidos.csv')
partidos = pd.read_csv(partidos_rute)

# Mostrar los primeros registros del dataset
print(equipo.head())
print(partidos.head())

# Primero vamos a nombrar las columnas de los dataset
equipo.columns = ['nombre_equipo', 'escudo_equipo', 'pais', 'id_equipo']
partidos.columns = ['fecha', 'id_equipo_local', 'id_equipo_visitante', '33', '34', 'goles_local', 'goles_visitante']
# Quitamos las columnas 33 y 34
partidos = partidos.drop(['33', '34'], axis=1)

# Printo los resultados
print(equipo.head())
print(partidos.head())

# Ahora que hemos limpiado los datos, vamos a unir los dos dataset
data = pd.merge(partidos, equipo, left_on='id_equipo_local', right_on='id_equipo')

# Mostramos los primeros registros
print(data.head())

# Ahora vamos a hacer gráficos de los datos
# Vamos a hacer un gráfico de barras con los goles que ha marcado cada equipo
data.groupby('nombre_equipo')['goles_local'].sum().plot(kind='bar')
plt.xlabel('Equipo')
plt.ylabel('Goles')
plt.title('Goles marcados por equipo')
plt.show()

# Vamos a graficar los 5 equipos que más goles han marcado
data.groupby('nombre_equipo')['goles_local'].sum().sort_values(ascending=False).head().plot(kind='bar')
plt.xlabel('Equipo')
plt.ylabel('Goles')
plt.title('Goles marcados por equipo')
plt.show()



# Gráfico de dispersión con el promedio de goles en casa por equipo
promedio_goles_local = data.groupby('nombre_equipo')['goles_local'].mean()
promedio_goles_local.plot(marker='o', linestyle='-', color='green')
plt.xlabel('Equipo')
plt.ylabel('Promedio de goles en casa')
plt.title('Promedio de goles por partido en casa por equipo')
plt.show()

# Vamos a graficar el numero de partidos jugados por equipo
partidos_jugados = data['nombre_equipo'].value_counts()
partidos_jugados.plot(kind = 'bar')
plt.xlabel('Equipo')
plt.ylabel('Partidos Jugados')
plt.title('Partidos jugados por equipo')
plt.show()

# Después de ver los gráficos, vamos a hacer un análisis de los datos
# Vamos a calcular el promedio de goles en casa por equipo
promedio_goles_local = data.groupby('nombre_equipo')['goles_local'].mean()
print("\n")
print('-----------------Promedio de goles en casa por equipo-----------------')
print("\n")
print(promedio_goles_local)

# Vamos a calcular el desempeño de los equipos
partidos_jugados = data['nombre_equipo'].value_counts()
goles_marcados = data.groupby('nombre_equipo')['goles_local'].sum()
goles_recibidos = data.groupby('nombre_equipo')['goles_visitante'].sum()
desempeno = pd.DataFrame({'partidos_jugados': partidos_jugados, 'goles_marcados': goles_marcados, 'goles_recibidos': goles_recibidos})
desempeno['diferencia_goles'] = desempeno['goles_marcados'] - desempeno['goles_recibidos']
print("\n")
print('-----------------Desempeño de los equipos-----------------')
print("\n")
print(desempeno)

# Vamos a calcular el número de partidos jugados por equipo
partidos_jugados = data['nombre_equipo'].value_counts()
print("\n")
print('-----------------Número de partidos jugados por equipo-----------------')
print("\n")
print(partidos_jugados)

# Vamos a calcular la varianza y la desviación estandar de los goles marcados por equipo
varianza_goles_marcados = data.groupby('nombre_equipo')['goles_local'].var()
desviacion_estandar_goles_marcados = data.groupby('nombre_equipo')['goles_local'].std()
print("\n")
print('-----------------Varianza de los goles marcados por equipo-----------------')
print("\n")
print(varianza_goles_marcados)
print("\n")
print('-----------------Desviación estandar de los goles marcados por equipo-----------------')
print("\n")
print(desviacion_estandar_goles_marcados)


# De los datos, podemos concluir que el equipo al que mejor le ha ido esta temporada es a Manchester United, con una media de 3.3 goles por partido y ningún gol en su contra.
# Tambien podemos ver que el peor equipo es el Dynamo Kyiv, con una media de 0.3 goles por partido, y 12 goles totales en contra.