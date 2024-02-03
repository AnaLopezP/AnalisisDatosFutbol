import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Cargar los dataset
equipo = pd.read_csv('equipo.csv')
partidos = pd.read_csv('partidos.csv')

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
