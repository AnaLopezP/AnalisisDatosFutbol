import numpy as np 
import matplotlib as plt
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