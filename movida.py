import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# cargo los datos
d_uefa = pd.read_csv('datos_uefa.csv', delimiter=',')
d_uefa.columns = ["Posicion", "Club", "Pais", "Participaciones", "Titulos", "Partidos_Jug", "Partidos_Gan", "Partidos_Empat", "Partidos_perd", "Goles_favor", "Goles_contra", "Puntos", "Diferencia_goles"]
print(d_uefa.head())

#Para hallar la probabilidad de ganar un partido, dividimos el numero de partidos ganados entre los jugados y multiplicamos por 100
prob_ganar = (d_uefa['Partidos_Gan']/d_uefa['Partidos_Jug'])*100
#lo mismo para empatar y perder:
prob_perder = (d_uefa['Partidos_perd']/d_uefa['Partidos_Jug'])*100
prob_empat = (d_uefa['Partidos_Empat']/d_uefa['Partidos_Jug'])*100

# Añadimos el dato a una nuevas columnas
d_uefa['Prob_ganar'] = prob_ganar
d_uefa['Prob_Empat'] = prob_empat
d_uefa['Prob_perder'] = prob_perder

print(d_uefa.head())