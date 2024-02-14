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

# Con esto he obtenido las probabilidades individuales de cada equipo, pero ahora tengo que las probabilidades enfrentadas. Es decir, las prob de un equipo con las de otro equipo en un partido. 
# Para ello, voy a hacer una función que me devuelva las probabilidades enfrentadas de dos equipos.

def prob_enfrentada(equipo1, equipo2):
    # con esto cogemos el valor de la probabilidad del equipo que coincide con el nombre del club que le pasamos
    prob_ganar1 = d_uefa[d_uefa['Club'] == equipo1]['Prob_ganar'].values[0] 
    prob_ganar2 = d_uefa[d_uefa['Club'] == equipo2]['Prob_ganar'].values[0]
    prob_empat1 = d_uefa[d_uefa['Club'] == equipo1]['Prob_Empat'].values[0]
    prob_empat2 = d_uefa[d_uefa['Club'] == equipo2]['Prob_Empat'].values[0]
    prob_perder1 = d_uefa[d_uefa['Club'] == equipo1]['Prob_perder'].values[0]
    prob_perder2 = d_uefa[d_uefa['Club'] == equipo2]['Prob_perder'].values[0]
    # la probabilidad de ganar de un equipo es: prob ganar* prob perder del otro equipo. misma lógica en los otros dos casos
    prob_enfrentada_ganar = (prob_ganar1*prob_perder2)/100
    prob_enfrentada_empat = (prob_empat1*prob_empat2)/100
    prob_enfrentada_perder = (prob_perder1*prob_ganar2)/100
    print(prob_enfrentada_ganar, prob_enfrentada_empat, prob_enfrentada_perder)
    return prob_enfrentada_ganar, prob_enfrentada_empat, prob_enfrentada_perder