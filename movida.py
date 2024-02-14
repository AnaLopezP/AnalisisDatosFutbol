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
    # con esto cogemos el valor de la probabilidad del equipo que coincide con el id del club que le pasamos
    prob_ganar1 = d_uefa[d_uefa['Posicion'] == equipo1]['Prob_ganar'].values[0] 
    prob_ganar2 = d_uefa[d_uefa['Posicion'] == equipo2]['Prob_ganar'].values[0]
    prob_empat1 = d_uefa[d_uefa['Posicion'] == equipo1]['Prob_Empat'].values[0]
    prob_empat2 = d_uefa[d_uefa['Posicion'] == equipo2]['Prob_Empat'].values[0]
    prob_perder1 = d_uefa[d_uefa['Posicion'] == equipo1]['Prob_perder'].values[0]
    prob_perder2 = d_uefa[d_uefa['Posicion'] == equipo2]['Prob_perder'].values[0]
    # la probabilidad de ganar de un equipo es: prob ganar* prob perder del otro equipo. misma lógica en los otros dos casos
    prob_enfrentada_ganar = (prob_ganar1*prob_perder2)/100
    prob_enfrentada_empat = (prob_empat1*prob_empat2)/100
    prob_enfrentada_perder = (prob_perder1*prob_ganar2)/100
    print(prob_enfrentada_ganar, prob_enfrentada_empat, prob_enfrentada_perder)
    return prob_enfrentada_ganar, prob_enfrentada_empat, prob_enfrentada_perder

# Ahora que podemos calcular las probabilidades, necesito un csv con los partidos reales que se van a jugar. 
# Las columnas van a ser las siguientes: fecha y hora, id equipo locar, id equipo visitante, probabilidad gana local, probabilidad empate, probabilidad gana visitante
# Para el id de los equipos voy a usar la columna posicion del csv que he cargado.

partidos = pd.DataFrame(columns = ['Fecha y Hora', 'Id_local', 'Id_visitante', 'Prob_ganar_local', 'Prob_empate', 'Prob_ganar_visitante'])

partidos['Fecha y Hora'] = ['13/02/2024 21:00', '13/02/2024 21:00', '14/02/2024 21:00', '14/02/2024 21:00', '20/02/2024 21:00', '20/02/2024 21:00', '21/02/2024 21:00', '21/02/2024 21:00']

partidos['Id_local'] = [d_uefa[d_uefa['Club'].str.contains('København', case=False)]['Posicion'].values[0],
                        d_uefa[d_uefa['Club'].str.contains('leipzig', case=False)]['Posicion'].values[0],
                        d_uefa[d_uefa['Club'].str.contains('paris saint', case=False)]['Posicion'].values[0],
                        d_uefa[d_uefa['Club'].str.contains('lazio', case=False)]['Posicion'].values[0],
                        d_uefa[d_uefa['Club'].str.contains('inter', case=False)]['Posicion'].values[0],
                        d_uefa[d_uefa['Club'].str.contains('PSV', case=False)]['Posicion'].values[0],
                        d_uefa[d_uefa['Club'].str.contains('porto', case=False)]['Posicion'].values[0],
                        d_uefa[d_uefa['Club'].str.contains('napoli', case=False)]['Posicion'].values[0]]

partidos['Id_visitante'] = [d_uefa[d_uefa['Club'].str.contains('Manchester city', case=False)]['Posicion'].values[0],
                            d_uefa[d_uefa['Club'].str.contains('real madrid', case=False)]['Posicion'].values[0],
                            d_uefa[d_uefa['Club'].str.contains('real sociedad', case=False)]['Posicion'].values[0],
                            d_uefa[d_uefa['Club'].str.contains('bayern', case=False)]['Posicion'].values[0],
                            d_uefa[d_uefa['Club'].str.contains('atlético de madrid', case=False)]['Posicion'].values[0],
                            d_uefa[d_uefa['Club'].str.contains('dortmund', case=False)]['Posicion'].values[0],
                            d_uefa[d_uefa['Club'].str.contains('arsenal', case=False)]['Posicion'].values[0],
                            d_uefa[d_uefa['Club'].str.contains('barcelona', case=False)]['Posicion'].values[0]]

partidos['Prob_ganar_local'] = [prob_enfrentada(303, 27)[0],
                                prob_enfrentada(94, 1)[0],
                                prob_enfrentada(20, 133)[0],
                                prob_enfrentada(68, 2)[0],
                                prob_enfrentada(13, 17)[0],
                                prob_enfrentada(19, 16)[0],
                                prob_enfrentada(9, 14)[0],
                                prob_enfrentada(60, 3)[0]]

partidos['Prob_empate'] = [prob_enfrentada(303, 27)[1],
                            prob_enfrentada(94, 1)[1],
                            prob_enfrentada(20, 133)[1],
                            prob_enfrentada(68, 2)[1],
                            prob_enfrentada(13, 17)[1],
                            prob_enfrentada(19, 16)[1],
                            prob_enfrentada(9, 14)[1],
                            prob_enfrentada(60, 3)[1]]

partidos['Prob_ganar_visitante'] = [prob_enfrentada(303, 27)[2],
                                    prob_enfrentada(94, 1)[2],
                                    prob_enfrentada(20, 133)[2],
                                    prob_enfrentada(68, 2)[2],
                                    prob_enfrentada(13, 17)[2],
                                    prob_enfrentada(19, 16)[2],
                                    prob_enfrentada(9, 14)[2],
                                    prob_enfrentada(60, 3)[2]]

print(partidos)
# Ahora que tengo los datos, voy a guardarlos en un csv
partidos.to_csv('partidos_definitivos.csv', index=False)