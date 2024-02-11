import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# cargo los datos
d_uefa = pd.read_csv('uefa-movida-bien.csv', delim_whitespace=True)
#d_uefa.columns = ["Posicion", "Club", "Pais", "Participaciones", "Titulos", "Partidos_Jug", "Partidos_Gan", "Partidos_Empat", "Partidos_perd", "Goles_favor", "Goles_contra", "Puntos", "Diferencia_goles"]
print(d_uefa.head())

# quito las comillas y que me molestan tio
for dato in d_uefa:
    d_uefa[dato] = d_uefa[dato].str.replace('"', '')
    

# guardo los datos en un archivo nuevo
d_uefa.to_csv('uefa.csv', index=False, sep=',')