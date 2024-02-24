import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from crearCSV import prob_enfrentada
import os

directorio_script = os.path.dirname(__file__)

# cargo los datos
df_ruta = os.path.join(os.path.dirname(__file__), 'partidos_definitivos.csv')
df = pd.read_csv(df_ruta, delimiter = ',')
datos_uefa_ruta = os.path.join(os.path.dirname(__file__), 'datos_uefa.csv')
d_uefa = pd.read_csv(datos_uefa_ruta, delimiter = ',')

# divido los datos en variables independientes y dependientes
X = df[['Prob_ganar_local', 'Prob_empate', 'Prob_ganar_visitante']]
# la variable dependiente tendría que ser una columna que contenga el resultado del partido, así que la creo
# Suponemos que el ganador es el equipo con mayor probabilidad de ganar
# Ponemos por defecto que gana el local
df['Resultado'] = 1 # Local
#  Comparamos las probabilidades
df.loc[df['Prob_ganar_local'] < df['Prob_ganar_visitante'], 'Resultado'] = 2 # Visitante
df.loc[df['Prob_empate'] > df[['Prob_ganar_local', 'Prob_ganar_visitante']].max(axis=1), 'Resultado'] = 0 # Empate

y = df['Resultado']

# divido los datos en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Entreno el modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Evalúo el modelo
y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy)
print(classification_report(y_test, y_pred))

# Ahora que tengo el modelo, puedo predecir los resultados de los partidos que se van a jugar
predicciones = modelo.predict(X)

print("Predicciones:")
for i, pred in enumerate(predicciones):
    if pred == 0:
        print(f"En el partido {i+1}, es un empate.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == df.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == df.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")
    elif pred == 1:
        print(f"En el partido {i+1}, gana el equipo local.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == df.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == df.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")
    else:
        print(f"En el partido {i+1}, gana el equipo visitante.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == df.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == df.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")
        

# Ahora que tengo las predicciones, guardo en un csv los partidos que se van a jugar con los ganadores
Ronda2 = pd.DataFrame(columns = ['Id_local', 'Id_visitante', 'Prob_ganar_local', 'Prob_empate', 'Prob_ganar_visitante'])
Ronda2['Id_local'] = [d_uefa[d_uefa['Club'].str.contains('Manchester City FC', case=False)]['Posicion'].values[0],
                      d_uefa[d_uefa['Club'].str.contains('Real Madrid CF', case=False)]['Posicion'].values[0],
                      d_uefa[d_uefa['Club'].str.contains('Paris Saint-Germain', case=False)]['Posicion'].values[0],
                      d_uefa[d_uefa['Club'].str.contains('FC Internazionale Milano', case=False)]['Posicion'].values[0]]
                      
Ronda2['Id_visitante'] = [d_uefa[d_uefa['Club'].str.contains('FC Barcelona', case=False)]['Posicion'].values[0],
                          d_uefa[d_uefa['Club'].str.contains('FC Bayern München', case=False)]['Posicion'].values[0],
                          d_uefa[d_uefa['Club'].str.contains('Arsenal FC', case=False)]['Posicion'].values[0],
                          d_uefa[d_uefa['Club'].str.contains('Borussia Dortmund', case=False)]['Posicion'].values[0]]

# Calculamos las probabilidades enfrentadas para los partidos de la Ronda 2
Ronda2['Prob_ganar_local'] = [prob_enfrentada(27, 3)[0],
                              prob_enfrentada(1, 2)[0],
                              prob_enfrentada(20, 14)[0],
                              prob_enfrentada(13, 16)[0]]

Ronda2['Prob_empate'] = [prob_enfrentada(27, 3)[1],
                         prob_enfrentada(1, 2)[1],
                         prob_enfrentada(20, 14)[1],
                         prob_enfrentada(13, 16)[1]]

Ronda2['Prob_ganar_visitante'] = [prob_enfrentada(27, 3)[2],
                                   prob_enfrentada(1, 2)[2],
                                   prob_enfrentada(20, 14)[2],
                                   prob_enfrentada(13, 16)[2]]

# Ponemos las fechas de los partidosç
Ronda2['Fecha y Hora'] = ['13/02/2024 21:00', '13/02/2024 21:00', '14/02/2024 21:00', '14/02/2024 21:00']

# Pongo la columna fecha y hora en formato datetime
Ronda2['Fecha y Hora'] = pd.to_datetime(Ronda2['Fecha y Hora'], format='%d/%m/%Y %H:%M')

# lo guardo en un csv
Ronda2.to_csv(os.path.join(directorio_script, 'partidos_ronda2.csv'), index=False)
# Realizamos las predicciones para los partidos de la Ronda 2
predicciones_ronda2 = modelo.predict(Ronda2[['Prob_ganar_local', 'Prob_empate', 'Prob_ganar_visitante']])

print("Predicciones Ronda 2:")
for i, pred in enumerate(predicciones_ronda2):
    if pred == 0:
        print(f"En el partido {i+1}, es un empate.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == Ronda2.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == Ronda2.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")
    elif pred == 1:
        print(f"En el partido {i+1}, gana el equipo local.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == Ronda2.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == Ronda2.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")
    else:
        print(f"En el partido {i+1}, gana el equipo visitante.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == Ronda2.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == Ronda2.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")
        
# lo mismo para la semifinal
Semifinal = pd.DataFrame(columns = ['Id_local', 'Id_visitante', 'Prob_ganar_local', 'Prob_empate', 'Prob_ganar_visitante'])
Semifinal['Id_local'] = [d_uefa[d_uefa['Club'].str.contains('FC Barcelona', case=False)]['Posicion'].values[0],
                         d_uefa[d_uefa['Club'].str.contains('FC Bayern München', case=False)]['Posicion'].values[0]]
Semifinal['Id_visitante'] = [d_uefa[d_uefa['Club'].str.contains('Paris Saint-Germain', case=False)]['Posicion'].values[0],
                             d_uefa[d_uefa['Club'].str.contains('FC Internazionale Milano', case=False)]['Posicion'].values[0]]

# Calculamos las probabilidades enfrentadas para los partidos de la Semifinal
Semifinal['Prob_ganar_local'] = [prob_enfrentada(3, 20)[0],
                                 prob_enfrentada(2, 13)[0]]
Semifinal['Prob_empate'] = [prob_enfrentada(3, 20)[1],
                            prob_enfrentada(2, 13)[1]]
Semifinal['Prob_ganar_visitante'] = [prob_enfrentada(3, 20)[2],
                                     prob_enfrentada(2, 13)[2]]

# Ponemos las fechas de los partidos
Semifinal['Fecha y Hora'] = ['20/02/2024 21:00', '20/02/2024 21:00']

# Pongo la columna fecha y hora en formato datetime
Semifinal['Fecha y Hora'] = pd.to_datetime(Semifinal['Fecha y Hora'], format='%d/%m/%Y %H:%M')

# lo guardo en un csv
Semifinal.to_csv(os.path.join(directorio_script, 'partidos_semifinal.csv'), index=False)

# Realizamos las predicciones para los partidos de la Semifinal
predicciones_semifinal = modelo.predict(Semifinal[['Prob_ganar_local', 'Prob_empate', 'Prob_ganar_visitante']])
print("Predicciones Semifinal:")
for i, pred in enumerate(predicciones_semifinal):
    if pred == 0:
        print(f"En el partido {i+1}, es un empate.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == Semifinal.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == Semifinal.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")
    elif pred == 1:
        print(f"En el partido {i+1}, gana el equipo local.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == Semifinal.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == Semifinal.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")
    else:
        print(f"En el partido {i+1}, gana el equipo visitante.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == Semifinal.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == Semifinal.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")
        
# lo mismo para la final
Final = pd.DataFrame(columns = ['Id_local', 'Id_visitante', 'Prob_ganar_local', 'Prob_empate', 'Prob_ganar_visitante'])
Final['Id_local'] = [d_uefa[d_uefa['Club'].str.contains('FC Barcelona', case=False)]['Posicion'].values[0]]
Final['Id_visitante'] = [d_uefa[d_uefa['Club'].str.contains('FC Bayern München', case=False)]['Posicion'].values[0]]

# Calculamos las probabilidades enfrentadas para los partidos de la Final
Final['Prob_ganar_local'] = [prob_enfrentada(3, 2)[0]]
Final['Prob_empate'] = [prob_enfrentada(3, 2)[1]]
Final['Prob_ganar_visitante'] = [prob_enfrentada(3, 2)[2]]

# Ponemos las fechas de los partidos
Final['Fecha y Hora'] = ['21/02/2024 21:00']

# Pongo la columna fecha y hora en formato datetime
Final['Fecha y Hora'] = pd.to_datetime(Final['Fecha y Hora'], format='%d/%m/%Y %H:%M')

# lo guardo en un csv
Final.to_csv(os.path.join(directorio_script, 'partidos_final.csv'), index=False)

# Realizamos las predicciones para los partidos de la Final
predicciones_final = modelo.predict(Final[['Prob_ganar_local', 'Prob_empate', 'Prob_ganar_visitante']])
print("Predicciones Final:")
for i, pred in enumerate(predicciones_final):
    if pred == 0:
        print(f"En el partido {i+1}, es un empate.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == Final.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == Final.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")
    elif pred == 1:
        print(f"En el partido {i+1}, gana el equipo local.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == Final.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == Final.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")
    else:
        print(f"En el partido {i+1}, gana el equipo visitante.")
        equipo_local = d_uefa.loc[d_uefa['Posicion'] == Final.iloc[i]['Id_local'], 'Club'].values[0]
        equipo_visitante = d_uefa.loc[d_uefa['Posicion'] == Final.iloc[i]['Id_visitante'], 'Club'].values[0]
        print(f"Los equipos que juegan son {equipo_local} y {equipo_visitante}.")
        print("\n")

print("--------------------------------------------------------------------------")
print("\n")
print("Según mi predicción, debería ganar el FC Barcelona. Pronto lo descubriremos.")
print("\n")



