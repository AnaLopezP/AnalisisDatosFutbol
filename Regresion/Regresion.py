import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from crearCSV import prob_enfrentada

# cargo los datos
df = pd.read_csv('partidos_definitivos.csv', delimiter = ',')
d_uefa = pd.read_csv('datos_uefa.csv', delimiter = ',')

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

Ronda2['Prob_ganar_local'] = [prob_enfrentada(303, 27)[0],]