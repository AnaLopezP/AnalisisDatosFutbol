import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# cargo los datos
df = pd.read_csv('partidos_definitivos.csv', delimiter = ',')

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
    elif pred == 1:
        print(f"En el partido {i+1}, gana el equipo local.")
    else:
        print(f"En el partido {i+1}, gana el equipo visitante.")