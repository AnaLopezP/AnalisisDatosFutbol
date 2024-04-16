import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import os

# Cargar los datos
data_ruta = os.path.join(os.path.dirname(__file__), 'datos_uefa.csv')
data = pd.read_csv(data_ruta, delimiter=',')

# Ingeniería de características
features = data[['Participaciones', 'Titulos', 'Partidos_Jug', 'Partidos_Gan', 'Partidos_Empat', 
                  'Partidos_perd', 'Goles_favor', 'Goles_contra', 'Puntos', 'Diferencia_goles',
                  'Prob_ganar', 'Prob_empatar', 'Prob_perder', 'Probabilidad_marcar_gol', 
                  'Probabilidad_recibir_gol']]

# Escalar las características
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Reducción de dimensionalidad
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

# K-Means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(principal_components)
centroides = kmeans.cluster_centers_
etiquetas = kmeans.labels_
print("Centroides: ", centroides)
print("Etiquetas: ", etiquetas)
data['Cluster_KMeans'] = kmeans.labels_

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(principal_components)
etiquetas_dbscan = dbscan.labels_
print("Etiquetas DBSCAN: ", etiquetas_dbscan)
data['Cluster_DBSCAN'] = dbscan.labels_

# GMM clustering
gmm = GaussianMixture(n_components=3)
gmm.fit(principal_components)
etiquetas_gmm = gmm.predict(principal_components)
print("Etiquetas GMM: ", etiquetas_gmm)
data['Cluster_GMM'] = gmm.predict(principal_components)

# Hierarchical clustering
agglomerative = AgglomerativeClustering(n_clusters=3)
agglomerative.fit(principal_components)
etiquetas_agglomerative = agglomerative.labels_
print("Etiquetas Agglomerative: ", etiquetas_agglomerative)
data['Cluster_Agglomerative'] = agglomerative.labels_

# Evaluación del clustering
silhouette_kmeans = silhouette_score(principal_components, kmeans.labels_)
silhouette_dbscan = silhouette_score(principal_components, dbscan.labels_)
silhouette_gmm = silhouette_score(principal_components, gmm.predict(principal_components))
silhouette_agglomerative = silhouette_score(principal_components, agglomerative.labels_)

print("Silhouette Score KMeans:", silhouette_kmeans)
print("Silhouette Score DBSCAN:", silhouette_dbscan)
print("Silhouette Score GMM:", silhouette_gmm)
print("Silhouette Score Agglomerative:", silhouette_agglomerative)

# Visualización de los clusters
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("KMeans Clustering")

plt.subplot(1, 4, 2)
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=dbscan.labels_, cmap='viridis')
plt.title("DBSCAN Clustering")

plt.subplot(1, 4, 3)
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=gmm.predict(principal_components), cmap='viridis')
plt.title("GMM Clustering")

plt.subplot(1, 4, 4)
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=agglomerative.labels_, cmap='viridis')
plt.title("Agglomerative Clustering")

plt.show()

# Análisis de series temporales
team_data = data

# Crear una serie temporal con los datos de participaciones
time_series = team_data.set_index('Posicion')['Participaciones']
print(time_series)

# Dividir la serie temporal en datos de entrenamiento y prueba
train_size = int(len(time_series) * 0.8)
print("Número de datos de entrenamiento:", train_size)
train_data, test_data = time_series.iloc[:train_size], time_series.iloc[train_size:]
print("Número de datos de entrenamiento:", train_data)
print("Número de datos de prueba:", len(test_data))

# Entrenar el modelo ARIMA
model = ARIMA(train_data)
model_fit = model.fit()

# Realizar predicciones
predictions = model_fit.forecast(steps=len(test_data))

# Calcular el error cuadrático medio
mse = mean_squared_error(test_data, predictions)
print("Mean Squared Error:", mse)

# Visualizar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data.values, label='Training Data')
plt.plot(test_data.index, test_data.values, label='Test Data')
plt.plot(test_data.index, predictions, label='Predictions')
plt.title("ARIMA Model")
plt.xlabel("posicion")
plt.ylabel("Puntos")
plt.legend()
plt.show()
