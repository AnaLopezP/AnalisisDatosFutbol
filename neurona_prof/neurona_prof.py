import torch as t
import torch.nn as nn
import torch.utils.data as td
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

#Ponemos una semilla aleatoria
t.manual_seed(0)

# Cargamos los datos
d_uefa_ruta = os.path.join(os.path.dirname(__file__), 'datos_uefa.csv')
d_uefa = pd.read_csv(d_uefa_ruta, delimiter=',')

# Separamos 70-30 en train y test
features = d_uefa.drop(columns=['Posicion'])
labels = d_uefa['Posicion']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)
print('Set entrenamiento: %d, Set de prueba: %d \n' % (len(X_train), len(X_test)))

# miramos los primero 25 de entrenamiento
for i in range(0, 24):
    print(X_train.iloc[i], y_train.iloc[i])
    
# Preparamos los datos para torch
# Hacemos un dataset y un loader para los datos de entrenamiento
train_X = t.Tensor(X_train).float()
train_Y = t.Tensor(y_train).long()
train_ds = td.TensorDataset(train_X, train_Y)
train_loader = td.DataLoader(train_ds, batch_size=20, shuffle=True, num_workers=1)

# Hacemos lo mismo para los datos de prueba
test_X = t.Tensor(X_test).float()
test_Y = t.Tensor(y_test).long()
test_ds = td.TensorDataset(test_X, test_Y)
test_loader = td.DataLoader(test_ds, batch_size=20, shuffle=True, num_workers=1)
print("HASTA AQUÍ TODO BIEN")

# Definimos la red neuronal
#numero de capas ocultas
hl = 10

# definimos la red
class UefaNet(nn.Module):
    def __init__(self):
        super(UefaNet, self).__init__()
        self.fc1 = nn.Linear(len(features), hl)
        self.fc2 = nn.Linear(hl, hl)
        self.fc3 = nn.Linear(hl, len(labels))
        
    def forward(self, x):
        x = t.relu(self.fc1(x))
        x = t.relu(self.fc2(x))
        x = t.relu(self.fc3(x))
        return x
    
# creamos una instancia
modelo = UefaNet()
print(modelo)

# Entremanos la red
def train(modelo, data_loader, optimizador):
    modelo.train()
    train_loss = 0
    
    for batch, tensor in enumerate(data_loader):
        data, target = tensor
        # feed forward
        optimizador.zero_grad()
        out = modelo(data)
        # calculamos la pérdida
        loss = t.nn.functional.cross_entropy(out, target)
        # backpropagation
        loss.backward()
        optimizador.step()
        
    # calculamos la pérdida media
    avg_loss = train_loss / len(data_loader.dataset)
    print('Entrenamiento: pérdida media: %f' % avg_loss)
    return avg_loss