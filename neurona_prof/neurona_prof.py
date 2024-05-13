import torch as t
import torch.nn as nn
import torch.utils.data as td
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Definimos la red neuronal
#numero de capas ocultas
hl = 10

# definimos la red
class UefaNet(nn.Module):
    def __init__(self):
        super(UefaNet, self).__init__()
        self.fc1 = nn.Linear(14, hl)
        self.fc2 = nn.Linear(hl, hl)
        self.fc3 = nn.Linear(hl, 14)
        
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

def test(modelo, data_loader):
    modelo.eval()
    test_loss = 0
    correct = 0
    
    with t.no_grad():
        batch_count = 0
        for batch, tensor in enumerate(data_loader):
            batch_count += 1
            data, target = tensor
            out = modelo(data)
            
            test_loss += t.nn.functional.cross_entropy(out, target, reduction='sum').item()
            _, pred = t.max(out.data, 1)
            correct += t.sum(target == pred).item()
            
    # calculamos la pérdida media y la precisión
    avg_loss = test_loss / batch_count
    print('Prueba: pérdida media: %f, precisión: %f' % (test_loss, 100. * correct / len(data_loader.dataset)))
    return avg_loss

loss_crit = nn.CrossEntropyLoss()

# Usamos adam como optimizador
learning_rate = 0.001
optimizador = t.optim.Adam(modelo.parameters(), lr=learning_rate)
#optimizador.zero_grad()

if __name__ == '__main__':
    
    # Vamos a añadir una columna clasidficando el estilo de juego de los equipos en ataque o defensa
    # Si la probabilidad de meter gol es alta (mas 0.5) --> ataque
    # Si la probabilidad de recibir gol es baja (menos 0.5) --> defensa
    # Cargamos los datos
    d_uefa_ruta = os.path.join(os.path.dirname(__file__), 'datos_uefa_mejorados_clasif.csv')
    d_uefa = pd.read_csv(d_uefa_ruta, delimiter=',')

    def clasificar_estilo(row):
        if row['Probabilidad_marcar_gol'] > 0.5 and row['Probabilidad_recibir_gol'] > 0.5:
            return 1 # Ataque
        if row['Probabilidad_marcar_gol'] < 0.5 and row['Probabilidad_recibir_gol'] < 0.5:
            return 0 # Defensa
        else:
            return 2 # Equilibrado

    d_uefa['estiloFutbol'] = d_uefa.apply(clasificar_estilo, axis=1)

    # Guardmos el csv con la nueva columna
    d_uefa.to_csv(d_uefa_ruta, index=False)

    # Quito las columnas que no son numericas
    d_uefa = d_uefa.drop(columns=['Club', 'Pais'])

    #Ponemos una semilla aleatoria
    t.manual_seed(0)

    # Separamos 70-30 en train y test
    features = d_uefa.drop(columns=['Prob_ganar' ,'Prob_empatar','Prob_perder'])
    #label = d_uefa['Prob_ganar' ,'Prob_empatar','Prob_perder']
    label = d_uefa["estiloFutbol"]

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=0)
    print(X_train, X_test, y_train, y_test)

    # miramos los primero 25 de entrenamiento
    for i in range(0, 24):
        print(X_train.iloc[i], y_train.iloc[i])
        
    print("HASTA AQUÍ TODO BIEN 1")
        
    # Preparamos los datos para torch
    # Hacemos un dataset y un loader para los datos de entrenamiento
    train_X = t.Tensor(X_train.values).float()
    train_Y = t.Tensor(y_train.values).long()
    train_ds = td.TensorDataset(train_X, train_Y)
    train_loader = td.DataLoader(train_ds, batch_size=20, shuffle=True, num_workers=1)

    # Hacemos lo mismo para los datos de prueba
    test_X = t.Tensor(X_test.values).float()
    test_Y = t.Tensor(y_test.values).long()
    test_ds = td.TensorDataset(test_X, test_Y)
    test_loader = td.DataLoader(test_ds, batch_size=20, shuffle=True, num_workers=1)
    print("HASTA AQUÍ TODO BIEN")


    
    epoch_nums = []
    training_loss = []
    validation_loss = []

    epochs = 50
    for epoch in range(1, epochs +1):
        print('Epoch %d' % epoch)
        train_loss = train(modelo, train_loader, optimizador)
        test_loss = test(modelo, test_loader)
        
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        
    # Graficamos la pérdida
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

    for param_tensor in modelo.state_dict():
        print(param_tensor, '\n', modelo.state_dict()[param_tensor].size(), '\n', modelo.state_dict()[param_tensor], '\n')
        
    # evaluamos el modelo
    modelo.eval()
    x = t.Tensor(X_test.values).float()
    _, pred = t.max(modelo(x).data, 1)

    # Guardamos el modelo
    modelo_ruta = os.path.join(os.path.dirname(__file__), 'modelo_uefa.pth')
    t.save(modelo.state_dict(), modelo_ruta)
    del modelo
    print('Modelo guardado en %s' % modelo_ruta)
    print('Modelo guardado como modelo_uefa.pth en la carpeta actual.')
    
# Ponnemos el modelo a prueba
modelo = UefaNet()
modelo.load_state_dict(t.load(modelo_ruta))
modelo.eval()
x_nuevos = None
x = t.Tensor(x_nuevos.values).float()
_, pred = t.max(modelo(x).data, 1)
print("Predicciones: ", pred)