import pandas as pd
import torch
from neurona_prof import UefaNet
import os

# cargamos los datos que vamos a clasificar
d_uefa_ruta = os.path.join(os.path.dirname(__file__), 'datos_uefa_mejorados_clasif.csv')
d_uefa = pd.read_csv(d_uefa_ruta, delimiter=',')
x_nuevos = d_uefa.drop(['club', 'pais'], axis=1)
x_nuevos = x_nuevos.head(5)
hl = 10
entrada = x_nuevos.shape[1]

# Cargamos el modelo
modelo = UefaNet(entrada, hl)
ruta_modelo = os.path.join(os.path.dirname(__file__), 'modelo_uefa.pth')
modelo.load_state_dict(torch.load(ruta_modelo))
modelo.eval()

# predecimos
x = torch.Tensor(x_nuevos.values).float()
_, pred = torch.max(modelo(x), 1)
print("Predicciones:\n", pred)