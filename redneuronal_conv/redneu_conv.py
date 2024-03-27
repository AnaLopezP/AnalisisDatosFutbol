import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Cargamos los datos
data_path = 'data/shapes/'

classes = os.listdir(data_path)
classes.sort()
print(len(classes), 'classes:', classes)

# Mostramos la primera imagen en cada carpeta
fig = plt.figure(figsize=(8, 12))
i = 0
for sub_dir in os.listdir(data_path):
    i+=1
    img_file = os.listdir(os.path.join(data_path,sub_dir))[0]
    img_path = os.path.join(data_path, sub_dir, img_file)
    img = mpimg.imread(img_path)
    a=fig.add_subplot(1, len(classes),i)
    a.axis('off')
    imgplot = plt.imshow(img)
    a.set_title(img_file)
plt.show()

# Función para cargar datos utilizando cargadores de entrenamiento y prueba
def load_dataset(data_path):
    # Cargar todas las imágenes
    transformación = transforms.Compose([
        # transformar a tensores
        transforms.ToTensor(),
        # Normalizar los valores de píxeles (en los canales R, G y B)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Cargar todas las imágenes, transformándolas
    conjunto_completo = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformación
    )
    
    
    # Dividir en conjuntos de entrenamiento (70%) y prueba (30%)
    tamaño_entrenamiento = int(0.7 * len(conjunto_completo))
    tamaño_prueba = len(conjunto_completo) - tamaño_entrenamiento
    conjunto_entrenamiento, conjunto_prueba = torch.utils.data.random_split(conjunto_completo, [tamaño_entrenamiento, tamaño_prueba])
    
    # definir un cargador para los datos de entrenamiento que podemos iterar en lotes de 50 imágenes
    cargador_entrenamiento = torch.utils.data.DataLoader(
        conjunto_entrenamiento,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )
    
    # definir un cargador para los datos de prueba que podemos iterar en lotes de 50 imágenes
    cargador_prueba = torch.utils.data.DataLoader(
        conjunto_prueba,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )
        
    return cargador_entrenamiento, cargador_prueba


# Obtener los cargadores iterativos para los datos de prueba y entrenamiento
cargador_entrenamiento, cargador_prueba = load_dataset(data_path)
print('Cargadores de datos listos')

# Creamos una clase de red neuronal
class Net(nn.Module):
    # Constructor
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        
        # Las imagener son rgb, asi que el canal es el 3, aplicamos 12 filtros en la primera convolucional
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # We'll apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # A second convolutional layer takes 12 input channels, and generates 12 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # A third convolutional layer takes 12 inputs and generates 24 outputs
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        
        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)
        
        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
        # So our feature tensors are now 32 x 32, and we've generated 24 of them
        # We need to flatten these and feed them to a fully-connected layer
        # to map them to  the probability for each class
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # Use a relu activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))
      
        # Use a relu activation function after layer 2 (convolution 2 and pool)
        x = F.relu(self.pool(self.conv2(x)))
        
        # Select some features to drop after the 3rd convolution to prevent overfitting
        x = F.relu(self.drop(self.conv3(x)))
        
        # Only drop the features if this is a training pass
        x = F.dropout(x, training=self.training)
        
        # Flatten
        x = x.view(-1, 32 * 32 * 24)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return log_softmax tensor 
        return F.log_softmax(x, dim=1)
    
print("CNN model class defined!")