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
import random

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
        
        # Las imágenes son RGB, así que el canal es 3. Aplicamos 12 filtros en la primera capa convolucional.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # Aplicaremos un max pooling con un tamaño de kernel de 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Una segunda capa convolucional toma 12 canales de entrada y genera 12 salidas
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        # Una tercera capa convolucional toma 12 entradas y genera 24 salidas
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        
        # Una capa de eliminación elimina el 20% de las características para ayudar a prevenir el sobreajuste
        self.drop = nn.Dropout2d(p=0.2)
        
        # Nuestros tensores de imagen de 128x128 serán reducidos dos veces con un tamaño de kernel de 2. 128/2/2 es 32.
        # Por lo tanto, nuestros tensores de características ahora son de 32 x 32, y hemos generado 24 de ellos
        # Necesitamos aplanar estos y alimentarlos a una capa totalmente conectada
        # para asignarlos a la probabilidad de cada clase
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # Usar una función de activación ReLU después de la capa 1 (convolución 1 y pool)
        x = F.relu(self.pool(self.conv1(x)))
      
        # Usar una función de activación ReLU después de la capa 2 (convolución 2 y pool)
        x = F.relu(self.pool(self.conv2(x)))
        
        # Seleccionar algunas características para eliminar después de la tercera convolución para prevenir el sobreajuste
        x = F.relu(self.drop(self.conv3(x)))
        
        # Solo eliminar las características si este es un pase de entrenamiento
        x = F.dropout(x, training=self.training)
        
        # Aplanar
        x = x.view(-1, 32 * 32 * 24)
        # Alimentar a la capa totalmente conectada para predecir la clase
        x = self.fc(x)
        # Devolver el tensor log_softmax
        return F.log_softmax(x, dim=1)
    
print("¡Clase del modelo CNN definida!")

#Entenamiento del modelo
def train(model, device, cargador_entrenamiento, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(cargador_entrenamiento):
        # Use the CPU or GPU as appropriate
        data, target = data.to(device), target.to(device)
        
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)
        
        # Get the loss
        loss = loss_criteria(output, target)
        
        # Keep a running total
        train_loss += loss.item()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Print metrics for every 10 batches so we see some progress
        if batch_idx % 10 == 0:
            print('Training set [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                batch_idx * len(data), len(cargador_entrenamiento.dataset),
                100. * batch_idx / len(cargador_entrenamiento), loss.item()))
            
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss
            
            
def test(model, device, da):
    # Ponemos el modelo en modo evaluacion
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in da:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Cogemos las predicciones de este lote
            output = model(data)
            
            # Calculamos la perdida de este lote
            test_loss += loss_criteria(output, target).item()
            
            # Calculamos la exactitud del lote
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculamos la perdida media y la exactitud general
    avg_loss = test_loss/batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(da.dataset),
        100. * correct / len(da.dataset)))
    
    # devolvemos la perdida media
    return avg_loss
    
    
# Usamos las funciones de evaluacion y entrenamiento con el modelo    

device = "cpu"
if (torch.cuda.is_available()):
    # si esta la gpu disponible, usamos cuda
    device = "cuda"
print('Training on', device)

# Create an instance of the model class and allocate it to the device
model = Net(num_classes=len(classes)).to(device)

# Usamos el optimizador adam para ajustar los pesos
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Especificamos los criteros de perdida
loss_criteria = nn.CrossEntropyLoss()

# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 5 epochs (in a real scenario, you'd likely use many more)
epochs = 5
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, cargador_entrenamiento, optimizer, epoch)
        test_loss = test(model, device, cargador_prueba)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        

#visualizamos la perdida
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

#matriz de confusion
from sklearn.metrics import confusion_matrix

# ponemos el modelo en modo evaluacion
model.eval()

# Get predictions for the test data and convert to numpy arrays for use with SciKit-Learn
print("Getting predictions from test set...")
truelabels = []
predictions = []
for data, target in cargador_prueba:
    for label in target.cpu().data.numpy():
        truelabels.append(label)
    for prediction in model.cpu()(data).data.numpy().argmax(1):
        predictions.append(prediction) 

# Matriz de confusion
cm = confusion_matrix(truelabels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Shape")
plt.ylabel("Actual Shape")
plt.show()

# Guardamos el modelo
model_file = 'models/shape_classifier.pt'
torch.save(model.state_dict(), model_file)
del model
print('model saved as', model_file)

# funcion para predecir la clase de una imagen
def predict_image(classifier, image):
    import numpy
    
    classifier.eval()
    
    # Ponemos las mismas condiciones que pusimos al principio
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Preprocesamos las imagenes
    image_tensor = transformation(image).float()

    # Ponemos una dimension extra en los lotes
    image_tensor = image_tensor.unsqueeze_(0)

    input_features = Variable(image_tensor)

    # Predecimos la clase de la imagen
    output = classifier(input_features)
    index = output.data.numpy().argmax()
    return index


# Funcion para crear una imagen de un cuadrado, trianguo o circulo aleatoriamente
def create_image (size, shape):
    from random import randint
    import numpy as np
    from PIL import Image, ImageDraw
    
    xy1 = randint(10,40)
    xy2 = randint(60,100)
    col = (randint(0,200), randint(0,200), randint(0,200))

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    if shape == 'circle':
        draw.ellipse([(xy1,xy1), (xy2,xy2)], fill=col)
    elif shape == 'triangle':
        draw.polygon([(xy1,xy1), (xy2,xy2), (xy2,xy1)], fill=col)
    else: # square
        draw.rectangle([(xy1,xy1), (xy2,xy2)], fill=col)
    del draw
    
    return np.array(img)

# Creamos una imagen de prueba aleatoria
classnames = os.listdir(os.path.join('data', 'shapes'))
classnames.sort()
shape = classnames[random.randint(0, len(classnames)-1)]
img = create_image ((128,128), shape)

# Display the image
plt.axis('off')
plt.imshow(img)

# Create a new model class and load the saved weights
model = Net()
model.load_state_dict(torch.load(model_file))

# Call the predction function
index = predict_image(model, img)
print(classes[index])