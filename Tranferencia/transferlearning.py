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
from sklearn.metrics import confusion_matrix

# PREPARAMOS EL MODELO BASE
# Cargamos el modelo preentrenado
model = torchvision.models.resnet34(pretrained=True)
print(model)

# PREPARAMOS LOS DATOS
# Funcion para cargar los datos usando cargadores de entrenamiento y validacion
def load_dataset(data_path):
    
    # Cambiamos el tamaño a 256 x 256, recortamos el centro a 224x224 (tamaño de entrada de ResNet)
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # cargamos todas las imagenes, transformandolas
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )
    
    # dividimos el dataset en entrenamiento (70%) y validacion (30%)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # definimos un cargador para los datos de entrenamiento que podemos iterar en lotes de 30 imagenes
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=30,
        num_workers=0,
        shuffle=False
    )
    
    # hacemos lo mismo para los datos de validacion
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=30,
        num_workers=0,
        shuffle=False
    )
        
    return train_loader, test_loader


# cargamos las imagenes
import os  
data_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenes")
print('DIRECTORIO: ', data_path)

# cogemos los cargadores de datos de entrenamiento y validacion
train_loader, test_loader = load_dataset(data_path)

# obtenemos los nombres de las clases
classes = os.listdir(data_path)
classes.sort()
print('class names:', classes)

# CREAMOS UNA CAPA DE PREDICCION
# Ponemos las capas de extraccion de caracteristicas en modo de solo lectura
for param in model.parameters():
    param.requires_grad = False
    
# sustituimos la capa de prediccion
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
print(model)

# ENTRENAMOS EL MODELO
def train(model, device, train_loader, optimizer, epoch):
    # Ponemos el modelo en modo de entrenamiento
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Procesamos los lotes de datos
    for batch_idx, (data, target) in enumerate(train_loader):
        # Usamos cpu o gpu segun corresponda
        data, target = data.to(device), target.to(device)
        
        # Reseteamos el optimizador
        optimizer.zero_grad()
        
        # Empujamos los datos a traves del modelo
        output = model(data)
        
        # Calculamos la perdida
        loss = loss_criteria(output, target)
        
        # Acumulamos la perdida
        train_loss += loss.item()
        
        # Calculamos los gradientes
        loss.backward()
        optimizer.step()
        
        # Mostramos el progreso cada 10 lotes
        if batch_idx % 10 == 0:
            print('Training set [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    # Devolvemos la perdida media
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

def test(model, device, test_loader):
    # Ponemos el modelo en modo de evaluacion
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Pasamos los datos a traves del modelo
            output = model(data)
            
            # Calculamos la perdida
            test_loss += loss_criteria(output, target).item()
            
            # Calculamos la precision
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculamos la perdida media y la precision de la epoca
    avg_loss = test_loss/batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # Devolvemos la perdida media
    return avg_loss

# Usamos las funciones
device = 'cpu'
if (torch.cuda.is_available()):
    device = 'cuda'
print('Training on', device)

# Creamos una instancia del modelo y lo movemos al dispositivo
model = model.to(device)

# Usamos Adam como optimizador
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Especificamos los criterios de perdida
loss_criteria = nn.CrossEntropyLoss()

# Hacemos un seguimiento de la perdida y la precision en cada epoca
epoch_nums = []
training_loss = []
validation_loss = []

# Entrenamos 3 epocas
epochs = 3
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        
# MOSRTRAMOS LA PERDIDA
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

# EVALUAMOS EL MODELO
# Ponemos el modelo en modo de evaluacion
model.eval()

# Obtenemos las predicciones del conjunto de prueba
print("Getting predictions from test set...")
truelabels = []
predictions = []
for data, target in test_loader:
    for label in target.cpu().data.numpy():
        truelabels.append(label)
    for prediction in model.cpu()(data).data.numpy().argmax(1):
        predictions.append(prediction) 

# Mostramos la matriz de confusion
cm = confusion_matrix(truelabels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Shape")
plt.ylabel("Actual Shape")
plt.show()

# USAMOS EL MODELO ENTRENADO    
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

# Funcion para predecir la forma de una imagen
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

shape = classes[np.random.choice(len(classes)-1)]
img = create_image((128,128), shape)
print('Shape:', shape)
plt.imshow(img)

index = predict_image(model, img)
print(classes[index])
