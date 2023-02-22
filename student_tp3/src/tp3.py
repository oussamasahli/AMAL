from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime


# Téléchargement des données

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()


# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)



#  TODO: 

# Question 1
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Dataset_MNIST(Dataset):

	def __init__(self, x, y):
		super(Dataset_MNIST, self).__init__()
		self.labels = torch.from_numpy(y)
		self.data = torch.from_numpy(x).float()
	def __getitem__(self, index):
		return self.data[index], self.labels[index]
	def __len__(self):
		return len(self.labels)
    
    
train_dataset = Dataset_MNIST(train_images, train_labels)
trainloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

#for data, label in trainloader:
   # print(data.shape,'//',label.shape)
   
   
# question 2

class AutoEncoder(nn.Module):
    
	def __init__(self, dim, lat):
		super(AutoEncoder, self).__init__()
		self.encoder = torch.nn.Linear(dim, lat)
		self.decoder = torch.nn.Linear(lat, dim)

	def encode(self, x):
		return nn.functional.relu(self.encoder(x))

	def decode(self, x):
		return torch.sigmoid(self.decoder(x))

	def forward(self, x):
		return self.decode(self.encode(x))
    
    

# question 3

print("train shapes: ", train_images.shape, train_labels.shape)
print("test shapes: ", test_images.shape , test_labels.shape )
    
   
# Normalisation des images entre 0 et 1
    
train_images = train_images.astype(float)
test_images = test_images.astype(float)    
   
    
train_images /= 255
test_images /= 255 
    
# Dataset et Dataloader

train_dataset = Dataset_MNIST(train_images, train_labels)
trainloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = Dataset_MNIST(test_images , test_labels)
testloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

lat=16   
iterations = 25

# Modele / cout / optimizer
# Checkpointing
savepath = Path("model.pch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class State:
    def __init__(self, model, optim):
        self.model=model
        self.optim = optim
        self.epoch, self.iteration = 0,0
        

if savepath.is_file():
    with savepath.open('rb') as fp :
        state = torch.load(fp)       # on recommence depuis le modèle sauvegardé

else:
    print("Aucun modèle sauvegardé auparavant")
    model = AutoEncoder(train_images.shape[1], lat) 
    model=model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    state= State(model,optimizer)



# decscente de gradient pour l'autoencodeur
    
for i in range(state.iteration,iterations):

    train_losses = []
    for data, label in trainloader:
        
        state.model.train()
        state.optim.zero_grad()
        
        data=data.to(device)
        code = state.model(data)
        code=code.to(device)
        loss = criterion(data, code)
    
        loss.backward()
        state.optim.step()
        
        train_losses.append(loss.data)

    test_losses = []
    
    with torch.no_grad():
        for data, label in testloader:
            state.model.eval()
            
            data=data.to(device)
            code = state.model(data)
            code=code.to(device)
            loss = criterion(data, code)

            test_losses.append(loss.data)

    
    # Affichage des résultats
    Loss_train = np.array(train_losses).mean()
    Loss_test = np.array(test_losses).mean()
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', Loss_train, i)
    writer.add_scalar('Loss/test', Loss_test, i)
    print(f"Itérations {i}: loss_train {Loss_train} / loss_test {Loss_test}")

    
   
    
   
    
    
    
    