import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click

from datamaestro import prepare_dataset
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime



# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05

def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


#  TODO:  Implémenter


################################################################################

                                    # DONNEES

################################################################################


ds=prepare_dataset("com.lecun.mnist")

train_img, train_labels = ds.train.images.data(), ds.train.labels.data()
test_img, test_labels = ds.test.images.data(), ds.test.labels.data()


# convert to tensor

train_img, train_labels = torch.tensor(train_img), torch.tensor(train_labels)
test_img, test_labels = torch.tensor(test_img), torch.tensor(test_labels)




################################################################################

                                    # DATASETS

################################################################################


#On a besoin d'une classe dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class dataset(Dataset):
    def __init__(self, X, y):
        self.X = X.view(-1,D_in*D_in) # flat (300,28,28) to (300,784)
        self.y = y
        self.len = len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len



################################################################################

                                    # MODELES

################################################################################

D_in = train_img.shape[2]
D_out = 10


class NetDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(D_in*D_in, 100)  # (784,100)
        self.do1 = nn.Dropout(0.2)
        self.l2 = torch.nn.Linear(100, 100)
        self.do2 = nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(100, 100)
        self.do3 = nn.Dropout(0.2)
        self.l4 = torch.nn.Linear(100, D_out)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.do1(self.l1(x))
        x = self.do2(self.l2(x))
        x = self.do3(self.l3(x))
        x = self.sig(self.l4(x))
        return x


class NetBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(D_in*D_in, 100)  # (784,100)
        self.bn1 = nn.BatchNorm1d(100) # le but du batch Norm est de centrer/ réduire les données , on le place avant le dropout
        #self.do1 = nn.Dropout(0.2)
        self.l2 = torch.nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        #self.do2 = nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(100, 100)
        self.bn3 = nn.BatchNorm1d(100)
        #self.do3 = nn.Dropout(0.2)
        self.l4 = torch.nn.Linear(100, D_out)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        #x = self.do1(self.bn1(self.l1(x)))
        #x = self.do2(self.bn2(self.l2(x)))
        #x = self.do3(self.bn3(self.l3(x)))
        #x = self.sig(self.l4(x))
        
        x = self.bn1(self.l1(x))
        x = self.bn2(self.l2(x))
        x = self.bn3(self.l3(x))
        x = self.sig(self.l4(x))
        
        return x


class NetLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(D_in*D_in, 100)  # (784,100)
        self.ln1 = nn.LayerNorm(100) 
        #self.do1 = nn.Dropout(0.2)
        self.l2 = torch.nn.Linear(100, 100)
        self.ln2 = nn.LayerNorm(100) 
        #self.do2 = nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(100, 100)
        self.ln3 = nn.LayerNorm(100) 
        #self.do3 = nn.Dropout(0.2)
        self.l4 = torch.nn.Linear(100, D_out)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        
        x = self.ln1(self.l1(x))
        x = self.ln2(self.l2(x))
        x = self.ln3(self.l3(x))
        x = self.sig(self.l4(x))
        
        return x



class Model_Initial(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(D_in*D_in, 100)  # (784,100)
        self.l2 = torch.nn.Linear(100, 100)
        self.l3 = torch.nn.Linear(100, 100)
        self.l4 = torch.nn.Linear(100, D_out)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.sig(self.l4(x))
        
        return x


#model = Model_Initial()
model = NetDropout()
# model = NetBatchNorm()
# model = NetLayerNorm()

learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-2   ## le pas d'apprentissage
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # l'optimiseur

batch_size = 300

    

################################################################################

                                    # SPLIT DES DONNEES

################################################################################
  

# On prend que 5 % des données d'entrainement
X_train, X_, y_train, y_ = train_test_split( train_img, train_labels, test_size=0.95, random_state=42)

train_data = dataset(X_train, y_train)
test_data = dataset(test_img , test_labels)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True)



################################################################################

                                    # APPRENTISSAGE

################################################################################

# Utilise ces lignes de commandes dans le terminal annaconda
# se mettre dans le repertoire où il y a dossier runs ( pas dans runs, avant)
#tensorboard --logdir=runs

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs


writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Nepochs = 1000 


for epoch in range(Nepochs):
    iteration_loss_train = []
    model.train()
    
    for i, data in enumerate(train_loader):
        X, y = data
        
        optimizer.zero_grad()   
        prediction = model(X.float())  
        loss_train = criterion(prediction, y) 
        
        #Replaces abs() with pow(2.0) for L2 regularization
     
        l2_lambda = 0.001
        l2_norm = sum(p.abs().sum() for p in model.parameters())
     
        loss_train = loss_train + l2_lambda * l2_norm
        
        loss_train.backward()      
        optimizer.step() 
        iteration_loss_train.append(loss_train.item())
        
    #test

    with torch.no_grad():
        iteration_loss_test = []
        model.eval()
        
        for data, label in test_loader:
            
            data=data.to(device)
            pred= model(data.float())
            pred=pred.to(device)
            loss_test = criterion(pred, label)
            iteration_loss_test.append(loss_test.item())


    # Affichage des résultats
    Loss_train = np.array(iteration_loss_train).mean()
    Loss_test = np.array(iteration_loss_test).mean()
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', Loss_train, epoch)
    writer.add_scalar('Loss/test', Loss_test, epoch)
    if(epoch % 10 == 0):
        print(f"epoch {epoch}: loss_train {Loss_train} / loss_test {Loss_test}")




# Remarque Question 2 ( Régularisation L1 / L2)
"""
On constate qu'en utilisant une régularisation L2 , le calcule de la courbe d'erreur en train
est un peu plus long ( 1m 56s vs 2m 41 s).
De plus, on remarque que la courbe d'erreur en test coverge plus vers 1.59 ,
tandis qu la courbe d'erreur en test calculée à partir du modèle initial sans régularisation
converge plus vers 1.60.
Par contre, en ce qui concerne l'erreur en train, lorsque qu'on utilise une régularisation L2,
l'erreur en train converge vers 1.58 tandis que celle sans régularisation converge vers 1.51.

Ainsi, on constate une petite amélioration sur la courbe d'erreur en test en utilisant une régularisation
 L2, mais une augmentation de l'erreur sur la courbe en train. 
On peut dire que le modèle généralise mieux, car avec la régularisation L2, l'erreur en Test est moins
importante; même si celle en train est un peu plus importante.

En ce qui concerne la régularisation L1, on remarque le même comportement. 
C'est à dire que l'erreur
en test avec Régularisation L1 est un peu plus faible que celle calculée sans régularisation
(1.59 vs 1.6),  et l'erreur en train est un peu plus élevé lorsqu'on utilise un erégularisation L1
(1.72 vs 1.51)
"""

# Remarque Question 3 ( Dropout) ( seul sans régularisation L1/L2)
"""
On observe qu'avec le Dropout , l'erreur en test est un peu meilleur , mais 
l'erreur en train est un peu moins meilleur.
"""

# Remarque Queestion 4 (Batch Norm)
"""
En appliquant un BatchNorm avant chaque dropout, on remarque que l'erreur  en train et test est plus 
importante que celles obtenues avec le modèle initiale sans régularisation. 
Si on applique seulement les couches de batchNorm sans les dropout, on remarque que
l'erreur en test et en train est meilleur que la précédente.
Si on compare ce modèle au modèle initiale sans régularisation, on constate que 
le temps d'éxécution reste un peu plus long que le modèle initial ( 2m 21s vs  1m 56s / 
et en test on a : la même chose)

"""

#Remarque Question 5
"""
On remarque qu'appliquer une couche LayerNorm entre chaque couche Linéaire , rend l'apprentissage
plus long par rapport au modèle initiale sans régularisation.
La courbe d'erreur en train pour le modèle initial sans régularisation est calculée en : 1 min 56 s,
tandis qu'en utilisant des couches Layer norm elle est calculée en : 4 min 54s. C'est la mêS
me différence
en Test.

"""

# TEST régularisation L1 avec Dropout 

"""
La courbe  d'erreur en train de ce modèle converge vers 1.88 tandis pour le modèle initial
sans régularisation, la courbe d'erreur en train convergait vers 1.51.

Et, la courbe d'erreur en test de ce modèle converge vers 1.59 tandis pour le modèle initial
sans régularisation, la courbe en test convergait vers 1.60.

Donc , l'erreur  en train de ce modèle est plus importante, mais l'erreur en test est meilleure.
Ainsi, ce modèle généralise mieux.

"""













