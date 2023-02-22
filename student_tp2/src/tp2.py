# -*- coding: utf-8 -*-
import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
from sklearn.model_selection import train_test_split


writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split( datax, datay, test_size=0.33, random_state=42)


# TODO: 
    
# Les paramètres du modèle à optimiser
w = torch.randn(datax.shape[1], datay.shape[1], requires_grad=True)
b = torch.randn(1,datay.shape[1], requires_grad=True)

#epsilon = 0.05
epsilon = 0.001 # c'est important de laisse ce learning rate parceque sinon la loss oscille

writer = SummaryWriter()

# FONCTIONS

#MSE
def mse(y_pred,y):
    q=y.shape[0]
    mse = 1/q * torch.linalg.norm(torch.sub(y_pred,y))
    return mse
 
# LINEAR
def Linear(x,w,b):
    f = torch.add(torch.mm(x,w), b)
    return f




"""
# DESCENTE DE GRADIENT CLASSIQUE

for n_iter in range(100):
    
    # forward Linear
    y_hat_train = Linear(X_train,w, b)
    y_hat_train.retain_grad()
    
    # loss
    loss_train = mse(y_hat_train,y_train)
    # backward loss
    loss_train.backward()
    

    ##  TODO:  Mise à jour des paramètres du modèle
    with torch.no_grad(): # Calculs effectué sans graphes de calculs
        #w = torch.sub(w,epsilon*w.grad) # on pointe sur un nouveau w, donc pas pas besoin de mettre son gradient à 0
        #b = torch.sub(b,epsilon*b.grad) # pareil ici
        w -= epsilon*w.grad # ici cest le même pointeur
        b -= epsilon*b.grad
        
        # les calculs pour le test sont à faire ici / car l'ensemble de test ne doit pas participer à la mise à jour du gradient
        y_hat_test = Linear(X_test,w, b)
        loss_test = mse(y_hat_test,y_test)
        
        # Remise à zéros des gradients
        w.grad.data.zero_() 
        b.grad.data.zero_() 
        y_hat_train.grad.data.zero_() 
 
    #w.requires_grad = True
    #b.requires_grad = True
    
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss_train, n_iter)
    writer.add_scalar('Loss/test', loss_test, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss_train {loss_train} / loss_test {loss_test}")
    
    

# Utilise ces lignes de commandes dans le terminal annaconda
# se mettre dans le repertoire où il y a dossier runs ( pas dans runs, avant)
#tensorboard --logdir=runs
  




"""
# DESCENTE DE GRADIENT mini -BATCH

from torch.utils.data import Dataset
from torch.utils.data import DataLoader



#On a besoin d'une classe dataset
class dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len
"""

train_data = dataset(X_train, y_train)

batch_size = 32 # on prend des batchs (lots) de 32 exemples

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)


for n_iter in range(100):
    
    iteration_loss_train = 0
    for i, data in enumerate(train_loader):
      X, y = data

      #forward Linear
      y_hat_train = Linear(X.float(),w, b)
      y_hat_train.retain_grad()

      
      # loss
      loss_train = mse(y_hat_train, y.reshape(-1,1).float())
      loss_train.backward()
      iteration_loss_train += loss_train
     
      with torch.no_grad(): 
          # On les met à jours sur chaque batch
          w -= epsilon*w.grad 
          b -= epsilon*b.grad

          
      # On remet les gradients à 0 pours les calculs sur le prochian batch
      w.grad.data.zero_() 
      b.grad.data.zero_() 
      y_hat_train.grad.data.zero_() 

    
    Loss_train = iteration_loss_train / (i+1)

    # TEST
    with torch.no_grad(): 
        y_hat_test = Linear(X_test,w, b)
        Loss_test = mse(y_hat_test,  y_test)

    
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', Loss_train, n_iter)
    writer.add_scalar('Loss/test', Loss_test, n_iter)
    
    # Sortie directe
    print(f"Itérations {n_iter}: loss_train {Loss_train} / loss_test {Loss_test}")





# DESCENTE DE GRADIENT STOCHASTIQUE
# La modification du vecteurw doit se faire à partir de chaque exemple
for n_iter in range(100):
    
    iteration_loss_train = 0
    for i in range(0,len(X_train)):
        # forward Linear
        y_hat_train = Linear(X_train[i].reshape(1,-1).float(),w,b)
        y_hat_train.retain_grad()
        
        # loss
        loss_train = mse(y_hat_train, y_train[i].reshape(1,-1).float())
        
        # backward loss
        loss_train.backward()
        iteration_loss_train += loss_train
        
        with torch.no_grad(): 
            
            w -= epsilon*w.grad 
            b -= epsilon*b.grad
            # Remise à zéros des gradients
            w.grad.data.zero_() 
            b.grad.data.zero_() 
            y_hat_train.grad.data.zero_() 
      
    Loss_train = iteration_loss_train / (i+1)    
    
    # TEST
    with torch.no_grad(): 
        y_hat_test = Linear(X_test,w, b)
        Loss_test = mse(y_hat_test,  y_test)
        
    
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', Loss_train, n_iter)
    writer.add_scalar('Loss/test', Loss_test, n_iter)
    
    # Sortie directe
    print(f"Itérations {n_iter}: loss_train {Loss_train} / loss_test {Loss_test}")
    
"""




# AVEC OPTIMISEUR 

# Optimiseur et réseaux à deux couches :
    
class Premier_Modele(torch.nn.Module):
	def __init__(self, inputSize):
		super(Premier_Modele, self).__init__()
		self.linear1 = torch.nn.Linear(inputSize, 16)
		self.linear2 = torch.nn.Linear(16, 1)
		self.activation = torch.nn.Tanh()

	def forward(self, x):
		return self.linear2(self.activation(self.linear1(x)))
    
    
class Second_Modele(torch.nn.Module):
	def __init__(self, inputSize):
		super(Second_Modele, self).__init__()
		self.m = torch.nn.Sequential(torch.nn.Linear(inputSize, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
	def forward(self, x):
		return self.m(x)





#model = Premier_Modele(datax.shape[1])
model = Second_Modele(datax.shape[1])
loss_MSE = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr = epsilon)

train_data = dataset(X_train, y_train)

batch_size = 32 # on prend des batchs (lots) de 32 exemples

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)

for n_iter in range(100):
 
    iteration_loss_train = 0
    for i, data in enumerate(train_loader):
        X, y = data    
        model.train()
        # Forward
        y_hat_train = model(X.float())
        loss_train = loss_MSE(y_hat_train, y.reshape(-1,1).float())

        loss_train.backward()
        iteration_loss_train += loss_train
        
        optim.step()
        optim.zero_grad()
    
    Loss_train = iteration_loss_train / (i+1)
    #TEST
    with torch.no_grad():
   
        model.eval()
        
        y_hat_test = model(X_test.float())
        Loss_test = mse(y_hat_test,  y_test)


    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', Loss_train, n_iter)
    writer.add_scalar('Loss/test', Loss_test, n_iter)
    
    # Sortie directe
    print(f"Itérations {n_iter}: loss_train {Loss_train} / loss_test {Loss_test}")











