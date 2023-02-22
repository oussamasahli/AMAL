# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:02:43 2022

@author: osahl
"""


################################################################################

                                    # OPTUNA

################################################################################



import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datamaestro import prepare_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



D_in = 28
D_out = 10
batch_size = 300


class my_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X.view(-1,D_in*D_in) # flat (300,28,28) to (300,784)
        self.y = y
        self.len = len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len
    
    
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
    
    


# Build a model by implementing define-by-run design from Optuna
def build_model_custom(trial):

    return Model_Initial()




# Train and evaluate the accuracy of neural network with the addition of pruning mechanism
def train_and_evaluate(param, model, trial):
    
    # DONNEES 
    
    ds=prepare_dataset("com.lecun.mnist")

    train_img, train_labels = ds.train.images.data(), ds.train.labels.data()
    #test_img, test_labels = ds.test.images.data(), ds.test.labels.data()
    # convert to tensor

    train_img, train_labels = torch.tensor(train_img), torch.tensor(train_labels)
    #test_img, test_labels = torch.tensor(test_img), torch.tensor(test_labels)
    
    
    # On prend que 5 % des données d'entrainement
    #X, X_, y, y_ = train_test_split( train_img, train_labels, test_size=0.95, random_state=42)
 
    #X_train, X_val , y_train, y_val = train_test_split(X, y,  test_size = 0.05, random_state = 42)
    """
    je vais pas garder que  5 % des données train de base , parceque sinon l'accuracy calculé est trop faible ( 0 sur le jeu de validation)
    """
    
    
    
    X_train, X_val , y_train, y_val = train_test_split( train_img, train_labels,  test_size = 0.05, random_state = 42)

    train_data = my_dataset(X_train, y_train)
    val_data = my_dataset(X_val , y_val)

    train_dataloader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True, drop_last=True)


    
    # APPRENTISSAGE

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(EPOCHS):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in train_dataloader:

                train_label = train_label.to(device)
                train_input = train_input.to(device)

                output = model(train_input.float())
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            
            accuracy_train = total_acc_train/len(train_data)
            #if(epoch_num % 10 == 0):
               # print("epoch :",epoch_num, ", accuracy_train : ",accuracy_train, )
                
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    val_input = val_input.to(device)

                    output = model(val_input.float())
                

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            accuracy_val = total_acc_val/len(val_data)
            
            #if(epoch_num % 10 == 0):
              #  print("epoch :",epoch_num, ", accuracy_val : ",accuracy_val)
            
            # Add prune mechanism
            trial.report(accuracy_val, epoch_num)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return accuracy_val # c'est l'accuracy sur le jeu de validation que l'on cherche à optimiser 
  
# Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
def objective(trial):

     params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              # 'n_unit': trial.suggest_int("n_unit", 15, 100),
              }
    
     model = build_model_custom(params) # on pourrait modifier aussi la couche lattente 15 au lieu de 100 : 

     accuracy = train_and_evaluate(params, model, trial)

     return accuracy
  
  

EPOCHS = 50 # je diminu à 50 parceque il y a plus de données vu que j'ai ps laissé que 5 % ( jai tous pris)
    
"""
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=10)



###############################################################################

                            # MEILLEUR HYPERPARAMETRES

###############################################################################



best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))
    
""" 
# BEST IS :
    
#learning_rate: 0.011456891749356934
#optimizer: SGD
    

###############################################################################

                            # APPLICATION SUR DONNES TESTS

###############################################################################    

# On reconstruit le modèle avec les meilleurs paramètres trouvés.
    
    
ds=prepare_dataset("com.lecun.mnist")

train_img, train_labels = ds.train.images.data(), ds.train.labels.data()
test_img, test_labels = ds.test.images.data(), ds.test.labels.data()
 # convert to tensor

train_img, train_labels = torch.tensor(train_img), torch.tensor(train_labels)
test_img, test_labels = torch.tensor(test_img), torch.tensor(test_labels)
    
# On prend que 5 % des données d'entrainement
#X, X_, y, y_ = train_test_split( train_img, train_labels, test_size=0.95, random_state=42)
#X_train, X_val , y_train, y_val = train_test_split(X, y,  test_size = 0.05, random_state = 42)
"""
je vais pas garder que  5 % des données train de base , parceque sinon l'accuracy calculé est trop faible ( 0 sur le jeu de validation)
"""
    
X_train, X_val , y_train, y_val = train_test_split( train_img, train_labels,  test_size = 0.05, random_state = 42)

train_data = my_dataset(X_train, y_train)
val_data = my_dataset(X_val , y_val)

train_dataloader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True, drop_last=True)

    

###############################################################################

                            # APPRENTISSAGE

###############################################################################

model = Model_Initial()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = getattr(optim, 'SGD')(model.parameters(), lr= 0.011456891749356934)

if use_cuda:

        model = model.cuda()
        criterion = criterion.cuda()

for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in train_dataloader:

            train_label = train_label.to(device)
            train_input = train_input.to(device)

            output = model(train_input.float())
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        
        accuracy_train = total_acc_train/len(train_data)
        if(epoch_num % 10 == 0):
            print("epoch :",epoch_num, ", accuracy_train : ",accuracy_train )
            
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                val_input = val_input.to(device)

                output = model(val_input.float())
            

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
        accuracy_val = total_acc_val/len(val_data)
        
        if(epoch_num % 10 == 0):
            print("epoch :",epoch_num, ", accuracy_val : ",accuracy_val)
          

print(" FIN APPRENTISSAGE ")
print("############################################")
print()
print(" DEBUT TEST ")

test_data = test_img.view(-1,D_in*D_in)

test_data = test_data.to(device)
test_labels = test_labels.to(device)
output = model(test_data.float())

test_loss = criterion(output, test_labels)

acc = (output.argmax(dim=1) == test_labels).sum().item()

print("loss_test : ", test_loss.item(), ', accuracy test : ', acc/ len(test_data))


# résultats : 
    
# loss_test :  1.5490556955337524 , accuracy test :  0.9101





    
    
