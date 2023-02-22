from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch
import numpy as np
from torchmetrics import Accuracy

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Question 3 : Prédiction de séries temporelles


latentdim = 16
output=2

rnn = RNN(DIM_INPUT, latentdim, output)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)

epochs = 25


for i in range(epochs):
    Loss_train = []
    
    for seq_batch, label_batch in data_train:
        optimizer.zero_grad()
        
        d=seq_batch.permute(2, 1, 0, 3)
        y=label_batch.permute(2, 1, 0, 3)
        
        #pour chaque classe (statio)
        for j, xt in enumerate(d):
            H = rnn(xt)
            for e, ht in enumerate(H): # on décode l'état caché à chaque pas de temps
                preds= rnn.decode(ht)

                loss = criterion(preds, y[j][e])
                optimizer.step()
                Loss_train.append(loss.item())
                
            loss.backward() # rétropapagtion une fois que toute la séquence a été vue.


    Loss_test = []
    with torch.no_grad():
        for seq_batch, label_batch in data_test:

            d=seq_batch.permute(2, 1, 0, 3)
            y=label_batch.permute(2, 1, 0, 3)

            for j, xt in enumerate(d):
                H = rnn(xt)
                for e, ht in enumerate(H): 
                    preds= rnn.decode(ht)
                    loss = criterion(preds, y[j][e])
                    optimizer.step()
                    Loss_test.append(loss.item())


    print(f"Itérations {i}: loss_train {np.mean(Loss_train)} / loss_test {np.mean(Loss_test)}")
