from utils import RNN, device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader
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


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test=SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence


latentdim = 16

rnn = RNN(DIM_INPUT, latentdim, CLASSES)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)
accuracy = Accuracy()

epochs = 100



for i in range(epochs):
    Loss_train = []
    acc_train = []
    
    for seq_batch, label_batch in data_train:
        optimizer.zero_grad()
        H = rnn(seq_batch.permute(1, 0, 2))
        preds= rnn.decode(H[-1]) # il n'y a que la sortie de la fin de la séquence qui nous intéresse ici.

        loss = criterion(preds, label_batch)
        loss.backward()
        optimizer.step()
        Loss_train.append(loss.item())
        
        #Accuracy
        acc = accuracy(torch.argmax(preds,dim=1) , label_batch)
        acc_train.append(acc.item())

    
    Loss_test = []
    acc_test = []
    with torch.no_grad():
        for seq_batch, label_batch in data_test:
            H = rnn(seq_batch.permute(1, 0, 2))
            preds = rnn.decode(H[-1])

            loss = criterion(preds, label_batch)
            Loss_test.append(loss.item())
            
            #Accuracy
            acc = accuracy(torch.argmax(preds,dim=1) , label_batch)
            acc_test.append(acc.item())

    print(f"Itérations {i}: loss_train {np.mean(Loss_train)} / loss_test {np.mean(Loss_test)} / acc_train {np.mean(acc_train)} / acc_test {np.mean(acc_test)}")