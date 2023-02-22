import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO: 
    
from torch.utils.data import random_split
    
PATH = "data/"
text = ""
BATCH_SIZE = 32

with open(PATH+'trump_full_speech.txt') as f:
    for line in f.readlines():
        text = text+line
    
data = TrumpDataset(text)
lenght = data.__len__()

train_data, test_data = random_split(data, [round(lenght*0.8), round(lenght*0.2)])

data_train = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# APPRENTISSAGE

n_vocab =  len(LETTRES)+1 #Taille du vocabulaire
embedding_dim = 16
latentdim=16

embedding = torch.nn.Embedding(n_vocab,embedding_dim)

rnn = RNN(embedding_dim, latentdim, n_vocab)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)

epochs = 1
import numpy as np

for i in range(epochs):
    Loss_train = []
    for seq_batch, label_batch in data_train:
        optimizer.zero_grad()
        
        X = torch.unsqueeze(seq_batch, 2).permute(1,0,2).to(torch.float)
        y= torch.unsqueeze(label_batch, 2).permute(1,0,2).to(torch.float)
        V=[]
        for k, xt in enumerate(X):
            e=embedding(xt.to(torch.long))
            V.append(e)
        
        H = rnn(V)
        
        S=[]
        for j, ht in enumerate(H):
            preds=rnn.decode(ht)
            S.append(preds)
        loss = criterion(S, y)
        
        loss.backward()
        optimizer.step()
        Loss_train.append(loss.item())
        
    print(f"Itérations {i}: loss_train {np.mean(Loss_train)}")
    
    
    
    
    