import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from pathlib import Path


Batch = namedtuple("Batch", ["text", "labels"])



# -----------------------------------------------------------------------------

#                                       DATASETS 
    
#------------------------------------------------------------------------------ 


class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):

        s = self.files[ix]
        
        if (isinstance(s, str)) :
            x = self.tokenizer(s)
        else :
            
            with open(s, 'r', encoding='utf-8') as f:  
                lines = f.readlines()[0]

            x = self.tokenizer(lines)
        
        #return self.tokenizer(s if isinstance(s, str) else s.read_text()) , self.filelabels[ix]
        
        return torch.tensor(x), self.filelabels[ix]
    
    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return Batch(torch.nn.utils.rnn.pad_sequence(data, batch_first=True), torch.LongTensor(labels))
    



# -----------------------------------------------------------------------------

#                                    Récupération des données
    
#------------------------------------------------------------------------------ 


def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)



# -----------------------------------------------------------------------------

#                                       TODO: 
    
#------------------------------------------------------------------------------    
    


# -----------------------------------------------------------------------------

#                                 Chargement des données
    
#------------------------------------------------------------------------------  


    
word2id, embeddings, ds_train, ds_test = get_imdb_data()
    
id2word = dict(zip(word2id.values(),word2id.keys()))

# La dimension maximal d'un exemple est 2470


# -----------------------------------------------------------------------------

#                                Baseline
    
#------------------------------------------------------------------------------ 



class Baseline(nn.Module):
    
    def __init__(self,embeddings) :
        super().__init__()
        
        self.embeddings = embeddings
        self.linear = nn.Linear(self.embeddings.shape[1], 2)
        
    def forward(self, x):
        
        emb = torch.tensor( self.embeddings[x], dtype=torch.float64 )
        t_hat = torch.mean(emb,1)
        
        return self.linear(t_hat.float())

    

# -----------------------------------------------------------------------------

#                                ATTENTION SIMPLE
    
#------------------------------------------------------------------------------ 



class Simple_Attention(nn.Module):
    
    def __init__(self,embeddings) :
        super().__init__()
        
        self.embeddings = embeddings
        self.prob_attention =  nn.Linear(self.embeddings.shape[1], self.embeddings.shape[1])
        self.linear = nn.Linear(self.embeddings.shape[1], 2)
        
    def forward(self, x):
        
        emb = torch.tensor( self.embeddings[x], dtype=torch.float64 )
        
        # ---------------------------------------------------------------------
        # Probabilité de l'attention
        
        p = torch.exp( self.prob_attention(emb.float()) )
         
        t_hat = torch.mean(emb*p,1)
        
        return self.linear(t_hat.float())
    



# -----------------------------------------------------------------------------

#                                Question et Valeur
    
#------------------------------------------------------------------------------ 

    
    
    

# ----------------------------------------------------------------------------
# Accuracy
   
def accuracy(preds,y):
    
    acc_y = torch.zeros(y.shape[0])
    
    for i in range(0,len(y)):
        acc_y[i] = ( torch.argmax(preds[i]).item() )
        
    correct = (acc_y == y).float().sum().item()
    
    return correct / y.shape[0]


# -----------------------------------------------------------------------------

#                                Apprentissage
    
#------------------------------------------------------------------------------ 

TRAIN_BATCHSIZE = 100
TEST_BATCHSIZE = 100
epochs = 25

val_size = 1000
train_size = len(ds_train) - val_size
train, val = torch.utils.data.random_split( ds_train, [train_size, val_size])
    

# -----------------------------------------------------------------------------
# DONNEES
    
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=FolderText.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=FolderText.collate)
test_iter = torch.utils.data.DataLoader(ds_test, batch_size=TEST_BATCHSIZE, collate_fn=FolderText.collate)
 

# -----------------------------------------------------------------------------
# Modèle

#model = Baseline(embeddings)
model = Simple_Attention(embeddings)

learning_rate = 0.001

optimizer = torch.optim.Adam( model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()


   
# -----------------------------------------------------------------------------
# Apprentissgae

print("\nLancement de l'apprentissage du Baseline") 
for i in range(epochs):
    
    Loss_train = []
    acc_train = []


    for x_train, y_train in train_iter :
        
        optimizer.zero_grad()
        
        preds = model(x_train)

        loss = criterion(preds, y_train)
        loss.backward() 
        
        optimizer.step()
        Loss_train.append(loss.item())
        
        # ---------------------------------------------------------------------
        # Accuracy
        
        acc_train.append(accuracy(preds,y_train))
        
     
    Loss_val = []
    acc_val = []
    
    with torch.no_grad():
        
        for x_val, y_val in val_iter :
            
            preds= model(x_val)

            loss = criterion(preds, y_val)
            Loss_val.append(loss.item())
            
            # ---------------------------------------------------------------------
            # Accuracy
            
            acc_val.append( accuracy(preds,y_val) )
    
    
    # -------------------------------------------------------------------------
    # Affichage de la Loss
    
    l_t = round (np.mean(Loss_train),3)
    l_v = round(np.mean(Loss_val),3)
    a_t = round(np.mean(acc_train),3)
    a_v = round(np.mean(acc_val),3)
        
        
    print("\nepoch :",i,"Loss-train :", l_t, " / Loss-val :",l_v, " / acc_train :",a_t, " / acc_val :", a_v )    
    
    
    
# -----------------------------------------------------------------------------
# Au bout de la 25 ème époque,  en utilisant le modèle Baseline, on obtient les résultats :
    
#  epoch : 24 Loss-train : 0.647  / Loss-val : 0.65  / acc_train : 0.655  / acc_val : 0.639

# -----------------------------------------------------------------------------
# Au bout de la 25 ème époque,  en utilisant le modèle Simple_Attention on obtient les résultats :

# epoch : 24 Loss-train : 0.38  / Loss-val : 0.38  / acc_train : 0.839  / acc_val : 0.839




















    
