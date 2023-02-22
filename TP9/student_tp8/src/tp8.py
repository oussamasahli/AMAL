

# -----------------------------------------------------------------------------
# Import des librairies

import logging
from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)
import heapq
from pathlib import Path
import gzip
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
from tp8_preprocess import TextDataset
import pandas as pd
import numpy as np


#------------------------------------------------------------------------------

#                             Nombre de classes

# -----------------------------------------------------------------------------


"""

Il y a seulement deux classes dans les données (0, 1) et pas trois (0,1,2)

"""




# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)


# -----------------------------------------------------------------------------

#                                   CHARGEMENT DES DONNEES
    
# -----------------------------------------------------------------------------   
    


def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train) - val_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)




# -----------------------------------------------------------------------------

#                                   DATA LOADER
    
# -----------------------------------------------------------------------------   
    

train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)



# -----------------------------------------------------------------------------

#                                   TODO :
    
# -----------------------------------------------------------------------------   



# -----------------------------------------------------------------------------

#                     Combinaison possible de kernel_size et stride_size
#                     en fonction de l'architecture du CNN

# -----------------------------------------------------------------------------

# On a définit nos batch de sorte que chaque élément soit de taille 153
# nb_op est le  nombre d'opération (convolution,pooling)
# Pour l'instant on se limite à 1 ou 2 opération
# on va tester si quelques combinaisons sont possibles

def combinaison(nb_op, embed_dim1, embed_dim2):
    
    x = torch.zeros(500,1,153)  # taille d'un batch quelconque
    
    kernel_size = [i for i in range(1,5)]
    stride_size = [i for i in range(1,5)]
    possibilites = []
    
    # -------------------------------------------------------------------------
    # 1 opération (convolution,pooling)
    
    if(nb_op) == 1:
        
        for s in stride_size :
            for k in kernel_size :
                try :
                    conv = nn.Conv1d( 1,embed_dim1, kernel_size = k, stride = s )
                    pool = nn.MaxPool1d( kernel_size = k, stride = s )
                    
                    t = pool(conv(x))
                    possibilites.append((k,s))
                except:
                    print("")
     
    # -------------------------------------------------------------------------
    # 2 opérations (convolution,pooling)
                
    if(nb_op == 2):
        
        for s in stride_size :
            for k in kernel_size :
                try :
                    conv1 = nn.Conv1d( 1,embed_dim1, kernel_size = k, stride = s )
                    pool1 = nn.MaxPool1d( kernel_size = k, stride = s )
                    
                    conv2 = nn.Conv1d( embed_dim1, embed_dim2, kernel_size = k, stride = s )
                    pool2 = nn.MaxPool1d( kernel_size = k, stride = s )
                    
                    t = pool2(conv2(pool1(conv1(x))))
                    possibilites.append((k,s))
                except:
                    print("")
        
    
    return possibilites

    


# -----------------------------------------------------------------------------

#                                   ACCURACY
    
# -----------------------------------------------------------------------------   


def accuracy(preds,y):
    
    acc_y = torch.zeros(y.shape[0])
    
    for i in range(0,len(y)):
        acc_y[i] = ( torch.argmax(preds[i]).item() )
        
    correct = (acc_y == y).float().sum().item()
    
    return correct / y.shape[0]


# -----------------------------------------------------------------------------

#                                 TRIVIAL                             

# -----------------------------------------------------------------------------
# Retourne la classe majoritaire

class Model_Trivial(nn.Module):
    
    def __init__(self) :
        super().__init__()
    
    def forward(self, x):
        
        pred = torch.zeros(x.shape[0],2)
        
        occ = torch.bincount(x)
        classe_maj =  torch.argmax(occ).item()
        
        if(classe_maj == 0):
            pred[:,:1] = 1
        else:
            pred[:,1:] = 1
        
        return pred
    
    

# -----------------------------------------------------------------------------


#                                    APPRENTISSAGE TRIVIAL

#------------------------------------------------------------------------------    
# Auncun paramètres à apprendre, donc pas besoin de caclul de gradient


trivial = Model_Trivial()
criterion = nn.CrossEntropyLoss()

print("\nLancement de l'apprentissage du modèle TRIVIAL")



Loss_train = []
acc_train = []


for x_train, y_train in train_iter :
        
    preds = trivial(y_train)
    loss = criterion(preds, y_train)
    Loss_train.append(loss.item())
        
    acc_train.append(accuracy(preds,y_train))
     
Loss_val = []
acc_val = []
    
        
for x_val, y_val in val_iter :
            
    preds= trivial(y_val)
    loss = criterion(preds, y_val)
    Loss_val.append(loss.item())
            
    acc_val.append( accuracy(preds,y_val) )
    
    
# -------------------------------------------------------------------------
# Affichage de la Loss
    
l_t = round (np.mean(Loss_train),3)
l_v = round(np.mean(Loss_val),3)
a_t = round(np.mean(acc_train),3)
a_v = round(np.mean(acc_val),3)
    
    
print("\nLoss-train :", l_t, " / Loss-val :",l_v, " / acc_train :",a_t, " / acc_val :", a_v )





# Pour le modèle trivial,  on obtient :
    
# Loss-train : 0.795  / Loss-val : 0.806  / acc_train : 0.518  / acc_val : 0.507


    

# -----------------------------------------------------------------------------

#                                   CNN

# -----------------------------------------------------------------------------




class CNN(nn.Module):
    
    def __init__(self, embed_dim_1, embed_dim_2, kernel_size_1, kernel_size_2, stride_size) :
        
        super().__init__()
    
        self.conv1 = nn.Sequential(
            
        nn.Conv1d( 1, embed_dim_1, kernel_size = kernel_size_1, stride = stride_size ),
        nn.ReLU(),
        
        nn.MaxPool1d( kernel_size = kernel_size_1, stride = stride_size ),
        nn.Dropout(0.1)
        
        )    
        
        
        self.conv2 = nn.Sequential(
            
        nn.Conv1d(embed_dim_1, embed_dim_2, kernel_size = kernel_size_2, stride = stride_size),
        nn.ReLU(),
        
        nn.MaxPool1d(kernel_size = kernel_size_2, stride = stride_size),
        nn.Dropout(0.25)
        
        ) 
            

        #----------------------------------------------------------------------
        # Calcul de la dimension de l'entrée de la couche linéaire apès
        # toute les convolutions et maxpooling : dim 
        
        
        # ---------------------------------------------------------------------
        # Calcul de la taille de la sortie après la 1 ère transformation
        
        if (kernel_size_1 <= stride_size):
            t1 = 153/ stride_size
        else:
            t1 =  int ( str( (153-kernel_size_1) / stride_size ).split('.')[0] ) +1
        
        if(kernel_size_2 <= stride_size):
            dim = t1/stride_size
        else :
            dim = int ( str( (t1-kernel_size_2) / stride_size ).split('.')[0] ) +1
        
        #----------------------------------------------------------------------
        # Calcul de la taille de la sortie  2 ème transformation
        
        

        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim_2 * int(dim), 100), # dim  est la taille de l'entrée après l'application des opération (convolution,pooling)
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        
        self.fc2 = nn.Sequential(
        nn.Linear(100, 2),
        nn.ReLU(),
        nn.Dropout(0.5)
    )
    


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
            
        # on passe de (500,8,153) à (500,153,8)
        # chacune des 153 dimensions est représenté par un vecteur de embed_dim dimensions
        
        x = x.transpose(1, 2)  
        
        # on a plaatit les données pour la couche linéaire , on passe de (500,153,8) à (500,153*8)
        x = torch.reshape(x, (x.shape[0],-1))

        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
        
    


# -----------------------------------------------------------------------------

#                                   Paramètres du CNN

#------------------------------------------------------------------------------

epochs = 10

embed_dim_1 = 4
embed_dim_2 = 8



# -----------------------------------------------------------------------------


#                                    APPRENTISSAGE CNN

#------------------------------------------------------------------------------


possibilites = combinaison(2, embed_dim_1, embed_dim_2 )

for (k,s) in possibilites :
    
    kernel_size_1 = k
    kernel_size_2 = k
    stride_size = s
    
    model = CNN( embed_dim_1, embed_dim_2, kernel_size_1, kernel_size_2, stride_size)
    
    learning_rate = 0.001
    
    optimizer = torch.optim.Adam( model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print("#--------------------------------------------------------#")
    print("\nkernel_size_1 :", k,"kernel_size_2:",k, "stride_size:",s)
    print("\nLancement de l'apprentissage du CNN")
    
    for i in range(epochs):
        
        Loss_train = []
        acc_train = []
    
    
        for x_train, y_train in train_iter :
            
            optimizer.zero_grad()
            
            # ---------------------------------------------------------------------
            
            # Padding sur les Batch
            # les données des batchs n'ont pas la même taille entre les différents batchs
            # et je dois définir une taille spécifique dans la prédiction du cnn
            
            # On dira que la taille maximale d'une entrée d'un batch est de 153 dimensions 
            # car c'est le max de dim pour une entrée dans train
            # on ajoute que du padding à droite 
            
            # Pour résumé : Dans le Dataloader on applique déjà un premier padding
            # pour que les éléméents d'un même batch soient de même dimensions
            # mais il y a un autre problème, les batchs n'ont pas la même taille
            # par exemple on peut avoir (500,63) et (500,115), et on 
            # a besoin d'un taille unique pour la prédiction (Linear) du CNN : nn.Linear(embed_dim * 153, 2) 
            # donc on met tous les batch a une taille de 153 dimensions en ajoutant du padding à droite 
            
            pad = 153 - x_train.shape[1]
            
            x = nn.functional.pad(x_train, (0, pad), "constant", 0)
            
    
            # ---------------------------------------------------------------------
            # reshape des données
            
            x = pd.DataFrame(x)
            x = torch.tensor(x.values, dtype=torch.float32)
            
            # on passe de (500,153) à (500,1,153) pour pouvoir appliquer Conv1D
            x = x.unsqueeze(dim=1)
            
            preds = model(x)
    
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
                
                pad = 153 - x_val.shape[1]
                
                x = nn.functional.pad(x_val, (0, pad), "constant", 0)
                
                # ---------------------------------------------------------------------
                # reshape des données
                
                x = pd.DataFrame(x)
                x = torch.tensor(x.values, dtype=torch.float32)
                x = x.unsqueeze(dim=1)
                
                preds= model(x)
    
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
        
    

    
