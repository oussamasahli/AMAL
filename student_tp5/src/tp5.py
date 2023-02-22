
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

#  TODO: 

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    criterion= nn.CrossEntropyLoss(reduce=None,ignore_index=padcar)
    loss = criterion(output, target)
    return loss

class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self, dim, latent, output, vocab_size, embedding_dim):
        super(RNN, self).__init__()
        self.dim = dim
        self.latent = latent
        self.output = output
        
        self.Wx = torch.nn.Linear(dim, latent)
        self.Wh = torch.nn.Linear(latent, latent)
        self.Wd = torch.nn.Linear(self.latent, self.vocab_size)
        self.embed = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x, h=None):
        """ x: sequence_length x batch x dim 
			h: batch x latent
			returns: length x batch x latent Tensor"""

        historique = []
		# pour chaque instant de la sequence:
        if h is None:
            ht=torch.zeros(x.size()[1], self.latent)
            historique.append(ht.numpy())
        for i, xt in enumerate(x):
			# ht: (batch x latent)
            #embedding
            xt = embed(xt)
            ht=self.one_step(xt, ht)
            if i> 0:
                historique.append(ht) # Ne pas enregistrer les h0

        return torch.Tensor(historique)


    def one_step(self, x, h):
        """ x: batch x dim 
			h: batch x latent
			returns: batch x latent Tensor """
        return torch.nn.functional.tanh(torch.add(self.Wx(x), self.Wh(h)))

    def decode(self,h):
        return  torch.nn.functional.softmax(self.Wd(h))


class LSTM(RNN):
    #  TODO:  Implémenter un LSTM
    def __init__(self, input_dim, latent_dim):
        super(LSTM, self).__init__()

        self.W_forget = torch.nn.Sequential(torch.nn.Linear(input_dim+latent_dim, latent_dim), torch.nn.Sigmoid())
        self.W_input = torch.nn.Sequential(torch.nn.Linear(input_dim+latent_dim, latent_dim), torch.nn.Sigmoid())
        self.W_output = torch.nn.Sequential(torch.nn.Linear(input_dim+latent_dim, latent_dim), torch.nn.Sigmoid())

        self.W_update = torch.nn.Sequential(torch.nn.Linear(input_dim+latent_dim, latent_dim), torch.nn.Tanh()) # Wc for internal memory update

    def one_step(self, x, h, c):

        ft = self.W_forget(torch.cat((x, h), 0))
        it = self.W_input(torch.cat((x, h), 0))
        Ct = ft * c + it * self.W_update(torch.cat((x, h), 0))
        ot = self.W_output(torch.cat((x, h), 0))
        ht = ot * torch.nn.Tanh(Ct)

        return ht, Ct

    def forward(self, x, h=None):

        hitorique = []
		# pour chaque instant de la sequence:
        if h is None:
            ht=torch.zeros(x.size()[1], self.latent)
        Ct = torch.zeros(x.size()[1], self.latent)
        for xt in x:
			# ht: (batch x latent)
            ht, Ct = self.one_step(xt, ht, Ct)
            historique.append(ht)
        return historique


class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    def __init__(self, input_dim, latent_dim):
        super(GRU, self).__init__()

        self.W_update = nn.Sequential(torch.nn.Linear(input_dim+latent_dim, latent_dim), torch.nn.Sigmoid())
        self.W_reset = nn.Sequential(torch.nn.Linear(input_dim+latent_dim, latent_dim), torch.nn.Sigmoid())

        self.W = torch.nn.Sequential(nn.Linear(input_dim+latent_dim, latent_dim), torch.nn.Tanh())

    def one_step(self, x, h):

        zt = self.W_update(torch.cat((x, h), 0))
        rt = self.W_reset(torch.cat((x, h), 0))
        ht = (1 - zt) * h + zt * self.tanh(self.W(torch.cat((x, rt * h), 0)))

        return ht

    def forward(self, x, h=None):

        historique = []
        # pour chaque instant de la sequence:
        if h is None:
            ht = torch.zeros(x.size()[1], self.latent)
        for xt in x:
			# ht: (batch x latent)
            ht = self.one_step(xt, ht)
            historique.append(ht)
        return historique



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
