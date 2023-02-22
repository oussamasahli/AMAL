# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/baskiotis/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        #  TODO:  Renvoyer la valeur de la fonction
        q=y.shape[0]
        return (1/q * torch.linalg.norm(torch.sub(yhat,y))).double()

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à
        #yhat et à y)
        q=y.shape[0]
        d_yhat = 1/q * 2 * torch.sub(yhat,y)
        d_y = 1/q * -2 * torch.sub(yhat,y)
        return d_yhat, d_y

#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE

class Linear(Function):
    """Début d'implementation de la fonction Linear"""
    @staticmethod
    def forward(ctx, X, W, b):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(X, W, b)

        #  TODO:  Renvoyer la valeur de la fonction
   
        return torch.add(torch.mm(X,W), b)

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        X, W, b = ctx.saved_tensors
      
        grad_X = torch.mm(grad_output,W.t())
        grad_W = torch.mm(grad_output.t(),X)
        #grad_b = torch.ones((1,W.shape[1]),dtype=torch.float64, requires_grad=True )
        grad_b = grad_output.sum(0)
        return grad_W.t(), grad_X.t(), grad_b

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

