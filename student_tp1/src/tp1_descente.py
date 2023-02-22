#-*- coding: utf-8 -*-
import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
#b = torch.randn(3)
b = torch.randn(1,3)

#epsilon = 0.05
epsilon = 0.001

writer = SummaryWriter()
linear=Linear()
mse=MSE()
Couts=[]
its=[i for i in range (0,100)]

for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    ctx1=Context()
    y_hat= linear.forward(ctx1, x, w, b)
    # `loss` doit correspondre au coût MSE calculé à cette itération
    ctx2=Context()
    loss=mse.forward(ctx2, y_hat ,y)
    Couts.append(loss)
    
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    d_yhat,d_y = mse.backward(ctx2,None)
    grad_w, grad_x, grad_b = linear.backward(ctx1, d_yhat)
    

    
    ##  TODO:  Mise à jour des paramètres du modèle
    w = torch.sub(w,epsilon*grad_w)
    b = torch.sub(b,epsilon*grad_b)

# Utilise ces lignes de commandes dans la console pour l'affichage
#%matplotlib inline 
#plt.plot(its,Couts)
#plt.ylabel('Loss MSE')