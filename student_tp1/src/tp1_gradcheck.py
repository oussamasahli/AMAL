import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
#torch.autograd.gradcheck(mse, inputs=(yhat.double(), y.double()))


#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)


X = torch.randn(10,2, requires_grad=True, dtype=torch.float64)
W = torch.randn(2,1, requires_grad=True, dtype=torch.float64)
b = torch.randn(1,1, requires_grad=True, dtype=torch.float64)
torch.autograd.gradcheck(linear, (X, W, b))