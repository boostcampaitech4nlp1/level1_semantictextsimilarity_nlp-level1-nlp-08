import torch.nn as nn
import torch.nn.functional as F


def nll_loss(output, target):
    loss_func = F.nll_loss()
    return loss_func(output, target)

def L1_loss(output, target):
    loss_func = nn.L1Loss()
    return loss_func(output, target)

def mse_loss(output, target) :
    loss_func = nn.MSELoss()
    return loss_func(output, target)