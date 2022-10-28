import torch.nn as nn
import torch.nn.functional as F


def nll_loss(output, target):
    loss_func = F.nll_loss()
    return loss_func(output, target)


def L1_loss(output, target):
    loss_func = nn.L1Loss()
    return loss_func(output, target)


def mse_loss(output, target):
    loss_func = nn.MSELoss()
    return loss_func(output, target)


def BCEWithLogitsLoss(output, target):
    loss_func = nn.BCEWithLogitsLoss()
    return loss_func(output, target)


loss_config = {
    "nll": nll_loss,
    "l1": L1_loss,
    "mse": mse_loss,
    "bce": BCEWithLogitsLoss,
}
