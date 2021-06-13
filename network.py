from torch import autograd
from setting import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from secrets import *


class DqnNetwork(nn.Module):
    def __init__(self):
        super(DqnNetwork, self).__init__()
        self.model = nn.Sequential(nn.Linear(FIELD_SIZE * FIELD_SIZE * 2, FIELD_SIZE * FIELD_SIZE + 3),
                                                nn.ReLU(),
                                                nn.Linear(FIELD_SIZE * FIELD_SIZE + 3, FIELD_SIZE * FIELD_SIZE))

    def forward(self, x):
        return self.model(x)


class NetworkUtil(object):
    def initialize(network: nn.Module):
        return network.type(DTYPE).to(device=DEVICE)

    def copy_param(from: nn.Module, to: nn.Module):
        to.type(DTYPE)
        to.load_state_dict(from.state_dict())
        to.eval()
        to.to(device=DEVICE)
        return to

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs) -> None:
        if torch.cuda.is_available():
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)
