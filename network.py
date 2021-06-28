from torch import autograd
from setting import *
import torch
import torch.nn as nn

from secrets import *


class DqnNetwork(nn.Module):
    def __init__(self):
        super(DqnNetwork, self).__init__()
        self.model = nn.Sequential(nn.Linear(FIELD_SIZE * FIELD_SIZE * 2, FIELD_SIZE * FIELD_SIZE * 13),
                                                nn.ReLU(),
                                                nn.Linear(FIELD_SIZE * FIELD_SIZE * 13, FIELD_SIZE * FIELD_SIZE * 13),
                                                nn.ReLU(),
                                                nn.Linear(FIELD_SIZE * FIELD_SIZE * 13, FIELD_SIZE * FIELD_SIZE * 13),
                                                nn.ReLU(),
                                                nn.Linear(FIELD_SIZE * FIELD_SIZE * 13, FIELD_SIZE * FIELD_SIZE))

    def forward(self, x):
        return self.model(x)


class NetworkUtil(object):
    def initialize(network: nn.Module) -> nn.Module:
        return network.type(DTYPE).to(device=DEVICE)

    def copy_param(frm: nn.Module, to: nn.Module) -> nn.Module:
        to.type(DTYPE)
        to.load_state_dict(frm.state_dict())
        to.eval()
        to.to(device=DEVICE)
        return to

    def to_binary(lst: torch.Tensor) -> torch.tensor:
        mask = 2 ** torch.arange(2).to(lst.device, lst.dtype)
        binary = lst.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
        return Variable(torch.reshape(binary, (-1, FIELD_SIZE * FIELD_SIZE * 2)).float())

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs) -> None:
        if torch.cuda.is_available():
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)
