import torch


# environment settings
FIELD_SIZE = 3

# torch settings
DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
