import torch


# environment settings
FIELD_SIZE = 3

# torch settings
DTYPE = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NW_LEARNING_RATE = 0.00025
NW_ALPHA = 0.95
NW_EPS = 0.01

# replay buffer settings
NUM_REPLAY_BUFFER = 10000

# e-greedy settings
EPS_TIMESTEPS = 800
EPS_END = 0.00001
EPS_START = 1.0

# simulation settings
BATCH_SIZE = 128
GAMMA = 0.999
