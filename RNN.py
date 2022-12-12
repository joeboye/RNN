import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 4
num_hiddens = 500

rnn_layer = nn.RNN(input_size=input_size, hidden_size=num_hiddens)

num_steps = 24
batch_size = 1
state = None
X = torch.rand(num_steps, batch_size, input_size)
Y, state_new = rnn_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)