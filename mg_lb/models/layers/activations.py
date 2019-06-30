import torch.nn as nn


class swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * x.sigmoid()


activation_dict = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'selu': nn.SELU,
    'swish': swish
}
