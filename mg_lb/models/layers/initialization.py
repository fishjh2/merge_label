import torch.nn.init as weight_init
import torch


def pytorch_init_linear(layer, w_init, b_init, activation_fn=None):
    """
    Initialize linear pytorch layer
    Args
    layer: pytorch linear module
    w_init: initialization params for weights
    b_init: initialization params for bias
    activation_fn: the activation fn for the layer
    """
    pytorch_initialize(layer.weight, w_init, activation_fn)

    if w_init[0] == 'repr':
        pytorch_initialize(layer.bias, ('constant', 0.0))
    else:
        pytorch_initialize(layer.bias, b_init)


def pytorch_initialize(tensor, i, activation_fn=None):
    """
    Initialize pytorch parameters using same terminology as for tf models
    Args
    tensor: a pytorch tensor object
    i: parameters for initialization - of form ('init_type', init_param)
       e.g. ('random_normal', 0.5)
    activation_fn: the activation function for the layer
    """
    if i[0] == 'random_normal':
        weight_init.normal_(tensor=tensor, std=i[1])
    elif i[0] == 'constant':
        weight_init.constant_(tensor=tensor, val=i[1])
    elif i[0] == 'xavier':
        weight_init.xavier_normal_(tensor=tensor, gain=i[1])
    elif i[0] == 'uniform':
        weight_init.uniform_(tensor=tensor, a=-i[1], b=i[1])
    elif i[0] == 'repr':
        weight_init.uniform_(tensor=tensor, a=-0.1, b=0.1)
        weight_init.uniform_(tensor=tensor[:i[1], :], a=-i[2], b=i[2])

        if activation_fn == 'selu':
            tensor.data[:i[1], :i[1]].diagonal().copy_(torch.tensor(0.9517))
        elif activation_fn in ['swish', None]:
            tensor.data[:i[1], :i[1]].diagonal().copy_(torch.tensor(1.0))
        else:
            raise Exception('Can\'t reproduce layer without linear activation')
    else:
        raise Exception('Unrecognised initializer type'.format(i[0]))