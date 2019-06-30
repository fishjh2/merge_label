import torch
from torch.autograd import Variable

from mg_lb.models.layers.initialization import pytorch_init_linear
from mg_lb.models.layers.activations import activation_dict


class linear_layer(torch.nn.Module):
    """
    Feedforward linear layer
    """

    def __init__(self, in_size, out_size, dropout, activation_fn,
                 w_init=('uniform', 0.1), b_init=('constant', 0.1), norm=False):

        super(linear_layer, self).__init__()

        if dropout == 0.0:
            self.dropout = None
        else:
            self.dropout = dropout

        self.norm = norm
        self.activation_fn = activation_fn

        # Build layer
        lin = torch.nn.Linear(in_size, out_size)
        pytorch_init_linear(lin, w_init, b_init, activation_fn=activation_fn)

        layer = torch.nn.ModuleList()

        layer.append(lin)

        # Add activation function
        if activation_fn not in [None, 'none']:
            layer.append(activation_dict[activation_fn]())

        # Add dropout
        if dropout is not None:
            layer.append(torch.nn.Dropout(dropout))

        self.layer = torch.nn.Sequential(*layer)

    def forward(self, x):

        f = self.layer(x)

        return f


class NN(torch.nn.Module):
    """
    Feedforward NN, with optional prediction of quantiles of output as well as the mean
    """

    def __init__(self, features_dim, targets_dim, num_hidden_nodes, num_layers,
                 activation_fn, w_init=('uniform', 0.1), b_init=('constant', 0.1),
                 f_w_init=('uniform', 0.1), f_b_init=('constant', 0),
                 multi_quantile=None, dropout=None, norm=False):

        super(NN, self).__init__()

        # Checks if trying to initialize to reproduce input tensor
        if w_init[0] == 'repr':
            assert f_w_init[0] == 'repr'
            if num_hidden_nodes[0] != 0:
                for h in num_hidden_nodes:
                    assert h >= w_init[1], 'All hidden nodes need to be larger than replicate size'
            assert features_dim >= w_init[1]
            assert targets_dim >= w_init[1]

        if float(dropout) == 0.0:
            self.dropout = None

        self.multi_quantile = multi_quantile

        if num_hidden_nodes[0] != 0:
            if type(num_hidden_nodes) not in [list, tuple]:
                num_hidden_nodes = [num_hidden_nodes] * num_layers

            self.num_layers = len(num_hidden_nodes)

        else:
            self.num_layers = 0

        # Build the hidden layers
        if self.num_layers > 0:
            layers = torch.nn.ModuleList()
            for l in range(num_layers):
                if l == 0:
                    input_size = features_dim
                else:
                    input_size = num_hidden_nodes[l - 1]

                # Build layer
                layer = linear_layer(input_size, num_hidden_nodes[l], dropout, activation_fn,
                                     w_init, b_init, norm)

                layers.append(layer)

            self.network = torch.nn.Sequential(*layers)

            final_input_size = num_hidden_nodes[-1]

        else:
            final_input_size = features_dim

        # Output layer
        self.output = torch.nn.Linear(final_input_size, targets_dim)
        pytorch_init_linear(self.output, f_w_init, f_b_init)

        # Quantile output layers
        if self.multi_quantile:
            for q in multi_quantile:
                q_output = torch.nn.Linear(features_dim + final_input_size, targets_dim)
                pytorch_init_linear(q_output, f_w_init, f_b_init)
                self.add_module('quantile_' + str(q), q_output)

    def forward(self, x):
        output = {}
        # Hidden layers
        if self.num_layers > 0:
            final_layer_out = self.network(x)

        else:
            final_layer_out = x

        # Output layer
        mean_out = self.output(final_layer_out)
        output['main'] = mean_out

        if self.multi_quantile:
            mean_detached = Variable(mean_out.data)

            q_in = torch.cat([x, Variable(final_layer_out.data)], dim=1)
            for q in self.multi_quantile:
                q_out = getattr(self, 'quantile_' + str(q))(q_in)
                if q > 0.5:
                    output[q] = mean_detached + q_out
                elif q < 0.5:
                    output[q] = mean_detached - q_out
                else:
                    output[q] = q_out

        return output