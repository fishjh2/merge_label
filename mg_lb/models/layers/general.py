import torch


def get_kernel_sizes(expand_ratio, num_layers, max_kernel_size):
    distances_away = [expand_ratio * (l + 1) for l in range(num_layers)]
    kernel_sizes = [r * 2 for r in distances_away]
    kernel_sizes = [min(k, max_kernel_size) for k in kernel_sizes]
    return kernel_sizes


def get_cl_levels(layer_list):
    return max(layer_list) - 1


def alternate_ix(ixs, tensor):
    return torch.index_select(tensor, 0, ixs.view(-1)).view(*ixs.size(), *tensor.size()[1:])


def gen_zeros(sz, gpu):

    if gpu:
        zrs = torch.cuda.FloatTensor(*sz).fill_(0.0)
    else:
        zrs = torch.ones(*sz)

    return zrs