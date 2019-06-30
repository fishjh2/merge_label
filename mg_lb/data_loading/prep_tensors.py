import torch


def prep_network_tensors(nb, args):

    # Put on gpu
    if args.gpu:
        for k, v in nb.items():
            if type(v) == torch.Tensor:
                nb[k] = nb[k].cuda()

            if k == 'kmasks':
                for ink, inv in v.items():
                    nb['kmasks'][ink] = inv.cuda()

    return nb
