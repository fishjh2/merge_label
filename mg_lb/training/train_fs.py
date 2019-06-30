import numpy as np

from mg_lb.problems.probs import labels_lookup
from mg_lb.eval.losses import compute_f1


limit_lookup = {
}


def acc_process(tensors, pred, prob, train=True):

    pred['acc_target'] = tensors['labels']

    return pred, tensors


def ents_process(tensors, pred, prob, train=True):

    # [batch, seq_len, num_levels, use_layers, num_ents] or [batch, seq_len, num_levels, num_ents]
    predictions = pred['ent_preds']

    # [batch, seq_len, num_levels]
    targets = tensors['ent_labels']

    # [batch, seq_len, num_levels]
    nonpad = tensors['ents_mask']

    # [num_nonmasked, num_ents]
    pred['ent_preds'] = predictions[nonpad]

    # [num_nonmasked, 1]
    tensors['ent_labels'] = targets[nonpad].unsqueeze(1)

    if not train:

        if 'ACE' in prob:
            # All levels for F1 approximation.
            tr_pad = nonpad.transpose(1, 2)

            pred['pred_first'] = predictions.transpose(1, 2)[tr_pad]
            pred['acc_target'] = targets.transpose(1, 2)[tr_pad].unsqueeze(1)

            # For F1 need to keep order so use transpose mask
            pred['f1_target'] = tensors['original_labels'].transpose(1, 2)[tr_pad]
            pred['acc_weights'] = pred['final_weights'].transpose(1, 2)[tr_pad][:, 0]

        else:
            # First level for F1 approximation. [batch, seq_len, num_levels]
            first_np = nonpad.clone()
            first_np[:, :, 1:] *= 0

            # [num_nonmasked]
            pred['pred_first'] = predictions[first_np]
            pred['acc_target'] = targets[first_np].unsqueeze(1)
            pred['f1_target'] = tensors['original_labels'][first_np]

            pred['acc_weights'] = pred['final_weights'][first_np][:, 0]

        # All levels prediction for printing
        pred['all_levels'] = predictions.data

    return pred, tensors


def get_accuracy(targets, preds, weights=None, f1_target=None, last=True, prob=None, args=None):

    # Sometimes haven't used whole dataset to save time during training
    if not last and prob in limit_lookup.keys():
        targets = targets[:preds.shape[0], :]

    if type(targets[0]) != np.str_:
        targets = targets.astype(int)

    if len(preds.shape) == 3:
        multi = True
        targets = np.expand_dims(targets, 2)
        preds = np.squeeze(preds, axis=2)
    else:
        multi = False

    if targets.shape[-1] == 1:
        targets = targets.squeeze(-1)

    if multi:
        assert targets.shape[0] == preds.shape[0]
    else:
        assert targets.shape == preds.shape, 'Targets {} and preds {} should have same size'.format(targets.shape, preds.shape)

    correct = preds == targets

    if multi:
        # Max over num_out_layers dim
        correct = np.max(correct, axis=1)

    accuracy = np.sum(correct) / len(targets)

    lookup = labels_lookup[prob]
    _, _, f1 = compute_f1(preds, f1_target, lookup, True, weights=weights,
                          cutoff=args.cutoff)

    return {'accuracy': accuracy, 'correct': correct, 'f1': f1}


def warmup(step, lr, warmup_steps):
    new_lr = (max(step, 100) / warmup_steps) * lr
    return new_lr


