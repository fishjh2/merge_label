from mg_lb.problems.probs import problems


def probs_to_losses(probs, loss_fns, loss_ws):
    """
    Create a dict of problems matching them with all the loss functions they'll be trained on
    """
    prob_losses = {}
    for p in probs:
        prob_losses[p] = {'loss': [], 'lm': False, 'coref': False, 'ent': False,
                          'load_ent': False, 'load_weights': False}

    # Decoders needed for different loss functions
    for ix, p in enumerate(probs):
        l = loss_fns[ix]
        i = prob_losses[p]
        i['loss'].append(l)
        if l == 'coref_loss':
            i['coref'] = True
        if l in ['coref_loss', 'named_ent_loss', 'named_ent_lm_loss']:
            i['ent'] = True
            i['load_ent'] = True
            i['load_weights'] = True
        if l in ['cluster_loss']:
            i['load_ent'] = True
        if l in ['wiki_lm_loss', 'ent_lm_loss']:
            i['load_weights'] = True
        if l in ['lm_loss', 'wiki_lm_loss', 'ent_lm_loss', 'named_ent_lm_loss']:
            i['lm'] = True
        if l in ['wiki_loss']:
            i['load_ent'] = True
        if l in ['cross_entropy_loss']:
            i['ent'] = True

        prob_losses[p]['fs_per_iter'] = problems[p]['fs_per_iter']

        prob_losses[p]['loss_weight'] = loss_ws[ix]

    return prob_losses