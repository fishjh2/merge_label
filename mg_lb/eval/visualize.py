import numpy as np
import itertools
import copy

from mg_lb.problems.probs import labels_lookup


def extract_clusters(preds, cluster_ixs, sents, average=False, links=False, equalize=False, get_label=False):
    cluster_preds = []
    for ix, c in enumerate(cluster_ixs):
        ps = preds[ix]

        if links:
            ps = [np.squeeze(i, axis=0) for i in np.split(ps, indices_or_sections=ps.shape[0], axis=0)]

        d = single_cluster(c, ps, sents, average, equalize, get_label)

        cluster_preds.append(d)

    return cluster_preds


def single_cluster(c, ps, sents, average, equalize, get_label):
    d = {}
    for k, v in c.items():
        if not sents:
            inps = ps[k]
        else:
            inps = ps
        newps = []
        for inner in v:
            grouped = [inps[i] for i in inner]
            if average:
                newps.append(np.average(grouped, axis=0))
            elif equalize:
                mainval = max(set(grouped), key=grouped.count)
                newps.append([mainval for _ in range(len(grouped))])
            elif get_label:
                # Remove 'B-' and 'I-'
                grouped = [g.split('-')[1] if g != 'O' else g for g in grouped]
                mainval = max(set(grouped), key=grouped.count)
                newps.append(mainval)
            else:
                newps.append(grouped)
        d[k] = newps
    return d


def split_to_sentences(sents, cl_ixs, cl_words, cl_ls, slens):
    """
    Split the sentences, cluster_ixs and cluster_words for an article down into sentences
    """
    # Get ixs of full stops
    fs_ixs = [ix for ix, w in enumerate(slens) if w == 1.0]

    ss = []
    ci = []
    cw = []
    cl = []

    def dsort(d, f, st, wrds=None):
        n = {}
        for k, v in d.items():
            if wrds is not None:
                n[k] = [wrds[k][ix] for ix, j in enumerate(v) if j[-1] <= f and j[-1] >= st]
            else:
                n[k] = [j for j in v if j[-1] <= f and j[-1] >= st]
        return n

    st = 0
    for f in fs_ixs:
        ss.append(sents[st:f + 1])
        ci.append(dsort(cl_ixs, f, st))
        cw.append(dsort(cl_ixs, f, st, wrds=cl_words))
        cl.append(dsort(cl_ixs, f, st, wrds=cl_ls))
        st = f + 1

    return ss, ci, cw, cl


def sort_simple(l, lens):
    """
    Helper function for sorting returns from network into separate sentences
    """
    l = [np.split(t, indices_or_sections=len(t), axis=0) for t in l]
    l = list(itertools.chain.from_iterable(l))
    l = [np.squeeze(i, axis=0) for i in l]

    if lens is not None:
        l = [i[:(lens[ix])] for ix, i in enumerate(l)]
    #else:
    #    l = [i[:(lens[ix]-1)] for ix, i in enumerate(l)]
    return l


def sort_clusters(weights, cutoff=0.7):
    """
    Args
    weights: list of sentences, all arrays of shape [seq_len, kernel_size+1, num_out_layers]
    """
    weights = [w[:-1, 1, :] for w in weights]

    num_out = weights[0].shape[1]

    all_clusters = []
    grouped_ixs = []

    for w in weights:
        clusters = {}
        cl_ixs = {}
        for o in range(num_out):
            cl = [0]
            all_ixs = []
            new = [0]
            cluster_val = 0
            layer = w[:, o]
            for ix, l in enumerate(layer):
                if l < cutoff:
                    all_ixs.append(new)
                    new = [ix + 1]
                    cluster_val += 1
                else:
                    new.append(ix + 1)

                cl.append(cluster_val)

            all_ixs.append(new)

            cl_ixs[o] = all_ixs
            clusters[o] = cl

        all_clusters.append(clusters)
        grouped_ixs.append(cl_ixs)

    return all_clusters, grouped_ixs


def sort_pos_labels(sentences, preds, pos_labels, dtype):

    lookup = labels_lookup[dtype]
    rev_lookup = {}
    for k, v in lookup.items():
        rev_lookup[v] = k

    ps = []
    ls = []

    for ix, s in enumerate(sentences):
        # [sentence_length, num_cluster_levels]
        pos_ind = pos_labels[ix]
        sep_labels = np.split(pos_ind, axis=1, indices_or_sections=pos_ind.shape[1])
        sep_labels = [np.squeeze(sep) for sep in sep_labels]
        d = {}
        for s_ix, set_l in enumerate(sep_labels):
            d[s_ix] = [rev_lookup[i] for i in set_l]
        ls.append(d)

        if preds is not None:
            num_ps = preds[0].shape[1]
            pd = {}
            for p_ix in range(num_ps):
                p = preds[ix][:, p_ix]
                pd[p_ix] = [rev_lookup[i] for i in p]
            ps.append(pd)

    return ps, ls


def data_from_iter(i, vocab, att=None, ans=True, pos=False):
    """
    Extract data from iterator into single sentences
    """
    ns = 'nonstack_labels' in i.inputs.keys()

    sents = [j.numpy() for j in i.inputs['sentences']]
    lens = [j.numpy() for j in i.inputs['len']]
    sentences = []
    answers = []
    attention = []
    lengths = []
    posret = []
    posorig = []

    for b_ix, s in enumerate(sents):
        for s_ix, sen in enumerate(s):
            ls_t = lens[b_ix][s_ix]
            if 'gpt2' in str(vocab.__class__):
                words = [vocab.token_decode(i) for i in sen[:ls_t]]
            else:
                words = [vocab.index_to_word[i] for i in sen[:ls_t]]
            sentences.append(words)
            if ans:
                answers.append(i.inputs['answers'][b_ix][s_ix])
            if att is not None:
                attention.append(att[b_ix][s_ix][:lens[b_ix][s_ix]])
            if pos:
                if 'original_labels' in i.inputs.keys():
                    posret.append(i.inputs['original_labels'][b_ix][s_ix, :ls_t, :].numpy())
                else:
                    posret.append(i.inputs[i.lab_key][b_ix][s_ix, :ls_t, :].numpy())
            if ns:
                posorig.append(i.inputs['nonstack_labels'][b_ix][s_ix, :ls_t, :].numpy())
            lengths.append(ls_t)

    rts = {
        'sentences': sentences,
        'answers': answers,
        'attention': attention,
        'lens': lengths,
        'pos': posret,
        'nonstack': posorig
    }

    return rts


def sort_results(res, iterator, vocab, problem, cutoff=0.8, cluster_labels=False):

    rts = {}

    if 'ent_preds' in res.keys() or cluster_labels:
        pos = True
    else:
        pos = False

    st = data_from_iter(iterator, vocab, ans=False, pos=pos)
    #sentences, _, _, lens, pos

    # Sort weights - list of articles all shape [seq_len, 2, num_out_layers]
    weights = sort_simple(res['final_weights'], st['lens'])
    weights = [np.swapaxes(w, 2, 1) for w in weights]
    rts['weights'] = weights

    # List of len num_articles of shape [seq_len, num_out_layers]
    if 'ch_mask' in res.keys():
        rts['ch_mask'] = sort_simple(res['ch_mask'], st['lens'])
        rts['pred_mask'] = sort_simple(res['pred_mask'], st['lens'])
        rts['cl_mask'] = sort_simple(res['cl_mask'], st['lens'])

    if 'article_theme' in res.keys():
        rts['article'] = sort_simple(res['article_theme'], st['lens'])

    # List of len num_articles of shape [num_layers, seq_len, embed_size]
    embeds = sort_simple(res['embeds'], st['lens'])
    embeds = [np.swapaxes(e, 0, 1) for e in embeds]

    # Links. list of batches of shape [seq_len, network_size, 2]
    if 'links' in res.keys():
        links = sort_simple(res['links'], st['lens'])

    # Sort clusters
    rts['clusters'], cluster_ixs = sort_clusters(weights, cutoff=cutoff)
    rts['cluster_words'] = extract_clusters(st['sentences'], cluster_ixs, sents=True)
    rts['cluster_ixs'] = cluster_ixs

    # Sort all the preds
    if 'all_levels' in res.keys() and len(res['all_levels']) > 0:
        all_preds = sort_simple(res['all_levels'], st['lens'])

        if len(all_preds[0].shape) == 4:
            all_ps = [np.argmax(allp[:, :, -1, :], axis=2) for allp in all_preds]
        else:
            all_ps = [np.argmax(allp, axis=2) for allp in all_preds]


        # Sort labels and preds
        rts['preds'], rts['labels'] = sort_pos_labels(st['sentences'], all_ps, st['pos'], dtype=problem)
        rts['all_preds'] = all_preds

        # Cluster the preds
        rts['cluster_preds'] = extract_clusters(rts['preds'], cluster_ixs, sents=False, equalize=True)

    if cluster_labels:
        _, ls = sort_pos_labels(st['sentences'], preds=None, pos_labels=st['pos'], dtype=problem)

        # Labels for each cluster
        rts['cluster_labels'] = extract_clusters(ls, cluster_ixs, sents=False, get_label=True)

    # Original labels without stacking up layers
    if len(st['nonstack']) > 0:
        _, rts['nonstack'] = sort_pos_labels(st['sentences'], all_ps, st['nonstack'], dtype=problem)

    # Cluster the embeds
    rts['cluster_embeds'] = extract_clusters(embeds, cluster_ixs, sents=False, average=True)

    # Cluster the links
    if 'links' in res.keys() and len(links) > 0:
        rts['cluster_links'] = extract_clusters(links, cluster_ixs, sents=True, average=False, links=True)

    rts['sentences'] = st['sentences']
    rts['d_labels'] = st['pos']

    # LM preds
    if 'lm_predictions' in res.keys():
        #if 'gpt2' in str(vocab.__class__):
        lm_preds = []
        for l in res['lm_predictions']:
            lm_preds.append([vocab.token_decode(i) for i in list(l)])
        rts['lm_preds'] = lm_preds

    return rts


def add_bs(predictions, rejoin=True):
    """
    Add 'B-'s back at start of ents
    """
    predictions = copy.deepcopy(predictions)
    # Add bs back to start of preds
    for art in predictions:
        for k, v in art.items():
            for inner in v:
                if inner[0] != 'O':
                    inner[0] = 'B-' + inner[0][2:]

    if rejoin:
        for art in predictions:
            for k, v in art.items():
                art[k] = list(itertools.chain.from_iterable(v))
    return predictions


