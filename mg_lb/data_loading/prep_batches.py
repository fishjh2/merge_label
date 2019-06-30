import numpy as np
from tqdm import tqdm
import torch

from mg_lb.data_loading.fs import tokenize, sent_to_idx, predict_word
from mg_lb.problems.probs import labels_lookup


def pad_batch(input_list, pad=0, array=False):
    """
    Given a batch (list of lists) where each list is the integer encoding for one sentence,
    pad each list with zeros until they're the same length
    Args
    input_list: a list of lists, where each sublist is the integer encoding of a sentence
    pad: the integer value to use for padding
    array: if inputs are in array form
    Returns
    padded: a list of np arrays, with one array for each sentence in the batch, each of shape
            [1, max_seq_len]
    """
    # Get the longest input in the batch
    max_len = max([len(i) for i in input_list])

    padded = []

    for i in input_list:
        diff = max_len - len(i)

        if array:
            n = np.zeros(shape=[diff, i.shape[1]]) + pad
        else:
            # Array of zeros for padding
            n = np.zeros(shape=[diff]) + pad

        arr = np.concatenate([np.array(i), n])

        # Reshape to size [1, len(seq)]
        arr = np.expand_dims(arr, axis=0)

        padded.append(arr)

    return padded


def prep_pos_batches(num_batches, batch_size, labels, problem, num_cluster_levels, all_splits=False, pad_val=None,
                     convert_bs=False):

    lookup = labels_lookup[problem]
    inv_lookup = {v: k for k, v in lookup.items()}
    inv_lookup['start'] = ''

    if pad_val is None:
        pad_val = -1.0

    assert pad_val < 0, 'Need a pad_val which is different to the "O" value else will mess up boolean mask below'

    # Group lookup vals of same entity type
    groups = {}
    for k, v in lookup.items():
        if 'I-' in k:
            groups[v] = [v, lookup['B-' + k.split('I-')[1]]]
        elif 'B-' in k:
            groups[v] = [v, lookup['I-' + k.split('B-')[1]]]
        else:
            groups[v] = []

    previous = []
    counter = 0

    for sent_dict in labels:

        pd = {}

        # Loop through the different layers of labels
        for k, v in sent_dict.items():

            current_previous = []

            for ix, ent_val in enumerate(v):

                tk = inv_lookup[ent_val]

                if 'I-' in tk:
                    current_previous.append(1.0)
                else:
                    current_previous.append(0.0)

            pd[k] = current_previous

            counter += 1

        previous.append(pd)

    if convert_bs:

        # Lookup if getting rid of 'B-'s and having just 'I's'
        bd = {}
        for k, v in lookup.items():
            if 'B-' in k:
                bd[v] = lookup['I-' + k.split('B-')[1]]
            else:
                bd[v] = v

        new_labels = []

        for l in labels:
            new = {}
            for k, inner in l.items():
                new[k] = [bd[i] for i in inner]
            new_labels.append(new)

    else:
        new_labels = labels

    label_batches = simple_batch(num_batches, batch_size, new_labels, pad_val)
    orig_label_batches = simple_batch(num_batches, batch_size, labels, pad_val)

    # Batches of values denoting the target weighting required for each word
    previous_batches = simple_batch(num_batches, batch_size, previous, pad_val=0.0)
    previous_batches = expand_prev_batches(previous_batches)

    # Boolean mask to extract only nps, ents etc. Used for clustering weights
    bool_batches = []
    fill_value = lookup['O']

    for l in label_batches:
        # [batch, seq_len, num_in_layers]
        not_fill = l != fill_value
        not_pad = l != pad_val

        if all_splits and problem not in ['wikipedia']:
            bb = not_pad
        else:
            bb = not_fill * not_pad

        # Shape [batch_size, seq_len, num_label_layers]
        bool_batches.append(bb.astype(np.float32))

    # Mask for the labels
    mask_batches = []

    for l in label_batches:
        bb = (l != pad_val)
        mask_batches.append(bb.astype(np.float32))

    # Only use as many labels as have out layers
    label_batches = [l[:, :, :num_cluster_levels] for l in label_batches]
    orig_label_batches = [l[:, :, :num_cluster_levels] for l in orig_label_batches]
    previous_batches = [l[:, :, :num_cluster_levels, :] for l in previous_batches]
    mask_batches = [l[:, :, :num_cluster_levels] for l in mask_batches]
    bool_batches = [l[:, :, :num_cluster_levels] for l in bool_batches]

    return label_batches, previous_batches, mask_batches, bool_batches, orig_label_batches


def expand_prev_batches(prev):
    """
    Args
    prev: list of batches of shape [batch_size, seq_len, num_label_layers]
    """
    new = []
    for p in prev:
        z = np.zeros([p.shape[0], 1, p.shape[2]])
        # [batch_size, seq_len, num_label_layers, 1]
        pin = np.expand_dims(np.concatenate([p[:, 1:, :], z], axis=1), axis=3)

        # [batch_size, seq_len, num_label_layers, 2]
        new.append(np.concatenate([np.expand_dims(p, axis=3), pin], axis=3).astype(np.float32))

    return new


def simple_batch(num_batches, batch_size, data, pad_val):
    # Join dicts into arrays
    newdata = []
    for d in data:
        vs = [np.expand_dims(np.array(i), 1) for i in d.values()]
        newdata.append(np.concatenate(vs, axis=1))

    batches = []

    for i in range(num_batches):
        l = newdata[i * batch_size:i * batch_size + batch_size]

        # Make sure each sentence in the batch the same length
        l_b = pad_batch(l, pad=pad_val, array=True)

        # Join to shape [batch_size, seq_len, num_label_layers]
        l_b = np.concatenate(l_b, axis=0)

        batches.append(l_b.astype(np.float32))

    return batches


def pad_flair(batch):
    # Find max sentence length
    max_len = max([b.size(0) for b in batch])

    rt_batch = []

    for b in batch:
        diff = max_len - b.size(0)
        if diff > 0:
            zeros = torch.zeros([diff, b.size(1)])
            b = torch.cat([b, zeros], dim=0)

        # Expand to [1, max_seq_len, embed_size]
        rt_batch.append(b.unsqueeze(0))

    # Concatenate to [batch_size, seq_len, embed_size]
    joined = torch.cat(rt_batch, dim=0)

    return joined


def prep_flair_batch(num_batches, batch_size, flair):
    batches = []

    for i in range(num_batches):
        # Get len(batch_size) list of lists, each of len(sent_len)
        batch = flair[i * batch_size:i * batch_size + batch_size]

        # Concatenate each sentences. List of len(batch_size) of arrays of size [sent_len, embed_size]
        batch = [torch.cat(b, dim=0) for b in batch]

        # Pad and join
        batches.append(pad_flair(batch))

    return batches


def word_to_caps(words):
    """
    Given a list of words, produce an index denoting whether all lowercase, all uppercase, first letter uppercase, or mixed
    """
    caps = []

    for w in words:
        if w.isupper():
            caps.append(0)
        elif w.islower():
            caps.append(1)
        elif w[1:].islower():
            caps.append(2)
        else:
            caps.append(3)
    return np.array(caps)


def prep_sent_batch(num_batches, batch_size, d, vocab, merge_ents, sos=False, eos=False, progress=True,
                    extend_vocab=True, mask_answerhere=False, tokenized=False, cap_features=False):
    """
    Split sentences into batches, and convert from strings to integer ids of strings
    """
    batches = []
    len_batches = []
    ans_mask_batches = []
    cap_batches = []

    for i in tqdm(range(num_batches), total=num_batches, disable=not progress):
        # Get len(batch_size) lists of sentences and labels
        sp = d[i * batch_size:i * batch_size + batch_size]
        batch = []
        lens = []
        ans_mask = []
        caps = []

        for b in range(len(sp)):

            if sp[b] is None:
                batch.append(None)
                continue

            # Record length of each sentence
            if not tokenized:
                tokens, doc = tokenize(sp[b], merge_ents=merge_ents, return_sp=True)
            else:
                tokens = sp[b]

            # Make sure have at least one answerhere and hasn't been included in a merged token
            if mask_answerhere:
                for ix, t in enumerate(tokens):
                    if predict_word in t:
                        tokens[ix] = predict_word

            # Add the words to the vocab
            if extend_vocab:
                for t in tokens:
                    if t is None:
                        print(tokens)
                    vocab.add_word(t)

            # Add <SOS> token at start
            if sos:
                tokens = ['<SOS>'] + tokens

            if eos:
                tokens.append('<EOS>')

            lens.append(np.array(len(tokens)))

            if mask_answerhere:
                m = [int(w != predict_word) for w in tokens]
                ans_mask.append(m)

                if np.sum(m) == len(tokens):
                    print('got here', tokens)

            # Convert from strings to indices
            batch.append(sent_to_idx(tokens, vocab))

            # Add cap features
            caps.append(word_to_caps(tokens))

        # Pad the batches so all sequences in batch have same length
        batch = pad_batch(batch)

        # Concatenate sentences into shape [batch_size, max_seq_len_in_batch] arrays
        batch = np.concatenate(batch, axis=0).astype(int)
        batches.append(batch)

        if mask_answerhere:
            ans_batch = pad_batch(ans_mask, pad='ones')
            ans_batch = np.concatenate(ans_batch, axis=0).astype(np.float32)
            ans_mask_batches.append(ans_batch)

        if cap_features:
            cap_batches.append(np.concatenate(pad_batch(caps), axis=0).astype(int))

        # Add lens
        lens = [np.expand_dims(i, 0) for i in lens]
        len_batches.append(np.concatenate(lens))

    return batches, len_batches, cap_batches, vocab, ans_mask_batches


