import copy
from collections import defaultdict


def calc_f1(prec, rec):
    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    return f1


def compute_f1(predictions, correct, ents_lookup, convert_bs, weights=None, cutoff=0.25,
               remove_np=True, remove_unk=True):

    rev_ents = {v: k for k, v in ents_lookup.items()}

    predictions = list(predictions.squeeze())
    correct = list(correct.squeeze())

    all_ps = [rev_ents[element] for element in predictions]

    all_ls = [rev_ents[element] for element in correct]

    # Remove 'NP'
    if remove_np:
        all_ps = [p if '-NP' not in p else 'O' for p in all_ps]
        all_ls = [l if '-NP' not in l else 'O' for l in all_ls]

    # Remove 'UNK'
    if remove_unk:
        all_ps = [p if '-UNK' not in p else 'O' for p in all_ps]
        all_ls = [l if '-UNK' not in l else 'O' for l in all_ls]

    if convert_bs:
        first = weights < cutoff
        all_ps = ['B-' + p[2:] if (first[ix] and p != 'O') else p for ix, p in enumerate(all_ps)]

    prec, _ = compute_precision(all_ps, all_ls)
    rec, _ = compute_precision(all_ls, all_ps)

    f1 = calc_f1(prec, rec)

    return prec, rec, f1


def f1_multi_layer(predictions, labels, remove_np=True, remove_unk=True, add_bs=True, all_layers=True):
    """
    F1 score when output is multilevel - used for ontonotes dataset
    """
    # Just use first row of labels
    all_ls = defaultdict(list)

    # Use ents from first and second rows
    for l in labels:
        for k in l.keys():
            all_ls[k] += l[k]

    all_ps = defaultdict(list)
    predictions = copy.deepcopy(predictions)

    # Add bs back to start of preds
    for sent in predictions:
        for k, v in sent.items():
            for inner in v:
                if add_bs:
                    if inner[0] != 'O':
                        inner[0] = 'B-' + inner[0][2:]
                all_ps[k] += inner

    for k in all_ls.keys():
        # Remove 'NP'
        if remove_np:
            all_ps[k] = [p if '-NP' not in p else 'O' for p in all_ps[k]]
            all_ls[k] = [l if '-NP' not in l else 'O' for l in all_ls[k]]

        # Remove 'UNK'
        if remove_unk:
            all_ps[k] = [p if '-UNK' not in p else 'O' for p in all_ps[k]]
            all_ls[k] = [l if '-UNK' not in l else 'O' for l in all_ls[k]]

    ks = [k for k in all_ls.keys() if k > 0]

    if all_layers:
        assert remove_unk and remove_np, 'Need to change if not removing unk and np'
        for k in ks:
            for ix, i in enumerate(all_ls[k]):
                if i != 'O' and all_ls[0][ix] != i:
                    all_ls[0][ix] = i
                    all_ps[0][ix] = all_ps[k][ix]

    # Compute f1
    prec, error_ixs = compute_precision(all_ps[0], all_ls[0])
    rec, rec_error_ixs = compute_precision(all_ls[0], all_ps[0])

    wrong = list(set(error_ixs + rec_error_ixs))

    f1 = calc_f1(prec, rec)

    return prec, rec, f1, wrong


def extract_ents(ls):
    """
    Given a list of ents in format ['B-PER', 'I-PER', 'O', 'B-ORG'] extract all the indices of the ents and their type in format
    (0, 2, PER)
    """
    ents = []
    started = False
    for ix, l in enumerate(ls):
        if l[0:2] == 'B-':
            if started:
                ents.append((st, ix, tp))
            started = True
            st = ix
            tp = l[2:]

        elif l == 'O':
            if started:
                ents.append((st, ix, tp))
            started = False

    if started:
        if ix == (len(ls) - 1):
            ix += 1
        ents.append((st, ix, tp))

    return set(ents)


def extract_from_layers(ls):
    """
    Given a dict of outputs from each layer, extract all ents into tuples of form (start_ix, end_ix, type)
    """
    sep_ls = []

    for v in ls.values():
        sep_ls += extract_ents(v)

    return set(sep_ls)


def f1_ace(predictions, labels, remove_np=True, remove_unk=True):
    """
    F1 calculation for ACE05 dataset
    """
    # Join labels into single lists
    all_ls = defaultdict(list)

    for ix, l in enumerate(labels):
        for k in l.keys():
            all_ls[k] += l[k]

    # Add bs back to start of preds
    all_ps = defaultdict(list)
    predictions = copy.deepcopy(predictions)

    for sent in predictions:
        for k, v in sent.items():
            for inner in v:
                if inner[0] != 'O':
                    inner[0] = 'B-' + inner[0][2:]
                all_ps[k] += inner

    for k in all_ls.keys():
        # Remove 'NP'
        if remove_np:
            all_ps[k] = [p if '-NP' not in p else 'O' for p in all_ps[k]]
            all_ls[k] = [l if '-NP' not in l else 'O' for l in all_ls[k]]

        # Remove 'UNK'
        if remove_unk:
            all_ps[k] = [p if '-UNK' not in p else 'O' for p in all_ps[k]]
            all_ls[k] = [l if '-UNK' not in l else 'O' for l in all_ls[k]]

    # Extract labels into tuples of form (st_ix, end_ix, type)
    sep_ls = extract_from_layers(all_ls)

    # Extract preds into tuples of form (st_ix, end_ix, type)
    sep_ps = list(extract_from_layers(all_ps))

    # Just the indices of the ents
    ps_clusters = [i[0:2] for i in sep_ps]
    ls_clusters = set([i[0:2] for i in sep_ls])

    # Calculate F1
    wrong = []
    acc_wrong = 0
    cl_wrong = 0

    p = 0
    for ix, prd in enumerate(sep_ps):
        if prd in sep_ls:
            p += 1
        else:
            wrong.append(prd[0])
            if ps_clusters[ix] in ls_clusters:
                acc_wrong += 1
            else:
                cl_wrong += 1

    if len(sep_ps) > 0:
        prec = p / len(sep_ps)
        recall = p / len(sep_ls)
    else:
        prec, recall = 0.0, 0.0

    f1 = calc_f1(prec, recall)

    return prec, recall, f1, wrong, acc_wrong, cl_wrong


def compute_precision(guessed, correct):
    correctCount = 0
    count = 0
    idx = 0

    wrong = []
    while idx < len(guessed):
        if guessed[idx][0] == 'B':  # a new chunk starts
            count += 1

            if guessed[idx] == correct[idx]:  # first prediction correct
                idx += 1
                correctlyFound = True

                while idx < len(guessed) and guessed[idx][0] == 'I':  # scan entire chunk
                    if guessed[idx] != correct[idx]:
                        correctlyFound = False
                        wrong.append(idx)

                    idx += 1

                if idx < len(guessed):
                    if correct[idx][0] == 'I':  # chunk in correct was longer
                        correctlyFound = False
                        wrong.append(idx)

                if correctlyFound:
                    correctCount += 1
            else:
                idx += 1
                wrong.append(idx)

        else:
            idx += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision, wrong
