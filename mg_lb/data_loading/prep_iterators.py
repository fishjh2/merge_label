import itertools
from collections import defaultdict
import math
import torch

from mg_lb.data_loading.prep_batches import prep_sent_batch, prep_pos_batches, prep_flair_batch
from mg_lb.data_loading.fs import basic_iterator
from mg_lb.models.layers.general import get_kernel_sizes, get_cl_levels
from mg_lb.models.model_args import model_args


def get_article_iterator(data, vocab, args, problem, prob_dict, extend_vocab, use_flair, margs=None):
    """
    Returns iterator for data.
    """
    if margs is None:
        margs = model_args[args.model]

    # Join into continuous text, and batch smaller articles together
    newdata, batch_order = join_and_batch(data, args.batch_size, args.bucket, args.one_article,
                                          split_sentences=args.split_sentences)

    data_dict = defaultdict(list)

    for data in newdata:
        num_batches = 1
        batch_size = len(data['sentences'])

        # Sentence batches
        sts, ls, caps, vocab, _ = prep_sent_batch(num_batches, batch_size, data['sentences'], vocab, progress=False,
                                            extend_vocab=extend_vocab, mask_answerhere=False, tokenized=True, merge_ents=False,
                                            cap_features=args.cap_features)

        torch_sents = torch.from_numpy(sts[0])
        data_dict['sentences'].append(torch_sents)
        data_dict['len'].append(torch.from_numpy(ls[0]))

        if args.cap_features:
            data_dict['cap_indices'].append(torch.from_numpy(caps[0]))

        # Flair embedding batches
        if use_flair:
            fl_batch = prep_flair_batch(num_batches, batch_size, data['flair'])[0]
            data_dict['flair'].append(fl_batch)

        # Sort pos label batches
        if prob_dict['load_ent'] or prob_dict['load_weights']:

            if margs['type'] == 'transformer':
                cl_levs = args.num_levels
            else:
                cl_levs = get_cl_levels(args.layer_list)

            pos, previous, mask, boolean, orig = prep_pos_batches(num_batches, batch_size, data[problem], problem,
                                                                  num_cluster_levels=cl_levs, all_splits=args.all_splits,
                                                                  convert_bs=True)

            assert len(pos) == 1, 'Should only be one batch as have run join_and_batch already'

            if prob_dict['load_ent']:
                data_dict['ent_labels'].append(torch.from_numpy(pos[0]).long())
                data_dict['ents_mask'].append(torch.from_numpy(mask[0]).byte())
                data_dict['original_labels'].append(torch.from_numpy(orig[0]))

            data_dict['previous'].append(torch.from_numpy(previous[0]))
            data_dict['boolean_mask'].append(torch.from_numpy(boolean[0]).byte())

        # Key to identify if new article or not
        data_dict['new_article'].append(data['new_article'])
        data_dict['mixed_article'].append(data['mixed_article'])

        if margs['kernel']:

            # Add sequence length mask for padded values [batch, seq_len, 1]
            msk = (torch_sents != 0).unsqueeze(2).float()
            data_dict['mask'].append(msk)

            sz = torch_sents.size()

            if margs['type'] == 'transformer':
                ksizes = [args.trans_kernel_size]
            else:
                ksizes = get_kernel_sizes(args.expand_ratio, len(args.layer_list), args.max_kernel_size)
                ksizes = list(set(ksizes + [args.first_kernel_size]))

            kmasks = kernel_masks(ksizes, sz[0], sz[1], args.embed_size, msk)

            data_dict['kmasks'].append(kmasks)

    # Create iterator which returns batches when method next_batch() called
    iterator = basic_iterator(data_dict, lab_key=problem, batch_order=batch_order,
                              shuffle=args.shuffle_batches)

    return iterator, vocab


def join_and_batch(data, max_batch_size, bucket, one_article, split_sentences=False):
    """
    Join articles into continuous text, and batch, by joining smaller articles into a single
    batch, and splitting longer articles into multiple batches
    """
    if split_sentences:
        # Leave sentences separate
        art_ix = 0
        new = {}

        for v in data.values():
            num = len(v['data']['sentences'])
            keys = v['data'].keys()
            for n in range(num):
                in_new = {'type': v['type'], 'data': {}}
                for k in keys:
                    in_new['data'][k] = v['data'][k][n]
                in_new['article_length'] = len(in_new['data']['sentences'])

                if in_new['article_length'] > 1:
                    new[art_ix] = in_new
                    art_ix += 1
        data = new

    else:
        # Join sentences into continuous text
        for v in data.values():

            keys = v['data'].keys()

            for k in keys:
                inner = v['data'][k]
                if type(inner[0]) == dict:
                    first = inner[0]
                    keys_inner = first.keys()
                    for d in inner[1:]:
                        for in_k in keys_inner:
                            first[in_k] += d[in_k]
                    v['data'][k] = first
                else:
                    v['data'][k] = list(itertools.chain.from_iterable(v['data'][k]))

            v['article_length'] = len(v['data']['sentences'])

    # Convert data to list of tuples
    data = [(k, v) for k, v in data.items()]

    # Sort articles by length
    data = sorted(data, key=lambda x: x[1]['article_length'])

    def reset_dict(keys):
        nd = {}
        for k in keys:
            nd[k] = []
        return nd

    # Join sentences into continuous text and batch smaller articles together
    newdata = []
    keys = data[0][1]['data'].keys()

    nd = reset_dict(keys)
    if bucket:
        allow_batch = True
    else:
        allow_batch = False

    if one_article:
        allow_batch = False

    num_arts = 0

    batch_order = []

    btch_ix = 0

    for d in data:

        data_in = d[1]['data']
        l = d[1]['article_length']
        # No longer allow batching after have reached articles of longer than max_batch_size
        if l >= max_batch_size and allow_batch:
            allow_batch = False
            # Append previous unfinished batch
            if len(nd['sentences']) > 0:
                nd['new_article'] = True
                nd['mixed_article'] = True
                newdata.append(nd)
                nd = reset_dict(keys)
                batch_order.append([btch_ix])
                btch_ix += 1

        if allow_batch:

            # Batch smaller articles together
            num_arts += 1
            if (num_arts * l) < max_batch_size:
                for k in keys:
                    nd[k].append(data_in[k])
            else:
                nd['new_article'] = True
                nd['mixed_article'] = True
                newdata.append(nd)
                batch_order.append([btch_ix])
                btch_ix += 1
                nd = reset_dict(keys)
                for k in keys:
                    nd[k].append(data_in[k])
                num_arts = 1

        else:
            # Split larger articles into multiple batches
            numsplits = math.ceil(l / max_batch_size)
            splitsize = math.ceil(l / numsplits)

            btch = []

            for sp in range(numsplits):
                nd = reset_dict(keys)
                for k in keys:
                    if type(data_in[k]) == dict:
                        newdict = {}
                        for ink in data_in[k].keys():
                            newdict[ink] = data_in[k][ink][sp * splitsize: (sp * splitsize + splitsize)]
                        nd[k].append(newdict)
                    else:
                        nd[k].append(data_in[k][sp * splitsize: (sp * splitsize + splitsize)])

                if sp == 0:
                    nd['new_article'] = True
                else:
                    nd['new_article'] = False

                btch.append(btch_ix)
                btch_ix += 1

                nd['mixed_article'] = False
                newdata.append(nd)

            batch_order.append(btch)

    # Edge case where no articles longer than max_batch_size - need to add last batch
    if allow_batch:
        # Append previous unfinished batch
        if len(nd['sentences']) > 0:
            nd['new_article'] = True
            nd['mixed_article'] = True
            newdata.append(nd)
            batch_order.append([btch_ix])

    return newdata, batch_order


def kernel_masks(kernel_sizes, batch_size, seq_len, embed_size, mask):
    """
    Generate masks for sequence length when tensors unfolded to have last dimension kernel_size
    Args
    mask: [batch, seq_len, 1]
    """
    masks = {}

    for k in kernel_sizes:

        half_k = int(k / 2)

        # Mask for seq_len. [batch, seq_len, 1]
        es = torch.ones([batch_size, seq_len, embed_size]) * mask

        # Zero pad on either end
        es = torch.nn.functional.pad(es, (0, 0, half_k, half_k), "constant", 0.0)

        # Unfold to shape [batch_size, seq_len, embed_size, kernel_size+1]
        out = es.unfold(dimension=1, size=k + 1, step=1)

        # Reshape to [batch, seq_len, kernel_size+1, embed_size]
        out = out.transpose(2, 3)

        # Mask. This masks out links that are linking to a greater than seq_len word
        # Shape [batch, seq_len, kernel_size+1, 1]
        m = (out[:, :, :, 0] != 0.0).unsqueeze(3).float()

        masks[k] = m

    return masks






