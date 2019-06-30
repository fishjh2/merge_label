import os
import copy
from sklearn.externals import joblib

from mg_lb.data_loading.fs import WordIndexer, read_csv
from mg_lb.data_loading.prep_iterators import get_article_iterator
from mg_lb.problems.probs import problems
from mg_lb.data_loading.fl_embeds import generate_flair, split_for_flair, add_flair_embeddings


def prep_data(args, problem, prob_dict, vocab, gpt, temp_store, main_temp):
    """
    Prep vocab object and data iterators for a pos problem
    Args
    problem: the string name of the problem e.g. one of ['quora', 'sentence_similarity' etc.]
    Returns
    iterators: dictionary where the keys are the dataset names, and the values are the iterators
               for each dataset
    vocab: WordIndexer object holding int_to_word and word_to_int methods
    """
    base = './data/problems/' + problem + '/'

    # Clear previous data store and make new one
    os.makedirs(temp_store, exist_ok=False)

    if gpt:
        newbase = base + 'gpt_data/'
    else:
        newbase = base + 'data/'

    # Get the datasets (train, val, test etc.)
    dsets = [d for d in os.listdir(newbase) if '.~lock' not in d]

    # Pre-generate flair embeddings for this problem
    fl = [e[6:] for e in args.embeddings if 'flair_' in e]

    use_flair = len(fl) > 0 and 'flair_splits' in problems[problem]

    if use_flair:
        fl_splits = problems[problem]['flair_splits']

        for f in fl:
            generate_flair(problem, f)

    if vocab is None:
        vocab = WordIndexer(problem + '_vocab')

    for d in dsets:

        dpath = newbase + d

        data, tokenized = read_csv(problem, dpath, args.add_upper)

        if 'train' in d and use_flair:
            all_dsets = split_for_flair(data, fl_splits)
        else:
            all_dsets = {d: data}

        ks = copy.copy(list(all_dsets.keys()))

        for name in ks:

            ddict = all_dsets[name]

            it, vocab = iterator_load(name, problem, ddict, vocab, args, use_flair, fl, not gpt,
                                      prob_dict, tokenized)

            if name in ['val.txt', 'test.txt']:
                name = name.split('.txt')[0]
            else:
                name = name[:-4]

            joblib.dump(it, temp_store + name)

    joblib.dump(vocab, main_temp + '/vocab')

    return True


def iterator_load(name, problem, ddict, vocab, args, use_flair, fl, extend_vocab, prob_dict, tokenized,
                  loud=True):

    if loud:
        print('Loading {} data'.format(name))

    # Add flair embeddings
    if use_flair:
        ddict = add_flair_embeddings(ddict, name, problem, fl)

    # Use article iterator
    it, vocab = get_article_iterator(ddict, vocab, args, problem=problem, prob_dict=prob_dict,
                                     extend_vocab=extend_vocab, use_flair=use_flair)

    return it, vocab



