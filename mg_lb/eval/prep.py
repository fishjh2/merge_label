import pickle
import torch
import os

from mg_lb.data_loading.prep_iterators import get_article_iterator
from mg_lb.training.trainer import nlp_eval
from mg_lb.data_loading.fs import read_csv, tokenize
from mg_lb.problems.probs import problems
from mg_lb.data_loading.fl_embeds import add_flair_embeddings, eval_flair
import mg_lb.models.models as modules
from mg_lb.training.utils.fs import probs_to_losses


def prep_for_eval(save_name, paper_model, gpu, model='merge_label', save_id='1'):
    """
    Load a saved model into memory ready for eval
    """
    if paper_model:
        saved_dir = './paper_models/'
    else:
        saved_dir = './saved_models/'

    saved_dir += save_name + '/'

    if not paper_model:
        saved_dir += model + '/' + save_id + '/'

    args = pickle.load(open(saved_dir + 'args.p', 'rb'))

    kwargs = {}

    if 'vocab.p' in os.listdir(saved_dir):
        vocab = pickle.load(open(saved_dir + 'vocab.p', 'rb'))
    else:
        vocab = None

    prob_dict = probs_to_losses(args.problems, args.loss_fn, args.loss_weights)

    args.gpu = gpu

    # Build the model
    model_class = getattr(modules, args.model)

    print('Building model...')
    model = model_class(args, vocab, **kwargs)

    # Restore the saved parameters (this includes the embeddings, regardless of whether or not
    # they were trained)
    print('Restoring model from checkpoint...')
    model.load_state_dict(torch.load(saved_dir + args.save_id))

    return model, vocab, args, prob_dict


def network_sentences(sentences, problem, args, vocab, model, prob_dict, gpu=True):
    """
    Given a list of sentences, prepare a data iterator object
    """
    data = prep_network_sentence(sentences, problem, False, vocab)

    # Add flair embeddings
    fl, use_flair = need_flair(args.embeddings, problem)

    if use_flair:
        data = eval_flair(data, fl)

    iterator, _ = get_article_iterator(data, vocab, args, problem=problem, prob_dict=prob_dict[problem],
                                       extend_vocab=False, use_flair=use_flair)

    res, iterator = run_network_forward(iterator, args, vocab, model, problem, prob_dict,
                                        batch_size=args.batch_size, gpu=gpu, eval_loss=False)

    return res, iterator


def prep_network_sentence(sentences, problem, gpt, vocab, num_cluster_levels=10):
    """
    Tokenize sentence and put in dict for input to iterator
    """
    tok = []
    labs = []
    corefs = []

    for s in sentences:
        if gpt:
            tk = vocab.split_word(s)
        else:
            tk = tokenize(s)

        tok.append(tk)

        ls = {}
        cs = {}
        for c in range(num_cluster_levels):
            ls['Ents_' + str(c)] = [0] * len(tk)
            cs['Corefs_' + str(c)] = [0] * len(tk)
        labs.append(ls)
        corefs.append(cs)

    d = {0: {'type': 'newswire', 'data': {'sentences': tok,
                                          problem: labs, 'corefs': corefs}}}

    return d


def eval_data_set(problem, dtype, args, vocab, model, prob_dict, extend_vocab=False, batch_size=50, gpu=True):
    """
    Run evaluation for a given problem and dataset
    """

    base = './data/problems/' + problem + '/'
    dpath = base + 'data/' + dtype

    if not hasattr(args, 'add_upper'):
        args.add_upper = False

    data, tokenized = read_csv(problem, dpath, add_upper=args.add_upper)

    # Pre-generate flair embeddings for this problem
    fl, use_flair = need_flair(args.embeddings, problem)

    # Add flair embeddings
    if use_flair:
        data = add_flair_embeddings(data, dtype, problem, fl)

    args.shuffle = False
    args.batch_size = batch_size

    # Use article iterator
    iterator, _ = get_article_iterator(data, vocab, args, problem=problem, prob_dict=prob_dict[problem],
                                       extend_vocab=extend_vocab, use_flair=use_flair)

    res, iterator = run_network_forward(iterator, args, vocab, model, problem, prob_dict=prob_dict, batch_size=batch_size,
                                        gpu=gpu, labels=True)

    return res, iterator


def need_flair(embeddings, problem):
    """
    Check if need flair embeddings
    """
    fl = [e[6:] for e in embeddings if 'flair_' in e]
    use_flair = len(fl) > 0 and 'flair_splits' in problems[problem]
    return fl, use_flair


def run_network_forward(iterator, args, vocab, model, problem, prob_dict, batch_size=400, gpu=True, labels=False,
                        eval_loss=True):
    """
    Forward pass of the network
    """

    if type(args.problems) == list:
        args.loss_fn = prob_dict[problem]['loss'][0]
        args.problems = problem

    args.gpu = gpu
    args.vocab = vocab
    args.shuffle = False
    args.batch_size = batch_size

    # Run forward pass
    res = nlp_eval(model, iterator, args, labels=labels, prob_dict=prob_dict,
                   show_progress=False, eval_loss=eval_loss)

    return res, iterator
