from flair.data import Sentence, Token
from flair.embeddings import FlairEmbeddings, ELMoEmbeddings, BertEmbeddings
import os
from tqdm import tqdm
from sqlitedict import SqliteDict
from random import shuffle
from collections import defaultdict
import torch
import itertools
import copy

from mg_lb.data_loading.fs import read_csv


def split_for_flair(data, num_splits):
    """
    Split dataset into num_splits separate datasets
    """
    # Shuffle article keys
    keys = list(data.keys())
    shuffle(keys)

    # Split into equal sized chunks
    def chunkify(lst, n):
        return [lst[i::n] for i in range(n)]

    keys = chunkify(keys, num_splits)

    keys = [k for k in keys if len(k) > 0]

    rt_d = {}

    for ix, ch in enumerate(keys):
        new = {}

        for k in ch:
            new[k] = data[k]

        rt_d['train_' + str(ix) + '.txt'] = new

    return rt_d


def sent_to_flair(sent):
    """
    Convert a tokenized sentence (list of words) to a Flair sentence object
    """
    sentence = Sentence()

    for w in sent:
        token = Token(w)
        sentence.add_token(token)
        sentence.infer_space_after()

    return sentence


def chunks(l, n):
    n = max(1, n)
    return list(l[i:i+n] for i in range(0, len(l), n))


def split_sentences(sents, toks_per_batch, fl_embed):

    l = copy.deepcopy(sents)

    all_bs = []
    b = []
    tok_b = []
    for s in l:
        jned = list(itertools.chain.from_iterable([fl_embed.tokenizer.tokenize(i) for i in s]))

        if len(tok_b) + len(jned) < toks_per_batch:
            b += s
            tok_b += jned
        else:
            if len(b) > 0:
                all_bs.append(b)
            b = s
            tok_b = jned
    if len(b) > 0:
        all_bs.append(b)

    return all_bs


def get_flair_embeds(sentences, fl_embed, toks_per_batch=500):
    """
    Get flair embeddings for tokenized sentences
    """
    with_embeds = []

    # Join sentences into batches
    sp = split_sentences(sentences, toks_per_batch, fl_embed)

    for sent in sp:
        # Convert to Flair sentence object
        fl_sent = sent_to_flair(sent)
        fl_sent = [fl_sent]
        fl_embed.embed(fl_sent)

        with_embeds += fl_sent[0]

    # Add embeddings as lists of same length as sentences
    assert len(with_embeds) == len(list(itertools.chain.from_iterable(sentences)))
    ix = 0
    art_embeds = []

    for s in sentences:
        sep_es = with_embeds[ix:(ix + len(s))]
        es = [i.embedding for i in sep_es]
        art_embeds.append(es)
        ix += len(s)

    return art_embeds


def add_flair_embeddings(data, dset, problem, fl_embeds):
    """
    Given a data dictionary, add any flair embeddings as a np array
    """
    if 'train_' in dset:
        dset = 'train.sq'
    else:
        dset = dset.split('.txt')[0] + '.sq'

    fl_es = defaultdict(list)

    for fl in fl_embeds:
        file_path = './data/problems/' + problem + '/flair_embeds/' \
                    + fl + '/' + dset

        if not os.path.exists(file_path):
            generate_flair(problem, fl)

        fl_dict = SqliteDict(file_path)

        for k in data.keys():
            fl_es[k].append(fl_dict[k])

    # Join different flair embeddings into one
    data = join_flair(data, fl_es)

    return data


def eval_flair(data, embed_types):
    """
    Add flair embeds to a new article for eval
    """
    fl_es = defaultdict(list)

    for e in embed_types:
        fl_embed = get_flair_class(e)

        for art, val in tqdm(data.items()):
            sentences = val['data']['sentences']

            # Generate the embeddings for the sentences
            art_embeds = get_flair_embeds(sentences, fl_embed)

            fl_es[art].append(art_embeds)

    # Join different flair embeddings into one
    data = join_flair(data, fl_es)

    return data


def join_flair(data, fl_es):
    """
    Join all different flair embeddings into one array for each word
    """
    for k in data.keys():
        # Length num_sentences_in_article list
        sents = list(zip(*fl_es[k]))

        # Concatenate all embeddings for each word and expand to shape [1, embed_size*n]
        sents = [[torch.cat(i).unsqueeze(dim=0) for i in list(zip(*e))] for e in sents]

        data[k]['data']['flair'] = sents

        assert len(sents) == len(data[k]['data']['sentences'])

    return data


def get_flair_class(embed_type):
    """
    Return the correct flair class for the embed type
    """
    if embed_type == 'elmo':
        fl_embed = ELMoEmbeddings()
    elif embed_type == 'bert':
        fl_embed = BertEmbeddings()
    else:
        fl_embed = FlairEmbeddings(embed_type)
    return fl_embed


def generate_flair(problem, embed_type):
    """
    Generate flair embeddings for this problem and embed type
    """
    fl_embed = get_flair_class(embed_type)

    bp = './data/problems/' + problem
    base = bp + '/data/'

    dsets = [d for d in os.listdir(base) if '~' not in d]

    fl_path = bp + '/flair_embeds/' + embed_type

    # Create folder for flair embeddings if doesn't exist
    os.makedirs(fl_path, exist_ok=True)

    for dset in dsets:
        print('Generating {} {} flair embeds'.format(dset, embed_type))

        file_path = fl_path + '/' + dset.split('.txt')[0] + '.sq'

        if not os.path.isfile(file_path):

            # Initialize sqlite dict
            art_dict = SqliteDict(file_path, autocommit=True)

            # Read in dataset
            data, _ = read_csv(problem, base + dset, add_upper=False)

            for art, val in tqdm(data.items()):
                sentences = val['data']['sentences']

                # Generate the embeddings for the sentences
                art_embeds = get_flair_embeds(sentences, fl_embed)

                art_dict[art] = art_embeds