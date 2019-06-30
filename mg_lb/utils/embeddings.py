import os
import numpy as np
import six
import array
from tqdm import tqdm
import torch
import io
import gensim
import operator
from sklearn.decomposition import PCA

from mg_lb.data_loading.fs import tokenize, predict_word


def pcs_from_embeddings(embeds, num_pcs):
    print('Calculating principal components as targets...')
    pca = PCA(n_components=num_pcs)
    np_embeds = embeds.data.numpy()
    pcs_np = pca.fit_transform(np_embeds)
    return torch.from_numpy(pcs_np)


def get_wv_dict(embed_type):
    if 'gensim' in embed_type:
        model = gensim.models.Word2Vec.load(embedding_paths[embed_type])
        return model.wv.vocab


def gensim_from_torch(embeds, vocab, use_indices=False):
    """
    Given a pytorch embedding tensor and vocab object, create a gensim word2vec instance from the
    trained embeddings, which can be used for nearest neighbour lookup etc
    """
    embeds = embeds.numpy()

    dim = embeds.shape[1]

    kv = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=dim)

    # Sort words based on index values
    if use_indices:
        sorted_words = range(embeds.shape[0])
    else:
        sorted_words = [i[0] for i in sorted(vocab.word_to_index.items(), key=operator.itemgetter(1))]

    kv.add(sorted_words, embeds)

    return kv


def gensim_from_dict(d):
    """
    Given a dictionary where the keys are words and the values are embeddings, create a gensim word2vec
    instance for looking up nearest neighbours
    """
    dim = d[list(d.keys())[0]].shape[0]
    kv = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=dim)

    kv.add(list(d.keys()), list(d.values()))

    return kv


def random_vector(size):
    return torch.from_numpy(np.random.normal(size=[size], scale=0.2)).float()


def lookup_vector(e, w, unknown_words, gensim=False):
    known = True

    if gensim:
        dim = e.vector_size
    else:
        dim = e.dim

    try:
        if gensim:
            vector = torch.from_numpy(e[w])
        else:
            vector = e.vectors[e.ix_lookup[w]]

    except KeyError:
        # <pad> and <unk> left as zeros
        known = False
        unknown_words.append(w)
        if w == predict_word:
            vector = torch.zeros([dim])
        # elif 'tttt' in w:
        #    # Use a random glove vector (to avoid model learning what a randomly generated vector is)
        #    ix = random.choice(range(len(e.vectors)))
        #    vector = e.vectors[ix]
        elif w in ['<PAD>', '<MASK>']:
            vector = torch.zeros([dim])
        else:
            vector = torch.zeros([dim])
            # vector = random_vector(dim)

    return vector, unknown_words, known


def embeddings_from_gensim(embed_path, target_vocab):
    """
    Load an embedding tensor for this vocab
    Args
    embed_path: path to gensim model
    target_vocab: vocab object holding words for this problem
    Returns
    embeds: [vocab_size, embed_size] tensor holding embeddings
    """
    total_words = target_vocab.n_words

    if '/fasttext/' in embed_path:
        model = gensim.models.KeyedVectors.load(embed_path)
    else:
        # Create and/or load torch version of embeddings
        model = gensim.models.Word2Vec.load(embed_path)

    # Empty embeddings tensor
    embeddings = torch.zeros([target_vocab.n_words, model.vector_size])

    # Loop through words in vocab and create embeddings file
    unknown_words = []

    for w in list(target_vocab.word_to_index.keys()):
        ix = target_vocab.word_to_index[w]
        v, unknown_words, known = lookup_vector(model, w, unknown_words, gensim=True)
        embeddings[ix] = v

    del model

    print('{} out of {} words unknown'.format(len(unknown_words), total_words))

    return embeddings


def embeddings_from_torch(embed_path, target_vocab, join_multi=False):
    """
    Load an embedding tensor for this vocab
    Args
    embed_path: path to embedding .pt file
    target_vocab: vocab object holding words for this problem
    Returns
    embeds: [vocab_size, embed_size] tensor holding embeddings
    """
    total_words = target_vocab.n_words

    # Create and/or load torch version of embeddings
    e = pretrained_vectors(embed_path)

    # Empty embeddings tensor
    embeddings = torch.zeros([target_vocab.n_words, e.dim])
    # embeddings_mask = torch.zeros([target_vocab.n_words, 1])

    # Loop through words in vocab and create embeddings file
    unknown_words = []

    for w in list(target_vocab.word_to_index.keys()):
        ix = target_vocab.word_to_index[w]

        if join_multi and ' ' in w:
            split_words = tokenize(w)

            # Remove 'the' if first word
            if split_words[0] in ['the', 'The']:
                split_words = split_words[1:]

            total_words += (len(split_words) - 1)
            # Get embedding tensor for each of the separate words and sum
            joint_vector = torch.zeros([e.dim]).float()

            if len(split_words) > 0:
                for s in split_words:
                    v, unknown_words, _ = lookup_vector(e, s, unknown_words)
                    joint_vector += v
                joint_vector = joint_vector / len(split_words)
            else:
                joint_vector = random_vector(e.dim)

            embeddings[ix] = joint_vector
            # embeddings_mask[ix] = 1

        else:
            v, unknown_words, known = lookup_vector(e, w, unknown_words)
            embeddings[ix] = v
            # if not known:
            #    embeddings_mask[ix] = 1

    del e

    print('{} out of {} words unknown'.format(len(unknown_words), total_words))

    return embeddings


class pretrained_vectors(object):
    """
    Read in torch tensor holding pretrained embeddings stored in path. If hasn't already been
    carried out, we convert an embedding .txt file to a .pt file holding the torch tensor, which
    is much faster for building a vocabulary with
    """

    def __init__(self, path):
        self.path = path
        self.path_pt = path + '.pt'
        self.get_vectors()

    def get_vectors(self):
        if not os.path.isfile(self.path_pt):
            # str call is necessary for Python 2/3 compatibility, since
            # argument must be Python 2 str (Python 3 bytes) or
            # Python 3 str (Python 2 unicode)
            itos, vectors, dim = [], array.array(str('d')), None

            # Try to read the whole file with utf-8 encoding.
            binary_lines = False
            try:
                with io.open(self.path, encoding="utf8") as f:
                    lines = [line for line in f]
            # If there are malformed lines, read in binary mode
            # and manually decode each word from utf-8
            except:
                print("Could not read {} as UTF8 file, "
                      "reading file as bytes and skipping "
                      "words with malformed UTF8.".format(self.path))
                with open(self.path, 'rb') as f:
                    lines = [line for line in f]
                binary_lines = True

            print("Building Pytorch vectors from {}".format(self.path))
            for line in tqdm(lines, total=len(lines)):
                # Explicitly splitting on " " is important, so we don't
                # get rid of Unicode non-breaking spaces in the vectors.
                entries = line.rstrip().split(b" " if binary_lines else " ")

                word, entries = entries[0], entries[1:]
                if dim is None and len(entries) > 1:
                    dim = len(entries)
                elif len(entries) == 1:
                    print("Skipping token {} with 1-dimensional vector {}; likely a header".format(word, entries))
                    continue
                elif dim != len(entries):
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, but previously "
                        "read vectors have {} dimensions. All vectors must have "
                        "the same number of dimensions.".format(word, len(entries), dim))

                if binary_lines:
                    try:
                        if isinstance(word, six.binary_type):
                            word = word.decode('utf-8')
                    except:
                        print("Skipping non-UTF8 token {}".format(repr(word)))
                        continue
                vectors.extend(float(x) for x in entries)
                itos.append(word)

            self.word_list = itos
            self.ix_lookup = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            print('Saving vectors to {}'.format(self.path_pt))
            torch.save((self.word_list, self.ix_lookup, self.vectors, self.dim), self.path_pt)
        else:
            self.word_list, self.ix_lookup, self.vectors, self.dim = torch.load(self.path_pt)

# Lookup dictionary for pretrained embeddings (path to .txt file not .pt file)

embedding_paths = {
    'glove': './data/pre_trained_embeddings/glove/glove.840B.300d.txt'
}

embedding_functions = {
    'glove': embeddings_from_torch
}