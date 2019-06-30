import pandas as pd
import numpy as np
import spacy
import random
import copy
import itertools

from mg_lb.problems.probs import labels_lookup


def pd_to_dict(df, dtype=list, index_keys=False):
    """
    Convert pandas df to dict
    """
    d = {}
    if index_keys:
        for i in df.index:
            d[i] = {}
            for c in df.columns:
                d[i][c] = df.loc[i, c]
    else:
        for c in df.columns:
            d[c] = dtype(df[c])
    return d


def dict_to_pd(d):

    df = pd.DataFrame(columns=list(d.keys()))

    for k,v in d.items():
        df[k] = v

    return df


nlp = spacy.load('en')
spacy.tokens.token.Token.set_extension('transient', default='', force=True)

predict_word = 'answerhere'
sentence_split = '---SENT-SPLIT---'


def sent_tokenize(s):
    s = nlp(s)
    return [str(i) for i in list(s.sents)]


def tokenize(s):

    doc = nlp(s)

    for ix, tok in enumerate(doc):
        # Set the transient attribute to the original text
        tok._.transient = tok.text

    tokens = [t._.transient for t in doc]

    return tokens


def sent_to_idx(tokens, vocab):
    """ Transforms a sequence of strings to the corresponding sequence of indices. """
    idx_list = [vocab.word_to_index[word] if word in vocab.word_to_index else 1 for
                word in tokens]
    return idx_list


def idx_to_sent(tokens, vocab):
    sent = [vocab.index_to_word[i] for i in tokens if i != 0]
    return ' '.join(sent)


def split_articles(df):
    """
    Given a df in article format, split into a dictionary where each value is a separate article
    """
    articles = {}

    sentences = list(df['sentences'])

    count = 0

    for ix, w in enumerate(sentences):
        if w == '---NEW-ARTICLE---':
            end_ix = ix - 1

            if end_ix > 5:
                articles[count] = {'type': desc, 'data': df.iloc[start_ix + 3:end_ix, :]}
                count += 1
            start_ix = ix
            desc = df.iloc[ix + 1, 0]

    # Add last article
    start_ix = end_ix + 1
    articles[count] = {'type': df.iloc[start_ix+1, 0], 'data': df.iloc[start_ix + 2:, :]}

    return articles


def parse_pos_data(df, lookup, convert_ents):
    """
    Given a df read from a pos style csv, parse into lists of sentences, labels and corefs
    """
    # Split into lists of sentences and associated labels
    sents = list(df['sentences'])
    pos_cols = [c for c in df.columns if 'Ents_' in c]
    coref_cols = [c for c in df.columns if 'Corefs_' in c]

    cs = len(coref_cols) > 0

    if cs:
        assert len(pos_cols) == len(coref_cols)

    pos = {}
    corefs = {}

    for i in pos_cols:
        pos[i] = list(df[i])

    if cs:
        for c in coref_cols:
            corefs[c] = list(df[c])

    sentences = []
    pos_labels = []
    coref_labels = []

    def new_dict(pc):
        n = {}
        for i in pc:
            n[i] = []
        return n

    current_sentence = []
    current_labels = new_dict(pos_cols)
    current_corefs = new_dict(coref_cols)

    for ix, word in enumerate(sents):

        if word != sentence_split:
            assert type(word) == str, ''.format(word)
            current_sentence.append(word)
            for i in pos_cols:
                v = pos[i][ix]
                if convert_ents:
                    if type(v) == float:
                        # Put nans as 'O's
                        current_labels[i].append(lookup['O'])
                    else:
                        ent_val = lookup[v]
                        current_labels[i].append(ent_val)
                else:
                    current_labels[i].append(v)

            if cs:
                for i in coref_cols:
                    current_corefs[i].append(corefs[i][ix])

        else:

            assert len(current_sentence) == len(current_labels['Ents_0'])
            if cs:
                assert len(current_sentence) == len(current_corefs['Corefs_0'])

            if len(current_sentence) > 0:
                sentences.append(current_sentence)
                pos_labels.append(current_labels)

                if cs:
                    coref_labels.append(current_corefs)

            current_sentence = []
            current_labels = new_dict(pos_cols)
            if cs:
                current_corefs = new_dict(coref_cols)

    # Add the last sentence if it doesn't have a nan after it
    if len(sentences) == 0 or len(current_sentence) > 0:
        if len(current_sentence) > 0:
            sentences.append(current_sentence)
            pos_labels.append(current_labels)
            if cs:
                coref_labels.append(current_corefs)

    if not cs:
        coref_labels = None

    return sentences, pos_labels, coref_labels


def upper_levels(df):
    """
    Copy named ents / NPs to higher levels as well
    """
    col = 'Ents'

    corefs = len([c for c in df.columns if 'Corefs_' in c]) > 0

    d = pd_to_dict(df, dtype=np.array)

    # Ents columns indices
    es = [int(k.split(col + '_')[1]) for k in d.keys() if col + '_' in k]

    # Sort correctly
    es = sorted(es)

    newd = copy.deepcopy(d)

    # Set all cols to the first col
    for e in es:
        newd[col + '_' + str(e)] = copy.deepcopy(newd[col + '_0'])
        if corefs:
            newd['Corefs_' + str(e)] = copy.deepcopy(newd['Corefs_0'])

    # Write over higher level cols if have changed
    for e in es[1:]:
        higher = [i for i in es if i >= e]

        # Original data in this column
        orig = d[col + '_' + str(higher[0])].astype(str)
        if corefs:
            orig_corefs = d['Corefs_' + str(higher[0])].astype(str)

        notnan = orig != 'nan'
        noto = orig != 'O'
        notnan = noto * notnan

        notnan_vs = orig[notnan]
        if corefs:
            notnan_corefs = orig_corefs[notnan]

        # Overwrite higher levels
        for h in higher:
            colname = col + '_' + str(h)
            newd[colname][notnan] = notnan_vs
            if corefs:
                newd['Corefs_' + str(h)][notnan] = notnan_corefs

    return dict_to_pd(newd)


def read_article_csv(path, problem, convert_ents=True, add_upper=True):
    """
    Read in data for named entity problem in article format
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[''], sep='\t')

    # Copy ents and corefs to higher levels
    if add_upper:
        df = upper_levels(df)

    # Split into articles
    articles = split_articles(df)

    ent_lookup = labels_lookup[problem]

    parsed = {}

    for key, value in articles.items():
        df = value['data']

        sentences, labels, corefs = parse_pos_data(df, ent_lookup, convert_ents=convert_ents)

        data = {
            'sentences': sentences,
            problem: labels,
        }

        parsed[key] = {'type': value['type'], 'data': data}

    return parsed


def read_csv(problem, path, add_upper):
    """
    Read in data from csv. Second return is whether or not data tokenized
    """
    return read_article_csv(path, problem, add_upper=add_upper), True


class basic_iterator(object):
    """
    Basic batch iterator
    """

    def __init__(self, inputs, lab_key='labels', shuffle=False, memory=False, batch_order=None):
        self.inputs = inputs
        self.num_inputs = len(inputs.keys())
        self.lab_key = lab_key
        self.shuffle = shuffle
        self.memory = memory

        # Check all inputs same length
        assert len(set([len(i) for i in inputs.values()])) == 1, 'All inputs should be same length'

        self.counter = 0
        self.num_batches = len(list(inputs.values())[0])

        if batch_order is None:
            self.batch_list = [[i] for i in np.arange(self.num_batches)]
        else:
            self.batch_list = batch_order

        self.batch_order = self.get_batch_order()

    def next_batch(self):
        rts = {}

        for k, v in self.inputs.items():
            rts[k] = self.inputs[k][self.batch_order[self.counter]]

        rts['index'] = self.batch_order[self.counter]

        # Return a random subset of ents ixs if shuffling them in
        if hasattr(self, 'ents_ixs'):
            rts['ents_random'] = random.choices(self.ents_ixs, k=500)
            rts['ents_vals'] = self.ents_vals

        self.counter += 1

        if self.counter == self.num_batches:
            self.counter = 0
            self.batch_order = self.get_batch_order()

        return rts

    def get_batch_order(self):
        bl = copy.deepcopy(self.batch_list)
        if self.shuffle:
            np.random.shuffle(bl)

        return list(itertools.chain.from_iterable(bl))

    def all_labels(self, ixs=None):

        if ixs is None:
            ixs = list(range(self.num_batches))

        if self.memory:
            ans = []

            for ix, v in enumerate(self.inputs['questions']):
                if v is not None:
                    ans.append(self.inputs[self.lab_key][ix])
            return np.concatenate(ans, axis=0)

        elif self.lab_key in ['pos', 'ontonotes_ents', 'conll_ents', 'ontonotes_cleaned']:
            # Reshape from [batch, seq_len, 1] to [batch * seq_len, 1]
            # return np.concatenate([np.reshape(i, [-1, 1]) for i in self.inputs[self.lab_key]])
            rt = []
            lens = self.inputs['len']
            labs = self.inputs[self.lab_key]
            for ix in ixs:
                l = lens[ix]
                for in_ix, len in enumerate(l):
                    rt.append(labs[ix][in_ix][:len])

                # rt.append(np.expand_dims(b[b != -1], 1))

            return np.expand_dims(np.concatenate(rt, axis=0), 1)

        elif self.lab_key in ['ontonotes', 'ontonotes_test']:
            rt = []
            labs = self.inputs[self.lab_key]
            masks = self.inputs['ents_mask']
            for ix, b in enumerate(labs):
                rt.append(b.numpy()[masks[ix].numpy().astype(np.bool)])

            return np.expand_dims(np.concatenate(rt, axis=0), axis=1)

        else:
            return np.concatenate(self.inputs[self.lab_key], axis=0)

    def reset_order(self):
        bl = copy.deepcopy(self.batch_list)
        self.batch_order = list(itertools.chain.from_iterable(bl))


class WordIndexer(object):
    """ Translates words to their respective indices and vice versa. """

    def __init__(self, name):
        self.name = name
        self.word_to_count = {'<PAD>': 1, '<UNK>': 1, '<SOS>': 1, '<EOS>': 1, 'answerhere': 1, '<MASK>': 1}
        # Specify start-and-end-of-sentence tokens
        self.index_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>', 4: 'answerhere', 5: '<MASK>'}
        self.word_to_index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3, 'answerhere': 4, '<MASK>': 5}
        self.n_words = 6

        assert self.word_to_index['<EOS>'] == 3, 'If change this need to make same change in translation_loss fn'

        self.max_sentence_len = 0

    def add_word(self, word):
        """ Adds words to index dict. """
        if word not in self.word_to_index:
            self.word_to_index[word] = self.n_words
            self.index_to_word[self.n_words] = word
            self.word_to_count[word] = 1
            self.n_words += 1
        else:
            self.word_to_count[word] += 1

    def add_sentence(self, sentence):
        """ Adds sentence contents to index dict. """
        sp = tokenize(sentence)
        for word in sp:
            self.add_word(word)

        if len(sp) > self.max_sentence_len:
            self.max_sentence_len = len(sp)

        return len(sp)









