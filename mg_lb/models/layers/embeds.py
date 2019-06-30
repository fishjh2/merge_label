import torch.nn as nn
import torch

from mg_lb.utils.embeddings import embedding_paths, embedding_functions
from mg_lb.models.layers.linear import linear_layer

embed_sizes = {
    'glove': 300,
    'flair_news-forward': 2048,
    'flair_news-backward': 2048,
    'flair_elmo': 3072,
    'flair_bert': 3072,
}


class wordDropout(nn.Module):
    """
    Sets complete embedding of words to zero
    """
    def __init__(self, dropout, gpu):
        super(wordDropout, self).__init__()

        self.dropout = dropout
        self.gpu = gpu

    def forward(self, embeds):
        """
        embeds: [batch, seq_len, embed_size]
        """
        if self.training and self.dropout > 0.0:
            if self.gpu:
                mask = torch.cuda.FloatTensor(embeds.size(0), embeds.size(1)).uniform_() > (1 - self.dropout)

            else:
                mask = torch.FloatTensor(embeds.size(0), embeds.size(1)).uniform_() > (1 - self.dropout)

            embeds[mask] = 0.0

        return embeds


class embeds_layer(nn.Module):

    def __init__(self, vocab, embed_types, embed_size, embed_dropout, word_dropout, finetune,
                 activation_fn, init_std, gpu, cap_features, cap_features_size):

        super(embeds_layer, self).__init__()

        vocab_size = vocab.n_words
        self.shrink_embeds = False
        self.flair = False
        self.cap_features = cap_features

        if embed_types[0] not in ['train_own', 'gpt_train_own']:

            self.es = [e for e in embed_types if 'flair_' not in e]
            self.fl_es = [e for e in embed_types if 'flair_' in e]

            if len(self.fl_es) > 0:
                self.shrink_embeds = True
                self.flair = True

            if len(self.es) > 0:

                es_tensor = []

                for e in self.es:
                    embed_path = embedding_paths[e]
                    es_tensor.append(embedding_functions[e](embed_path, vocab))

                # Concat to shape [vocab_size, embed_sizes]
                es_tensor = torch.cat(es_tensor, dim=1)

                # Embedding object
                self.embedding = nn.Embedding(vocab_size, es_tensor.size(1))
                self.embedding.weight.data.copy_(es_tensor)

                if not finetune:
                    self.embedding.weight.requires_grad = False

                if es_tensor.size(1) != embed_size:
                    self.shrink_embeds = True

            else:
                self.embedding = None

        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.shrink_embeds = False

        # Capital letter features embedding
        if cap_features:
            self.cap_embedding = nn.Embedding(4, cap_features_size)
            self.total_size = embed_size + cap_features_size
        else:
            self.total_size = embed_size

        # Embedding dropout
        self.embed_dropout = nn.Dropout(embed_dropout)

        # Word dropout
        self.word_dropout = wordDropout(word_dropout, gpu)

        if self.shrink_embeds:
            total_size = sum([embed_sizes[e] for e in embed_types])

            self.reduce_layer = linear_layer(total_size, embed_size, dropout=0.0, activation_fn=activation_fn,
                                             w_init=('uniform', init_std), b_init=('constant', 0.1))

    def forward(self, sentences, cap_indices=None, flair=None):
        """
        sentences: [batch, seq_len]
        """
        # Usual embeds
        if self.embedding is not None:
            # [batch, seq_len, embed_size]
            embeds = self.embedding(sentences)

        # Flair embeds
        if self.flair:
            if self.embedding is not None:
                # [batch, seq_len, cumulative_embed_size]
                embeds = torch.cat([embeds, flair], dim=2)
            else:
                embeds = flair

        # Pass through linear layer
        if self.shrink_embeds:
            # [batch, seq_len, embed_size]
            embeds = self.reduce_layer(embeds)

        # Add embedding of capital letter
        if self.cap_features:
            embeds = torch.cat([embeds, self.cap_embedding(cap_indices)], dim=2)

        # Apply dropout
        embeds = self.embed_dropout(embeds)

        # Apply word dropout
        embeds = self.word_dropout(embeds)

        return embeds

