import torch.nn as nn
import torch
import torch.nn.functional as F

from mg_lb.models.layers.linear import NN, linear_layer


class cumulative_layer(nn.Module):

    def __init__(self, embed_size, network_size, hidden_nodes, activation_fn, dropout, init_type, init_std, replicate,
                 repl_std, linear_update, layer_norm, gpu):
        super(cumulative_layer, self).__init__()

        self.linear_update = linear_update

        if linear_update:
            self.embed_linear = linear_layer(embed_size, 1, dropout, activation_fn,
                                             w_init=(init_type, init_std), b_init=('constant', 0.1))

        else:
            self.embed_update = embed_update((embed_size + network_size), embed_size + 1, hidden_nodes, activation_fn,
                                             dropout, init_type, init_std, gpu=gpu, update_type='none', layer_norm=layer_norm,
                                             replicate=replicate, repl_size=embed_size, repl_std=repl_std)

    def forward(self, embeds, kernel_size, weights, mask, cluster_weights, kmask, in_kmask, links):
        """
        embeds: shape [batch, seq_len, embed_size]
        kernel_size: int
        weights: shape [batch, seq_len, kernel_size+1]
        mask: seq_len mask. Shape [batch, seq_len, 1]
        cluster_weights: [batch, seq_len, kernel_size+1, 1]
        kmask: [batch, seq_len, kernel_size+1, 1]
        in_kmask: [batch, seq_len, kernel_size, 1]
        links: [batch, seq_len, kernel_size+1, network_size]
        """

        if self.linear_update:
            # Weighting layer. [batch, seq_len, 1]
            w = torch.sigmoid(self.embed_linear(embeds))

            # [batch, seq_len, embed_size+1]
            new = torch.cat([embeds, w], dim=2)

        else:
            # [batch, seq_len, kernel_size, network_size]
            from_embeds = from_embeds_pairs(embeds, mask, kernel_size)

            # [batch, seq_len, kernel_size, network_size]
            links = remove_from_center(links, kernel_size)

            # [batch, seq_len, kernel_size, embed_size+network_size]
            net_in = torch.cat([from_embeds, links], dim=3)

            # [batch, seq_len, embed_size+1]
            new = self.embed_update(net_in, embeds, in_kmask)

            # [batch, seq_len, embed_size+1]
            new[:, :, -1] = torch.sigmoid(new[:, :, -1])

        # expanded: [batch, seq_len, kernel_size+1, embed_size + 1]
        # m: [batch, seq_len, kernel_size+1, 1]
        expanded = to_embeds_pairs(new, mask, kernel_size, keep_own=True)

        # Weights for how much each words embedding contributes to the joint embedding
        # [batch, seq_len, kernel_size+1]
        w = expanded[:, :, :, -1]

        # Mask out for seq_len and padding. [batch, seq_len, kernel_size+1]
        weights = weights * kmask.squeeze(3)

        # Cluster weights - these are used to downsccale importance put on large cluster automatically
        # [batch, seq_len, 1]
        cluster_weights_new = torch.sum(weights, dim=2).unsqueeze(2)

        # Multiply by cluster weights
        # weights_new = weights / (cluster_weights.squeeze(3) + 0.0000001)

        # Weighted average of the weights. [batch, seq_len, kernel_size+1]
        top = (weights * w) + 0.0000001
        weighted_weights = top / torch.sum(top, dim=2, keepdim=True)

        # [batch, seq_len, kernel_size+1, 1]
        weighted_weights = weighted_weights.unsqueeze(3)

        # Shape [batch, seq_len, kernel_size+1, embed_size]
        expanded = expanded[:, :, :, :-1]
        new_embeds = weighted_weights * expanded

        # Sum across kernel_size dim. [batch, seq_len, embed_size]
        ret_embeds = torch.sum(new_embeds, dim=2) * mask

        return ret_embeds, weights, expanded, cluster_weights_new


class update_article(nn.Module):

    def __init__(self, embed_size, article_size, dropout, init_type, init_std):
        super(update_article, self).__init__()

        self.article_layer = linear_layer(embed_size, article_size + 1, dropout, activation_fn=None,
                                          w_init=(init_type, init_std), b_init=('constant', 0.1))

    def forward(self, embeds, previous, previous_weight, mask, mixed_article):
        """
        embeds: [batch, seq_len, embed_size]
        previous: [batch, article_size]
        previous_weight: [1]
        mask: [batch, seq_len, 1]
        mixed_article: bool
        """
        # Shape [batch, seq_len, article_size + 1]
        contribution = self.article_layer(embeds)

        # Mask out-of-sequence length words
        contribution = contribution * mask

        # Weight. Shape [batch, seq_len, article_size]
        weights = torch.relu(contribution[:, :, -1].unsqueeze(2))
        contribution = contribution[:, :, :-1] * weights

        if mixed_article:
            # Sum across seq_len dim. [batch, article_size]
            ow = torch.sum(weights, dim=1) + 0.00001
            article = (torch.sum(contribution, dim=1) + 0.00000001) / ow
        else:
            # Average into a single tensor. Shape [article_size]
            ow = torch.sum(weights) + 0.00001
            article = (torch.sum(torch.sum(contribution, dim=0), dim=0) + 0.000001) / ow

            # Expand to [batch_size, article_size]
            article = article.unsqueeze(0)
            article = article.expand(embeds.size(0), article.size(1))

            if previous is not None:
                # Weighted average with previous article tensor from same layer in different batch
                article = ((article * ow) + (previous * previous_weight)) / (ow + previous_weight)

        return article, ow


class network_layer(nn.Module):

    def __init__(self, embed_size, network_size, hidden_nodes, activation_fn, dropout, init_type,
                 init_std, gpu, input_num):
        super(network_layer, self).__init__()

        self.network_size = network_size
        self.input_num = input_num

        self.gpu = gpu

        self.ff = NN(features_dim=(embed_size * input_num), targets_dim=(network_size + 1),
                     num_hidden_nodes=hidden_nodes, num_layers=len(hidden_nodes),
                     activation_fn=activation_fn, w_init=(init_type, init_std),
                     b_init=('constant', init_std), f_w_init=(init_type, init_std),
                     f_b_init=('constant', 0), dropout=dropout)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, embeds, mask):
        """
        embeds: [batch, seq_len, embed_size]
        mask: [batch, seq_len, 1]
        """
        if self.input_num <= 2:

            # [batch, seq_len-1, embed_size]
            l = embeds[:, :-1, :]
            r = embeds[:, 1:, :]

            # [batch, seq_len-1, embed_size*2]
            res = torch.cat([l, r], dim=2)

        else:
            # [batch, seq_len-1, 4, embed_size]
            exp_es = to_embeds_pairs(embeds, mask, self.input_num, keep_own=True)[:, :-1, 1:, :]

            # [batch, seq_len-1, 4 * embed_size]
            res = exp_es.view(embeds.size(0), exp_es.size(1), -1)

        # Apply same feedforward NN to each pair of summaries
        # Shape [batch, seq_len-1, (network_size + 1)]
        out = self.ff(res)['main']

        # [batch, seq_len-1, 1]
        distances = torch.sigmoid(out[:, :, -1]).unsqueeze(2)

        # [batch, seq_len-1, network_size]
        links = out[:, :, :-1]

        return links, distances


class embed_update(nn.Module):

    def __init__(self, in_size, out_size, hidden_nodes, activation_fn, dropout, init_type, init_std,
                 update_type, layer_norm, replicate=False, repl_size='none', repl_std=0.01, gpu=False):
        super(embed_update, self).__init__()

        self.update_type = update_type
        self.layer_norm = layer_norm

        if replicate:
            w_init = ('repr', repl_size, repl_std)
            dropout = 0.0001
        else:
            w_init = (init_type, init_std)

        self.net = NN(features_dim=in_size, targets_dim=(out_size + 1),
                      num_hidden_nodes=hidden_nodes, num_layers=len(hidden_nodes),
                      activation_fn=activation_fn, w_init=w_init,
                      b_init=('constant', init_std), f_w_init=w_init, f_b_init=('constant', 0),
                      dropout=dropout)

        if layer_norm:
            self.lnorm = nn.LayerNorm(out_size)

    def forward(self, net_in, embeds, m, cl_weights=None, level_weights=None,
                total_level=None):
        """
        Args
        net_in: [batch, seq_len, kernel_size, in_size]
        embeds: [batch, seq_len, embed_size]
        m: mask for seq_len and padding. [batch, seq_len, kernel_size, 1]
        cl_weights: [batch, seq_len, kernel_size, 1]
        level_weights: [batch, seq_len, kernel_size, num_layers]
        total_level: [batch, seq_len, kernel_size]
        """

        # Pass through network. Size [batch, seq_len, kernel_size, embed_size + 1]
        out = self.net(net_in)['main']

        # Sigmoid the last output - this is a weighting of the link. Add small float to
        # avoid divide by zero error below
        out[:, :, :, -1] = torch.sigmoid(out[:, :, :, -1])

        # Mask so remove links to words that are greater than sequence length
        out = m * out

        # Shape [batch, seq_len, kernel_size].
        link_weight = out[:, :, :, -1] + 0.00000001

        # Automatically reweight outputs from levels which haven't been used
        if total_level is not None:
            link_weight = (link_weight * total_level) + 0.0000000001

        # Automatic reweighting of large clusters
        # if cl_weights is not None:
        #    link_weight = link_weight / (cl_weights.squeeze(3) + 0.0001)

        # [batch, seq_len, 1]
        norm = torch.sum(link_weight, dim=2, keepdim=True)

        # Normalize weights so they sum to one over kernel_size dim
        # [batch, seq_len, kernel_size, 1]
        weights = (link_weight / norm).unsqueeze(3)

        # Weighted average. [batch, seq_len, embed_size + 1]
        updates = torch.sum(out * weights, dim=2)

        if self.update_type == 'none':

            updates = updates[:, :, :-1]

            # Layer normalization
            if self.layer_norm:
                updates = self.lnorm(updates)
            return updates

        else:
            # Weights - shape [batch, seq_len, 1]
            upd_weights = updates[:, :, -1].unsqueeze(2)

            # Weighted average update. [batch, seq_len, embed_size]
            if self.update_type == 'residual':
                embeds_new = embeds + (upd_weights * updates[:, :, :-1])
            elif self.update_type == 'average':
                emb_weights = 2 - upd_weights
                embeds_new = ((updates[:, :, :-1] * upd_weights) + (embeds * emb_weights)) / 2

            if self.layer_norm:
                embeds_new = self.lnorm(embeds_new)

            return embeds_new


def get_relationships(links, distances, kernel_size):
    """
    Given the relationships between adjacent pairs of words, get the wider relationships
    in the network, up to a distance of kernel_size/2
    Args
    links: shape [batch, seq_len-1, network_size]
    distances: [batch, seq_len-1, 1]
    kernel_size: int, the size of the kernel
    """
    half_k = int(kernel_size / 2)

    # Scale links by distances
    # links = links * distances

    # [batch, seq_len-1, network_size+1]
    old_pairs_rel = torch.cat([links, distances], dim=2)

    # Zero pad on either end of num_adj_pairs dim
    pairs_rel = F.pad(old_pairs_rel, (0, 0, half_k, half_k), "constant", 0.0)

    # Unfold to shape [batch_size, seq_len, network_size+1, kernel_size]
    out = pairs_rel.unfold(dimension=1, size=kernel_size, step=1)

    # Split in half. Both shape [seq_len, network_size+1, kernel_size/2]
    out_prev = out[:, :, :, :half_k]
    out_after = out[:, :, :, half_k:]

    # Cumulative sum to get relationships. Shape [batch, seq_len, network_size+1, kernel_size/2]
    # For words before, flip and make negative before cumsum as want reverse relationships
    dis_prev = torch.flip(torch.cumsum(torch.flip(-out_prev, dims=[3]), dim=3), dims=[3])
    dis_after = torch.cumsum(out_after, dim=3)

    # Join a row of zeros in middle for weightings on each word's own embedding.
    # [batch, seq_len, (kernel_size/2)+1]
    dis_prev = F.pad(dis_prev, (0, 1), "constant", 0.0)

    # Rejoin into one tensor of shape [batch_size, seq_len, network_size+1, kernel_size+1]
    dis = torch.cat([dis_prev, dis_after], dim=3)

    # Tranpose to [batch, seq_len, kernel_size+1, network_size+1]
    dis = dis.transpose(2, 3)

    # Relationships. [batch, seq_len, kernel_size+1, network_size]
    rels = dis[:, :, :, :-1]

    # Weights are 1 minus the distance. [batch, seq_len, kernel_size+1]
    weights = torch.clamp(1.0 - torch.abs(dis[:, :, :, -1]), min=0.0)

    return rels, weights


def min_expand(in_tensor, kernel_size):
    """
    in_tensor: [batch, seq_len, any]
    kernel_size: int
    """
    half_k = int(kernel_size / 2)

    # Zero pad on either end
    padded = F.pad(in_tensor, (0, 0, half_k, half_k), "constant", 0.0)

    # Unfold to shape [batch_size, seq_len, any, kernel_size+1]
    out = padded.unfold(dimension=1, size=kernel_size + 1, step=1)

    # [batch, seq_len, kernel_size+1, any]
    out = out.transpose(2, 3)

    return out


def to_embeds_pairs(embeds, mask, kernel_size, keep_own=False):
    """
    Get all embeds (reduced through linear layer to network size)  for the words that the relationship is to
    Args
    embeds: shape [batch, seq_len, network_size or embed_size]
    mask: shape [batch, seq_len, 1]
    kernel_size: int, the size of the kernel
    """
    half_k = int(kernel_size / 2)

    # Mask for seq_len. [batch, seq_len, embed_size/network_size]
    embeds = embeds * mask

    # Zero pad on either end
    embeds = F.pad(embeds, (0, 0, half_k, half_k), "constant", 0.0)

    # Unfold to shape [batch_size, seq_len, embed_size/network_size, kernel_size]
    out = embeds.unfold(dimension=1, size=kernel_size + 1, step=1)

    # Delete middle row. Shape [batch, seq_len, network_size, kernel_size]
    if not keep_own:
        out = torch.cat([out[:, :, :, :half_k], out[:, :, :, -half_k:]], dim=3)

    # Reshape to [batch, seq_len, kernel_size, network_size]
    out = out.transpose(2, 3)

    return out


def from_embeds_pairs(embeds, mask, kernel_size):
    """
    Get all embeds (reduced through linear layer to network size)  for the words that the relationship is from
    Args
    embeds: shape [batch, seq_len, network_size]
    mask: shape [batch, seq_len, 1]
    kernel_size: int, the size of the kernel
    """
    # Mask out words out of sequence len
    embeds = (embeds * mask).unsqueeze(2)

    # Expand kernel_size times to [batch, seq_len, kernel_size, network_size]
    from_embeds = embeds.expand(embeds.size(0), embeds.size(1), kernel_size, embeds.size(3))

    return from_embeds


def remove_from_center(tensor, kernel_size):
    half_k = int(kernel_size / 2)
    return torch.cat([tensor[:, :, :half_k], tensor[:, :, -half_k:]], dim=2)


# Decoder ######################################################################


class network_decoder(nn.Module):
    def __init__(self, num_ent_types, embed_size, decoder_hidden_nodes, decoder_activation_fn,
                 dropout, init_type, init_std):
        super(network_decoder, self).__init__()

        self.net = decoder_net(embed_size, num_ent_types, decoder_hidden_nodes, decoder_activation_fn,
                               dropout, init_type, init_std)

    def forward(self, embeds, mask):
        """
        embeds: [batch, seq_len, num_cluster_levels, embed_size]
        mask: [batch, seq_len, 1]
        """
        # Pass through decoder feedforward net. [batch, seq_len, num_cluster_levels, num_ent_types]
        ents = self.net(embeds)

        return ents


class decoder_net(nn.Module):

    def __init__(self, embed_size, num_ent_types, hidden_nodes, activation_fn,
                 dropout, init_type, init_std):
        super(decoder_net, self).__init__()

        self.ents_net = NN(features_dim=embed_size, targets_dim=num_ent_types,
                           num_hidden_nodes=hidden_nodes, num_layers=len(hidden_nodes),
                           activation_fn=activation_fn, w_init=(init_type, init_std),
                           b_init=('constant', init_std), f_w_init=(init_type, init_std), f_b_init=('constant', 0),
                           dropout=dropout)

    def forward(self, embeds):
        """
        Args
        embeds: the word vectors. Shape [batch, seq_len, num_cluster_levels, embed_size]
        """
        # Prediction NN for ent_type. Shape [batch, seq_len, num_cluster_levels, target_dim]
        pred = self.ents_net(embeds)['main']

        return pred