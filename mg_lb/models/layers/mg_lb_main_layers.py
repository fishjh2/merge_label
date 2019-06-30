from mg_lb.models.layers.mg_lb_sub_layers import *
from mg_lb.models.layers.initialization import pytorch_initialize


class static_layer(nn.Module):

    def __init__(self, embed_size, kernel_size, hidden_nodes, dropout,
                 activation_fn, init_type, init_std, update_type, replicate, repl_std, gpu, layer_norm):
        super(static_layer, self).__init__()

        self.kernel_size = kernel_size

        # Defunct layer, not used in network, but left in to avoid errors loading previously trained models
        self.embed_linear = linear_layer(embed_size, embed_size, dropout, activation_fn,
                                         w_init=(init_type, init_std), b_init=('constant', 0.1))

        # Learnable positional embeddings. Shape [1, max_sent_len, embed_size]
        self.position_embeds = nn.Parameter(torch.Tensor(1, 1, kernel_size, embed_size))
        pytorch_initialize(self.position_embeds, ('uniform', 0.1))

        self.embed_update = embed_update(embed_size * 2, embed_size, hidden_nodes, activation_fn, dropout, init_type, init_std,
                                         update_type, layer_norm, replicate, repl_std=repl_std, repl_size=embed_size, gpu=gpu)

    def forward(self, embeds, mask, kmask):
        """
        embeds: [batch, seq_len, embed_size]
        mask: [batch, seq_len, 1]
        kmask: [batch, seq_len, kernel_size+1, 1]
        """
        # expanded: [batch, seq_len, kernel_size, update_size]
        to_embeds = to_embeds_pairs(embeds, mask, self.kernel_size, keep_own=False)

        # Add positional embedding. [batch, seq_len, kernel_size, update_size]
        to_embeds = to_embeds + self.position_embeds

        # [batch, seq_len, kernel_size, update_size]
        from_embeds = from_embeds_pairs(embeds, mask, self.kernel_size)

        # Concat to [batch, seq_len, kernel_size, update_size*2]
        net_in = torch.cat([from_embeds, to_embeds], dim=3)

        # [batch, seq_len, kernel_size, 1]
        kmask = remove_from_center(kmask, self.kernel_size)

        # Update the embeds. [batch, seq_len, embed_size]
        new_embeds = self.embed_update(net_in, embeds, kmask)

        return new_embeds


class update_layer(nn.Module):

    def __init__(self, network_size, embed_size, update_size, num_cluster_levels, article_theme_size, hidden_nodes, activation_fn,
                 layer_hidden_nodes, layer_activation_fn, kernel_hidden_nodes,
                 dropout, init_type, init_std, update_type,
                 gpu, cl_embed_size, replicate, repl_std, linear_update, layer_norm, input_num, article_theme):
        super(update_layer, self).__init__()

        self.article_theme = article_theme

        if not article_theme:
            article_theme_size = 0

        self.embed_linear = linear_layer(embed_size, update_size, dropout, activation_fn,
                                         w_init=(init_type, init_std), b_init=('constant', 0.1))

        self.structure_layer = structure_layer(embed_size, network_size, num_cluster_levels,
                                               layer_hidden_nodes=layer_hidden_nodes,
                                               layer_activation_fn=layer_activation_fn, kernel_hidden_nodes=kernel_hidden_nodes,
                                               summ_activation_fn=activation_fn, dropout=dropout, init_type=init_type,
                                               init_std=init_std, gpu=gpu, update_type=update_type, replicate=replicate,
                                               repl_std=repl_std, linear_update=linear_update,
                                               layer_norm=layer_norm, input_num=input_num)

        self.embed_update = embed_update((network_size + embed_size * 2 + article_theme_size + cl_embed_size),
                                         embed_size, hidden_nodes, activation_fn, dropout, init_type, init_std,
                                         update_type=update_type, layer_norm=layer_norm, replicate=replicate, repl_std=repl_std,
                                         repl_size=embed_size, gpu=gpu)

        # Cluster level embeddings. Shape [1, max_sent_len, embed_size]
        self.level_embeds = nn.Parameter(torch.Tensor(1, 1, 1, num_cluster_levels+1, cl_embed_size))
        pytorch_initialize(self.level_embeds, ('uniform', 0.1))

    def forward(self, embeds, article_theme, kmask, kernel_size, num_layers, mask):
        """
        Args
        embeds: [batch, seq_len, embed_size]
        article_theme: [batch, article_theme_size]
        kmask: [batch, seq_len, kernel_size+1, 1]
        kernel_size: int
        num_layers: int
        mask: seq_len mask. Shape [batch, seq_len, 1]
        """

        # Accumulate ORIGINAL embeds.
        # to_embeds: [batch, seq_len, kernel_size, embed_size]
        # network: [batch, seq_len, kernel_size, network_size]
        # m: [batch, seq_len, kernel_size, 1]
        # ret_weights: [batch, seq_len, kernel_size, num_layers]
        # cl_weights: [batch, seq_len, kernel_size, 1]
        # level_weights: [batch, seq_len, kernel_size, num_layers]
        to_embeds, network, m, ret_weights, cl_weights, level_weights = self.structure_layer(embeds, kernel_size, kmask,
                                                                                             num_layers, mask, final=False)

        # [batch, seq_len, kernel_size, network_size]
        from_embeds = from_embeds_pairs(embeds, mask, kernel_size)

        # [batch, seq_len, kernel_size, num_layers, level_embed_size]
        level_embeds = self.level_embeds[:, :, :, :num_layers, :] * level_weights.unsqueeze(4)

        # [batch, seq_len, kernel_size, level_embed_size]
        level_embeds = torch.sum(level_embeds, dim=3)

        if self.article_theme:
            # Tile article theme vector to size [batch, seq_len, kernel_size, article_theme_size]
            art_theme = article_theme.view([article_theme.size(0), 1, 1, article_theme.size(1)])
            art_theme = art_theme.expand(art_theme.size(0), network.size(1), network.size(2), art_theme.size(3))

            # [batch, seq_len, kernel_size, network_size * 3]
            net_in = torch.cat([from_embeds, to_embeds, network, level_embeds, art_theme], dim=3)

        else:
            net_in = torch.cat([from_embeds, to_embeds, network, level_embeds], dim=3)

        # Total of level weights across all layers. [batch, seq_len, kernel_size]
        total_level = torch.sum(level_weights, dim=3)

        # [batch, seq_len, embed_size]
        embeds_new = self.embed_update(net_in, embeds, m, cl_weights,
                                       total_level=total_level)

        return embeds_new, ret_weights


class structure_layer(nn.Module):

    def __init__(self, embed_size, network_size, num_cluster_levels,
                 layer_hidden_nodes, layer_activation_fn,
                 kernel_hidden_nodes, summ_activation_fn,
                 dropout, init_type, init_std, gpu, update_type, replicate, repl_std,
                 linear_update, layer_norm, input_num):
        super(structure_layer, self).__init__()

        self.num_cluster_levels = num_cluster_levels
        self.gpu = gpu

        self.links_layer = network_layer(embed_size, network_size, layer_hidden_nodes, layer_activation_fn,
                                         dropout=dropout, init_std=init_std, init_type=init_type, gpu=gpu,
                                         input_num=input_num)

        self.layer = cumulative_layer(embed_size, network_size, kernel_hidden_nodes, summ_activation_fn, dropout, init_type,
                                      init_std, replicate, repl_std, linear_update, layer_norm, gpu=gpu)

    def forward(self, embeds, kernel_size, kmask, num_layers, mask, final):
        """
        embeds: shape [batch, seq_len, embed_size]
        kernel_size: int
        kmask: [batch, seq_len, kernel_size+1, 1]
        num_layers: int
        mask: seq_len mask. Shape [batch, seq_len, 1]
        final: bool
        """

        # [batch, seq_len, kernel_size, 1]
        in_kmask = remove_from_center(kmask, kernel_size)

        ret_weights = []
        ret_embeds = []
        level_weights = []
        dir_weights = []

        prev_distances = 1.0
        embeds_saved = 0.0
        links_saved = 0.0
        weights_prev = 0.0
        cl_weights_saved = 0.0

        if self.gpu:
            # [batch, seq_len, 1]
            cluster_weights = torch.cuda.FloatTensor(embeds.size(0), embeds.size(1), 1).fill_(1.0)
        else:
            cluster_weights = torch.ones([embeds.size(0), embeds.size(1), 1])

        # Layers
        for l in range(num_layers):

            # Calculate the links.
            # links: [batch, seq_len-1, network_size]
            # distances: [batch, seq_len-1, 1]
            links, distances = self.links_layer(embeds, mask)

            # Multiply
            distances = distances * prev_distances
            links = links * prev_distances
            prev_distances = distances

            # Unfold cluster_weights. [batch, seq_len, kernel_size+1, 1]
            cluster_weights_old = min_expand(cluster_weights, kernel_size)

            # Links and weights
            # ls [batch, seq_len, kernel_size+1, network_size]
            # weights [batch, seq_len, kernel_size+1]
            ls, weights = get_relationships(links, distances, kernel_size)

            # Update the embeds.
            # embeds: shape [batch, seq_len, embed_size]
            # exp_embeds: shape [batch, seq_len, kernel_size, embed_size]
            # cluster_weights: [batch, seq_len, 1]
            if l < self.num_cluster_levels:
                embeds, weights, exp_embeds, cluster_weights = self.layer(embeds, kernel_size, weights, mask,
                                                                          cluster_weights_old, kmask, in_kmask, ls)

                # [batch, seq_len, kernel_size+1]
                w_unsq = (weights - weights_prev)

            else:
                # [batch, seq_len, kernel_size+1]
                w_unsq = (1.0 - weights_prev)

                # Expand final layer accumulated embeds. [batch, seq_len, kernel_size+1, embed_size]
                exp_embeds = to_embeds_pairs(embeds, mask, kernel_size, keep_own=True)

            # [batch, seq_len, kernel_size+1, 1]
            w_unsq = w_unsq.unsqueeze(3)

            # [batch, seq_len, kernel_size+1, embed_size]
            embeds_saved += (w_unsq * exp_embeds)

            # [batch, seq_len, kernel_size+1, network_size]
            links_saved += (w_unsq * ls)

            cl_weights_saved += (w_unsq * cluster_weights_old)

            # [batch, seq_len, kernel_size+1, 1]
            level_weights.append(w_unsq)

            weights_prev = weights

            # Weights each word puts on next word. [batch, seq_len, kernel_size+1, 1]
            ret_weights.append(weights.unsqueeze(3))

            # Embeds [batch, seq_len, embed_size, 1]
            ret_embeds.append(embeds.unsqueeze(3))

            # Weights for the prediction from directions.
            # [batch, seq_len, kernel_size+1, 1]
            dir_weights.append((1.0 - weights).unsqueeze(3) * kmask)

        # [batch, seq_len, kernel_size, num_cluster_levels]
        # Ignore last set of weights as don't use them in network
        ret_weights = torch.cat(ret_weights[:self.num_cluster_levels], dim=3)
        ret_weights = remove_from_center(ret_weights, kernel_size)

        # [batch, seq_len, kernel_size, network_size]
        links_saved = remove_from_center(links_saved, kernel_size)

        if final:

            # Join into one tensor of shape [batch, seq_len, embed_size, num_cluster_levels]
            ret_embeds = torch.cat(ret_embeds[:self.num_cluster_levels], dim=3)

            # [batch, seq_len, kernel_size, num_cluster_levels]
            dir_weights = torch.cat(dir_weights[:self.num_cluster_levels], dim=3)
            dir_weights = remove_from_center(dir_weights, kernel_size)

            return ret_embeds, ret_weights, links_saved, links, dir_weights

        else:
            # Remove own embedding from center. [batch, seq_len, kernel_size, embed_size]
            embeds_saved = remove_from_center(embeds_saved, kernel_size)

            # [batch, seq_len, kernel_size, 1]
            cl_weights_saved = remove_from_center(cl_weights_saved, kernel_size)

            # [batch, seq_len, num_cluster_levels+1, kernel_size]
            level_weights = remove_from_center(torch.cat(level_weights, dim=3), kernel_size)

            return embeds_saved, links_saved, in_kmask, ret_weights, cl_weights_saved, level_weights



