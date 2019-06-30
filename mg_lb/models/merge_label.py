from mg_lb.models.layers.general import get_kernel_sizes, get_cl_levels
from mg_lb.models.layers.embeds import embeds_layer
from mg_lb.models.layers.mg_lb_main_layers import *


class merge_label(nn.Module):

    def __init__(self, args, vocab):
        super(merge_label, self).__init__()

        self.network_size = args.network_size
        self.gpu = args.gpu
        self.article_theme_size = args.article_theme_size
        self.embed_type = args.embeddings
        self.cap_features = args.cap_features
        self.article_theme = args.article_theme

        # Embedding layer
        self.embedding = embeds_layer(vocab, args.embeddings, args.embed_size, args.embed_dropout, args.word_dropout,
                                      args.finetune_embeds, args.summ_activation_fn, args.init_std, args.gpu,
                                      args.cap_features, args.cap_features_size)

        self.embed_size = self.embedding.total_size

        # Encoder
        self.encoder = network_encoder(args.network_size, self.embed_size, args.update_size,args.summ_hidden_nodes,
                                       args.summ_activation_fn, args.layer_hidden_nodes, args.layer_activation_fn,
                                       args.kernel_hidden_nodes, args.dropout, args.embed_dropout, args.init_type,
                                       args.init_std, args.expand_ratio, args.max_kernel_size, args.reuse_weights,
                                       args.update_type, args.gpu, args.article_theme_size, args.num_first_layers,
                                       args.first_kernel_size, args.layer_list, args.cl_embed_size, args.replicate,
                                       args.repl_std, args.linear_update, args.layer_norm, args.every_layer_weights,
                                       args.input_num, args.article_theme)

        # Decoder
        self.decoder = network_decoder(args.num_ent_types, self.embed_size, args.decoder_hidden_nodes,
                                       args.decoder_activation_fn, args.dropout, args.init_type, args.init_std)

    def forward(self, tensors):

        # Pass on article tensor from previous batch, if still in same article
        if tensors['new_article']:
            self.article_previous = None
            self.article_weights_previous = None

        # Embedding layer
        embed_args = {'sentences': tensors['sentences']}

        if self.embedding.flair:
            embed_args['flair'] = tensors['flair']

        if self.cap_features:
            embed_args['cap_indices'] = tensors['cap_indices']

        embeds = self.embedding(**embed_args)

        # Encoder
        # embeds: [batch, seq_len, embed_size, num_cluster_levels]
        # weights: [batch, seq_len, num_cluster_levels, 2]
        # f_weights: [batch, seq_len, num_cluster_levels, 2]
        # links_pairs: [batch, seq_len, network_size, 2]
        # links: [batch, seq_len, kernel_size, network_size]
        # nw_links: [batch, seq_len-1, network_size]
        # dir_weights: [batch, seq_len, kernel_size, num_cluster_levels]
        embeds, weights, f_weights, links_pairs, links, nw_links, dir_weights, article_theme, article_w = \
            self.encoder(embeds, tensors['mask'], tensors['kmasks'], tensors['mixed_article'], self.article_previous,
                         self.article_weights_previous)

        # [batch, seq_len, num_cluster_levels, embed_size]
        embeds = embeds.transpose(2, 3)

        preds = {}

        # Decoder
        # predictions: [batch, seq_len, num_cluster_levels, num_ent_types]
        predictions = self.decoder(embeds, tensors['mask'])

        # Store outputs
        preds['weights'] = weights
        preds['ent_preds'] = predictions

        if not self.training:
            preds['links'] = links_pairs.data
            preds['final_weights'] = f_weights.data
            preds['embeds'] = embeds.data

        if self.article_theme:
            self.article_previous = article_theme.data
            self.article_weights_previous = article_w.data

        return preds


# Encoder #######################################################################


class network_encoder(nn.Module):
    def __init__(self, network_size, embed_size, update_size, summ_hidden_nodes, summ_activation_fn, layer_hidden_nodes,
                 layer_activation_fn, kernel_hidden_nodes, dropout, embed_dropout, init_type, init_std, expand_ratio,
                 max_kernel_size, reuse_weights, update_type, gpu, article_theme_size, num_first_layers, first_kernel,
                 layer_list, cl_embed_size, replicate, repl_std, linear_update, layer_norm, every_layer_weights,
                 input_num, article_theme):
        super(network_encoder, self).__init__()

        self.num_cluster_levels = get_cl_levels(layer_list)
        self.reuse_weights = reuse_weights
        self.article_theme = article_theme_size
        self.num_first_layers = num_first_layers
        self.gpu = gpu
        self.layer_list = layer_list
        self.first_kernel = first_kernel
        self.every_layer_weights = every_layer_weights
        self.article_theme = article_theme

        # Kernel sizes
        self.kernel_sizes = get_kernel_sizes(expand_ratio, len(layer_list), max_kernel_size)

        # Initial layers
        static_layers = torch.nn.ModuleList()

        for l in range(num_first_layers):
            static_layers.append(static_layer(embed_size, first_kernel, summ_hidden_nodes,
                                            dropout, summ_activation_fn, init_type, init_std, update_type,
                                            replicate, repl_std, gpu=gpu, layer_norm=layer_norm))

        self.static_layers = torch.nn.Sequential(*static_layers)

        # Article Theme layer
        if self.article_theme:
            self.article_layer = update_article(embed_size, article_theme_size, dropout, init_type, init_std)

        # Update layers
        update_layers = torch.nn.ModuleList()

        if reuse_weights:
            l = 1
        else:
            l = len(layer_list)

        for _ in range(l):
            update_layers.append(update_layer(network_size, embed_size, update_size, self.num_cluster_levels,
                                              article_theme_size, summ_hidden_nodes, summ_activation_fn,
                                              layer_hidden_nodes, layer_activation_fn, kernel_hidden_nodes, dropout,
                                              init_type, init_std, update_type, gpu, cl_embed_size, replicate, repl_std,
                                              linear_update, layer_norm, input_num=input_num, article_theme=article_theme))

        self.update_layers = torch.nn.Sequential(*update_layers)

        # Final structure layer
        if not reuse_weights:
            self.structure_layer = structure_layer(embed_size, network_size, self.num_cluster_levels,
                                                   layer_hidden_nodes=layer_hidden_nodes, layer_activation_fn=layer_activation_fn,
                                                   kernel_hidden_nodes=kernel_hidden_nodes, summ_activation_fn=summ_activation_fn,
                                                   dropout=dropout, init_type=init_type, init_std=init_std, gpu=gpu, update_type=update_type,
                                                   replicate=replicate, repl_std=repl_std, linear_update=linear_update,
                                                   layer_norm=layer_norm, input_num=input_num)

        # Embed dropout
        self.embed_dropout = nn.Dropout(embed_dropout)

        # Indices for getting the weighting on previous and next word
        self.weights_ixs = {}
        for k in self.kernel_sizes:
            half = int(k / 2)
            self.weights_ixs[k] = {
                'before': half - 1,
                'after': half + 1
            }

    def add_weights(self, ws, weights, weightings):
        """
        ws: [batch, seq_len, num_cluster_levels, 2]
        weights: [batch, seq_len, num_cluster_levels, 2]
        """
        # [batch, seq_len, num_cluster_levels, 2]
        weights[:, :, :ws.size(2), :] += ws

        # Adjust the weightings to get correct division at end
        weightings[:ws.size(2)] += 1.0

        return weights, weightings

    def extract_links(self, ls):
        """
        Extract the links from adjacent pairs of words only
        Args
        ls: [batch, seq_len, kernel_size, network_size/num_layers]
        """
        kernel_size = ls.size(2)

        bf = self.weights_ixs[kernel_size]['before']

        # [batch, seq_len, 2, network_size/num_layers]
        joined = ls[:, :, bf:bf+2, :]

        # [batch, seq_len, network_size/num_layers, 2]
        joined = joined.transpose(2, 3)

        return joined

    def forward(self, embeds, mask, kmasks, mixed_article, article_previous, article_weights_previous):

        # Static layers. [batch, seq_len, embed_size]
        for s_layer in self.static_layers:
            embeds = s_layer(embeds, mask, kmasks[self.first_kernel])

        # Generate article theme
        if self.article_theme:
            article_theme, article_w = self.article_layer(embeds, article_previous,
                                                          article_weights_previous, mask, mixed_article)
        else:
            article_theme, article_w = None, None

        # Initialize tensors to store merge values, M, from each layer
        if self.gpu:
            # [batch, seq_len, num_cluster_levels, 2]
            weights = torch.cuda.FloatTensor(embeds.size(0), embeds.size(1), self.num_cluster_levels, 2).fill_(0)
            weight_weightings = torch.cuda.FloatTensor(self.num_cluster_levels).fill_(0)

        else:
            weights = torch.zeros([embeds.size(0), embeds.size(1), len(self.layer_list), 2])
            weight_weightings = torch.zeros([self.num_cluster_levels]).float()

        # Main network layers
        for ix, l in enumerate(self.layer_list):

            if self.reuse_weights:
                i = 0
            else:
                i = ix

            ksize = self.kernel_sizes[ix]

            # Update the word vecs based on the new connections
            # l_weights: [batch, seq_len, kernel_size, num_layers]
            embeds, l_weights = self.update_layers[i](embeds, article_theme, kmasks[ksize], ksize, l, mask)

            if self.every_layer_weights:
                # [batch, seq_len, num_layers, 2]
                f_weights = self.extract_links(l_weights)

                # [batch, seq_len, num_cluster_levels, 2]
                weights, weight_weightings = self.add_weights(f_weights, weights, weight_weightings)

        # Final Structure layer
        # embeds: [batch, seq_len, embed_size, num_cluster_levels]
        # l_weights: [batch, seq_len, kernel_size, num_cluster_levels]
        # links: [batch, seq_len, kernel_size, network_size]
        # nw_links: [batch, seq_len-1, network_size]
        # dir_weights: [batch, seq_len, kernel_size, num_cluster_levels]
        ksize = self.kernel_sizes[-1]

        if self.reuse_weights:
            final_s_layer = self.update_layers[i]
        else:
            final_s_layer = self.structure_layer

        embeds, l_weights, links, nw_links, dir_weights = final_s_layer(embeds, ksize, kmasks[ksize],
                                                                        self.num_cluster_levels+1, mask, final=True)

        # Extract links from adjacent pairs of words. [batch, seq_len, network_size, 2]
        links_pairs = self.extract_links(links)

        # Extract merge values from adjacent pairs of words [batch, seq_len, num_cluster_levels, 2]
        final_weights = self.extract_links(l_weights)

        # Store the merge values from each layer
        if self.every_layer_weights:
            # [batch, seq_len, num_cluster_levels, 2]
            weights, weight_weightings = self.add_weights(final_weights, weights, weight_weightings)

            # Normalize
            weights = weights / weight_weightings.view(1, 1, self.num_cluster_levels, 1)

        else:
            weights = final_weights

        return embeds, weights, final_weights, links_pairs, links, nw_links, dir_weights, article_theme, article_w

