
def str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise Exception('Arg should be bool')


def merge_label_args(p):
    p.add_argument('--problems', default=['ACE05'], nargs='+', type=str)
    p.add_argument('--ratios', default=[1], nargs='+', type=int)
    p.add_argument('--every_layer_weights', default=True, type=str2bool)
    p.add_argument('--all_splits', default=False, type=str2bool)
    p.add_argument('--cl_loss_mse', default=False, type=str2bool)
    p.add_argument('--add_upper', default=False, type=str2bool)
    p.add_argument('--coref_weight', default=1.0, type=float)
    p.add_argument('--cluster_weight', default=1.0, type=float)
    p.add_argument('--layer_list', default=[2, 3, 4], type=int, nargs='+')
    p.add_argument('--cl_embed_size', default=100, type=int)
    p.add_argument('--replicate', default=True, type=str2bool)
    p.add_argument('--repl_std', default=0.01, type=float)
    p.add_argument('--linear_update', default=False, type=str2bool)
    p.add_argument('--cutoff', default=0.25, type=float)

    p.add_argument('--np_weight', default=1.0, type=float)
    p.add_argument('--o_weight', default=1.0, type=float)

    p.add_argument('--network_size', default=200, type=int)
    p.add_argument('--update_size', default=100, type=int)
    p.add_argument('--update_type', default='none', type=str)

    p.add_argument('--article_theme_size', default=50, type=int)
    p.add_argument('--article_theme', default=True, type=str2bool)

    p.add_argument('--num_first_layers', default=1, type=int)
    p.add_argument('--first_kernel_size', default=6, type=int)
    p.add_argument('--expand_ratio', default=5, type=int)
    p.add_argument('--max_kernel_size', default=30, type=int)

    p.add_argument('--summ_hidden_nodes', default=[320, 320], type=int, nargs='+')
    p.add_argument('--summ_activation_fn', default='swish', type=str)

    p.add_argument('--kernel_hidden_nodes', default=[320], type=int, nargs='+')

    p.add_argument('--layer_hidden_nodes', default=[200, 200], type=int, nargs='+')
    p.add_argument('--layer_activation_fn', default='swish', type=str)
    p.add_argument('--input_num', default=4, type=int)

    p.add_argument('--decoder_hidden_nodes', default=[200], type=int, nargs='+')
    p.add_argument('--decoder_activation_fn', default='selu', type=str)

    p.add_argument('--links_decoder', default=False, type=str2bool)
    p.add_argument('--reuse_weights', default=False, type=str2bool)

    p.add_argument('--init_std', default=0.1, type=float)
    p.add_argument('--init_type', default='uniform', type=str)
    p.add_argument('--layer_norm', default=False, type=str2bool)

    p.add_argument('--embed_dropout', default=0.2, type=float)
    p.add_argument('--word_dropout', default=0.0, type=float)

    p.add_argument('--batch_size', default=1000, type=int)
    p.add_argument('--one_article', default=False, type=str2bool)
    p.add_argument('--dropout', default=0.1, type=float)
    p.add_argument('--num_epochs', default=300, type=int)
    p.add_argument('--gpu', default=True, type=str2bool)
    p.add_argument('--learning_rate', default=0.0005, type=float)
    p.add_argument('--learning_rate_decay', default='half_every_thirty', type=str)
    p.add_argument('--lr_warmup', default=300, type=int)
    p.add_argument('--max_grad_norm', default=1.0, type=float)
    p.add_argument('--viz_every', default='5', type=str)
    p.add_argument('--embed_size', default=300, type=int)
    p.add_argument('--embeddings', type=str, nargs='+', default=['glove'])
    p.add_argument('--cap_features', type=str2bool, default=False)
    p.add_argument('--cap_features_size', type=int, default=20)
    p.add_argument('--finetune_embeds', default=False, type=str2bool)
    p.add_argument('--loss_fn', default=['named_ent_loss'], nargs='+', type=str)
    p.add_argument('--loss_weights', default=[1.0], nargs='+', type=float)
    p.add_argument('--optimizer', default='Adam')
    p.add_argument('--use_accuracy', default=False, type=str2bool, nargs='+')
    p.add_argument('--acc_key', default='f1', type=str)
    p.add_argument('--bucket', default=True, type=str2bool)
    p.add_argument('--split_sentences', default=False, type=str2bool)
    p.add_argument('--shuffle_batches', default=True, type=str2bool)
    p.add_argument('--save', default=True, type=str2bool)
    p.add_argument('--save_id', default='1')

    # Set the model attribute so we can access which model we're using in python
    p.set_defaults(model='merge_label')
    return p



