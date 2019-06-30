import pickle


problems = {
    'ontonotes': {
        'save_preprocessed': False,
        'flair_splits': 16,
        'ent_types': 43,
        'preload': True,
        'fs_per_iter': 1,
        'article': True,
        'accuracy': {'key': 'pred_first', 'process': 'max_forty'},
        'save_tensors': ['ent_preds', 'network', 'distances', 'weights', 'all_levels',
                          'links', 'final_weights', 'embeds', 'pred_first', 'acc_target',
                          'acc_weights', 'f1_target', 'ch_mask', 'pred_mask', 'cl_mask', 'article_theme']

    },
    'ACE05': {
        'save_preprocessed': False,
        'flair_splits': 4,
        'ent_types': 15,
        'preload': True,
        'fs_per_iter': 1,
        'article': True,
        'accuracy': {'key': 'pred_first', 'process': 'max_first'},
        'save_tensors': ['ent_preds', 'network', 'distances', 'weights', 'all_levels', 'links', 'final_weights',
                         'embeds', 'pred_first', 'acc_target', 'acc_weights', 'f1_target']

    },
    'ontonotes_test': {
        'save_preprocessed': False,
        'flair_splits': 2,
        'ent_types': 43,
        'preload': True,
        'fs_per_iter': 1,
        'article': True,
        'accuracy': {'key': 'pred_first', 'process': 'max_forty'},
        'save_tensors': ['ent_preds', 'network', 'distances', 'weights', 'all_levels',
                       'links', 'final_weights', 'embeds', 'pred_first', 'acc_target',
                       'acc_weights', 'f1_target', 'ch_mask', 'pred_mask', 'cl_mask', 'article_theme'],
    }
}

labels_lookup = {
    'ontonotes': pickle.load(open('./mg_lb/problems/labels/ontonotes_lookup.p', 'rb')),
    'ontonotes_test': pickle.load(open('./mg_lb/problems/labels/ontonotes_lookup.p', 'rb')),
    'ACE05': pickle.load(open('./mg_lb/problems/labels/ACE05_lookup.p', 'rb'))
}

