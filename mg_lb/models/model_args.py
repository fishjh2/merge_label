import mg_lb.data_loading.prep_tensors as pts


model_args = {
    'merge_label': {
        'type': 'shape',
        'kernel': True,
        'prep_tensors': pts.prep_network_tensors
    }
}


