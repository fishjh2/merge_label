import argparse
import shutil
import os
import pickle
from sklearn.externals import joblib
from multiprocessing import Pool

import mg_lb.training.args as m_args
from mg_lb.training.utils.fs import probs_to_losses
from mg_lb.problems.probs import problems
from mg_lb.data_loading.data_prep import prep_data
import mg_lb.models.models as models
from mg_lb.training.trainer import trainer


def main(args=None):

    if args is None:
        # Top level arg parser
        parser = argparse.ArgumentParser()

        # Allow adding of subparsers - one for each model type
        subparsers = parser.add_subparsers(help='first argument should be the model to use e.g. BIMPM')

        # Add the parsers for the different model types
        ms = [i for i in dir(m_args) if '__' not in i]
        ms = [i for i in ms if i != 'str2bool']

        for m in ms:
            p = subparsers.add_parser(m.split('_args')[0])
            p = getattr(m_args, m)(p)

        args = parser.parse_args()

    kwargs = {}

    # Match problems to the loss fns they'll be trained with
    prob_losses = probs_to_losses(args.problems, args.loss_fn, args.loss_weights)

    vocab = None

    # Prep all the datasets
    prob_iterators = {}

    # Temporarily store the preprocessed data iterators
    temp_store = './data/data_temp/' + args.save_id + '/'
    if os.path.exists(temp_store):
        shutil.rmtree(temp_store)
    os.makedirs(temp_store)

    v_load = True

    for prob, dd in prob_losses.items():

        if problems[prob]['preload']:

            prob_temp = temp_store + prob + '/'

            # Read in the datasets - run in separate process so can free up memory when done
            print('Reading datasets and building vocab for {} problem...'.format(prob))
            p = Pool(processes=1)
            v_load = p.starmap(prep_data, [(args, prob, dd, vocab, False, prob_temp, temp_store)])
            p.close()

            v_load = v_load[0]

            # Load the vocab object as too large to pass back from multiprocess
            if v_load:
                vocab = joblib.load(temp_store + '/vocab')

            # Read the val iterator into memory and pass list of train iterator names
            fs = os.listdir(prob_temp)
            prob_iterators[prob] = {}
            prob_iterators[prob]['train'] = [prob_temp + t for t in fs if 'train' in t]
            if 'val' in fs:
                prob_iterators[prob]['val'] = joblib.load(prob_temp + 'val')

        else:
            base = './data/problems/' + prob + '/data/'
            prob_iterators[prob] = {'train': [base + f for f in os.listdir(base)]}

        if dd['ent']:
            setattr(args, 'num_ent_types', problems[prob]['ent_types'])

    if v_load:
        args.vocab_size = len(vocab.word_to_index)

    # Save the args and the vocab object
    if args.save:
        save_path = './saved_models/' + '_'.join(args.problems) + '/' + args.model + '/' + args.save_id + '/'
        os.makedirs(save_path, exist_ok=True)
        args.save_path = save_path
        # Save the params
        pickle.dump(args, open(save_path + 'args.p', 'wb'))
        # Save the vocab
        if v_load:
            pickle.dump(vocab, open(save_path + 'vocab.p', 'wb'))

    # Create the model
    model_class = getattr(models, args.model)

    print('Building model...')
    model = model_class(args, vocab, **kwargs)

    if args.gpu:
        model.cuda()

    # Train the model
    print('Training...')
    r = trainer(model, prob_iterators, args, vocab, prob_losses)

    return r


if __name__ == '__main__':
    main()




