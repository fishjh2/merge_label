import numpy as np
import itertools
import pickle
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from mg_lb.data_loading.fs import read_csv, basic_iterator
from mg_lb.data_loading.data_prep import iterator_load


def load_iter(path, problem, args, vocab, prob_dict):
    data, tokenized = read_csv(problem, path, args.add_upper, prob_dict[problem])

    name = path.split('/')[-1]

    # Load the iterator
    it, _ = iterator_load(name, problem, data, vocab, args, use_flair=False, fl=None, extend_vocab=False,
                          prob_dict=prob_dict[problem], tokenized=tokenized, loud=False)

    return it


class ptDataset(Dataset):

    def __init__(self, paths, problem=None, vocab=None, prob_dict=None, args=None):
        self.paths = paths
        self.load = '/data/problems/' in paths[0]
        self.problem = problem
        self.vocab = vocab
        self.prob_dict = prob_dict
        self.args = args

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):

        if self.load:
            it = load_iter(self.paths[ix], self.problem, self.args, self.vocab, self.prob_dict)
        else:
            it = pickle.load(open(self.paths[ix], 'rb'))

        return {'inputs': it.inputs, 'key': it.lab_key, 'batch_list': it.batch_list, 'file': self.paths[ix]}


def collate_fn(batch):

    inputs = defaultdict(list)
    batch_order = []

    for b in batch:
        for k, v in b['inputs'].items():
            inputs[k] += v

        if len(batch_order) > 0:
            max_b = np.max(batch_order[-1]) + 1
            batch_order += [[k + max_b for k in j] for j in b['batch_list']]
        else:
            batch_order += b['batch_list']

        label = b['key']

    it = basic_iterator(inputs=inputs, batch_order=batch_order, lab_key=label)

    return it


def gen_ratios(ratios, problems):
    assert 1 in list(ratios)
    ratios = [[k] * ratios[ix] for ix, k in enumerate(problems)]
    ratios = list(filter(None.__ne__, itertools.chain.from_iterable(itertools.zip_longest(*ratios))))

    return itertools.cycle(ratios)


class group_iterator:
    def __init__(self, iterators, viz_every, shuffle, args, vocab, prob_dict):
        """
        Iterator to cycle through all the different problems we're training on at the same time
        """
        if viz_every[:5] == 'steps':
            self.viz_number = int(viz_every[6:])
            self.viz_steps = True
        else:
            self.viz_every = int(viz_every)
            self.viz_steps = False

        self.shuffle = shuffle
        self.args = args
        self.vocab = vocab
        self.prob_dict = prob_dict

        # Get rid of the val and test iterators
        self.iterators = {}
        for k, v in iterators.items():
            self.iterators[k] = v['train']

        # Apply ratios - so can train more on one problem than others
        self.key_cycle = gen_ratios(args.ratios, args.problems)

        # Find the iterator with the highest number of total batches
        bs = 0

        # Keeps one iterator for each problem in memory
        self.current_iterators = {}

        # The number of iterators for each problem
        self.counts = {}

        # Iterators for each problem, which return the next train dataset string when called
        self.dataloaders = {}
        self.dls = {}

        for k, v in self.iterators.items():

            ds = ptDataset(v, problem=k, vocab=self.vocab, prob_dict=self.prob_dict, args=self.args)

            self.dls[k] = DataLoader(dataset=ds, shuffle=shuffle, batch_size=prob_dict[k]['fs_per_iter'],
                                     collate_fn=collate_fn)

            self.dataloaders[k] = iter(self.dls[k])

            # Get first iterator
            train_iter = next(self.dataloaders[k])

            total_batches = train_iter.num_batches * len(v)

            if total_batches > bs:
                self.main = k
                bs = total_batches

            # The total number of iterators for this problem
            self.counts[k] = len(v)

            # Keep one iterator for every problem in memory
            self.current_iterators[k] = train_iter
            if shuffle:
                self.current_iterators[k].shuffle = True
                self.current_iterators[k].batch_order = self.current_iterators[k].get_batch_order()

        # Set initial viz_number value
        if not self.viz_steps:
            nb = self.current_iterators[self.main].num_batches
            self.viz_number = max(1, int(nb / self.viz_every))

    def next_batch(self):
        """
        When called, return a batch of data. Cycles through all the different problems we're
        training on at the same time.
        """

        finished_epoch = False
        last = False

        # Next dataset
        key = next(self.key_cycle)

        batch = self.current_iterators[key].next_batch()

        # Move onto next iterator for this dataset if have gone through all batches
        if self.current_iterators[key].counter == 0:

            # Return bool denoting whether the last batch from this iterator
            if key == self.main:
                last = True

            if self.counts[key] == 1:

                if self.shuffle:
                    self.current_iterators[key].batch_order = self.current_iterators[key].get_batch_order()

                if key == self.main:
                    # Have reached end of the epoch
                    finished_epoch = True

            else:

                try:
                    self.current_iterators[key] = next(self.dataloaders[key])

                    if self.shuffle:
                        self.current_iterators[key].shuffle = True
                        self.current_iterators[key].batch_order = self.current_iterators[key].get_batch_order()

                except StopIteration:
                    if key == self.main:
                        # Have reached end of the epoch
                        finished_epoch = True

                    # Reset this iterator and cycle once
                    self.dataloaders[key] = iter(self.dls[key])
                    self.current_iterators[key] = next(self.dataloaders[key])
                    if self.shuffle:
                        self.current_iterators[key].shuffle = True
                        self.current_iterators[key].batch_order = self.current_iterators[key].get_batch_order()

                    if key == self.main and not self.viz_steps:
                        nb = self.current_iterators[key].num_batches
                        self.viz_number = max(1, int(nb / self.viz_every))

        return batch, finished_epoch, last, key
