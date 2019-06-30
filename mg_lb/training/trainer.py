from tqdm import tqdm
from torch.autograd import Variable
import time

from mg_lb.data_loading.prep_tensors import *
from mg_lb.data_loading.iterators import group_iterator
from mg_lb.models.layers.losses import pt_losses
from mg_lb.training.learning_rates import get_decay
from mg_lb.models.model_args import model_args as margs
from mg_lb.training.train_fs import *
from mg_lb.problems.probs import problems as pr_dict
import mg_lb.training.accuracy as accfs


def add_returns(prob_dict):
    for p, v in prob_dict.items():
        if v['ent']:
            v['returns'] = ents_process
        else:
            v['returns'] = None
    return prob_dict


def loss_kwargs(loss_type, problem, args):

    if loss_type == 'named_ent_loss':
        r = {'cluster_weight': args.cluster_weight, 'np_weight': args.np_weight,
             'o_weight': args.o_weight, 'lookup': labels_lookup[problem], 'mse': args.cl_loss_mse}
    elif loss_type == 'cluster_loss':
        r = {'mse': args.cl_loss_mse}
    else:
        r = {}

    return r


def get_loss(loss_type, loss_fn, pred, tensors):
    if loss_type == 'named_ent_loss':
        loss = loss_fn(pred['ent_preds'], tensors['ent_labels'], tensors['previous'],
                       pred['weights'], tensors['boolean_mask'])
    elif loss_type == 'cluster_loss':
        loss = loss_fn(tensors['previous'], pred['weights'], tensors['boolean_mask'])
    return loss


def run_dataset(model, problem, iterator, loss_fn, loss_name, args, prob_dict, show_progress=False, labels=True,
                last=False):

    curr_prob = pr_dict[problem]

    # Put model in eval mode and don't limit sentence length
    model.eval()

    # Put batches in a set order or shuffle if not using all of them
    if not last and problem in limit_lookup.keys():
        iterator.batch_order = list(np.linspace(0, iterator.num_batches - 1, limit_lookup[problem], dtype=int))
    else:
        iterator.reset_order()

    # Accuracy functions
    acc_func = getattr(accfs, curr_prob['accuracy']['process'])
    acc_key = curr_prob['accuracy']['key']

    returns_func = prob_dict[problem]['returns']

    iterator.counter = 0

    losses, weights, used_ixs = [], [], []

    save_vals = {}
    for k in curr_prob['save_tensors']:
        save_vals[k] = []

    for batch_ix in tqdm(range(iterator.num_batches), total=iterator.num_batches, disable=not show_progress):

        batch = iterator.next_batch()

        # Save which batch ixs have been used, so can lookup correct labels in accuracy function
        used_ixs.append(batch['index'])

        tensors = margs[args.model]['prep_tensors'](batch, args=args)
        tensors['problem'] = problem
        tensors['epoch'] = 1000

        # Forward pass
        with torch.no_grad():
            returns = model(tensors)

        # Process returns
        if returns_func is not None:
            returns, tensors = returns_func(tensors, returns, problem, train=False)

        rt = {}

        if labels:
            loss = get_loss(loss_name, loss_fn, returns, tensors)
            loss = loss * prob_dict[problem]['loss_weight']
            losses.append(loss.item())

            if 'sentences' in batch.keys():
                weights.append(batch['sentences'].shape[0] * batch['sentences'].shape[1])
            else:
                weights.append(batch['labels'].shape[0])

        for k, v in save_vals.items():
            if k in returns.keys():
                v.append(returns[k].cpu().data.numpy())

        # Break early to save time during training if not at the end of the epoch
        if not last and problem in limit_lookup.keys():
            if (batch_ix + 1) == limit_lookup[problem]:
                break

    if labels:
        # Calculate average loss for the dataset (weighted by batch sizes)
        rt['loss'] = np.average(losses, weights=weights)

    # Set model back in training mode
    model.train()

    # Process returns for accuracy calculation
    pred = [acc_func(i) for i in save_vals[acc_key]]

    if len(pred) > 0:
        save_vals['preds'] = np.concatenate(pred, axis=0)

        save_vals['acc_target'] = np.concatenate(save_vals['acc_target'], axis=0)

        if 'acc_weights' in save_vals.keys():
            save_vals['f1_weights'] = np.concatenate(save_vals['acc_weights'], axis=0)
        else:
            save_vals['f1_weights'] = None

        if 'f1_target' in save_vals.keys():
            save_vals['f1_target'] = np.concatenate(save_vals['f1_target'], axis=0)
        else:
            save_vals['f1_target'] = None

    rt = {**rt, **save_vals}

    rt['used_ixs'] = used_ixs

    return rt


def nlp_eval(model, iterator, args, prob_dict, show_progress=True, labels=True, eval_loss=True, print_loss=False):

    # Add returns functions
    prob_dict = add_returns(prob_dict)

    # Put model on GPU
    if args.gpu:
        model.cuda()

    model.eval()

    loss_ks = loss_kwargs(args.loss_fn, args.problems, args)
    loss_fn = pt_losses[args.loss_fn](**loss_ks).cuda()

    st = time.time()

    returns = run_dataset(model, args.problems, iterator, loss_fn, args.loss_fn, prob_dict=prob_dict, args=args, show_progress=show_progress,
                          labels=labels, last=True)

    returns['time'] = time.time() - st

    if eval_loss:

        if 'accuracy' in pr_dict[args.problems].keys() and 'preds' in returns.keys():
            # iterator.all_labels(returns['used_ixs'])

            ac_d = {}
            ac_c = {}

            acc = get_accuracy(returns['acc_target'], returns['preds'], returns['f1_weights'],
                               returns['f1_target'], prob=args.problems, args=args)

            ac_d[0] = acc['accuracy']
            ac_c[0] = acc['correct']

            if print_loss:
                if acc['f1'] is not None:
                    print('Accuracy: {:.4f}, F1: {:.4f}'.format(acc['accuracy'], acc['f1']))
                else:
                    print('Accuracy: {:.3f}'.format(acc['accuracy']))

            returns['accuracy'] = ac_d
            returns['correct'] = ac_c

        if print_loss:
            print('Loss: {:.4f}'.format(returns['loss']))

    return returns


def trainer(model, iterators, args, vocab, prob_dict):

    # The main problem which we'll save if model improves on
    main_prob = args.problems[0]

    # Data loading fs etc. for this model
    fs = margs[args.model]

    # Get an iterator which will return one batch from the different problems in turn when
    # .next_batch() called
    train_iter = group_iterator(iterators, viz_every=args.viz_every, shuffle=args.shuffle_batches,
                                args=args, vocab=vocab, prob_dict=prob_dict)

    model.train()

    # Get the loss function and optimizer
    loss_fns = {}
    loss_fs_sts = {}
    for prob, v in prob_dict.items():
        loss_ks = loss_kwargs(v['loss'][0], prob, args)
        loss_fns[prob] = pt_losses[v['loss'][0]](**loss_ks).cuda()
        loss_fs_sts[prob] = v['loss'][0]

    # Add returns functions
    prob_dict = add_returns(prob_dict)

    # Initialize optimizer
    opt_choice = getattr(torch.optim, args.optimizer)

    nps = [p for n, p in model.named_parameters() if p.requires_grad and 'memory_embeddings' not in n]
    optimizer = opt_choice(nps, lr=args.learning_rate)

    clip_params = model.parameters()

    best_val_loss = np.float('inf')
    best_val_acc = 0

    global_step = 0

    # Function for decaying learning rate
    lr_decay = get_decay(args.learning_rate_decay)

    for epoch in range(args.num_epochs):

        print('Epoch {}'.format(epoch))

        # Adjust the learning rate
        if args.learning_rate_decay != 'None' and global_step > args.lr_warmup:
            new_lr = lr_decay(args.learning_rate, epoch)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

            print('Learning rate set at {}'.format(new_lr))

        # Keep track of train losses as we go through epoch
        train_losses = {}
        for prob in args.problems:
            train_losses[prob] = []

        if epoch != 0 and args.save:
            # Reset model to the best val accuracy/loss achieved in previous epochs
            model.load_state_dict(torch.load(args.save_path + args.save_id))

        st = time.time()

        for step in range(10000000):

            # Learning rate warmup
            if global_step < args.lr_warmup and global_step % 100 == 0:
                newlr = warmup(global_step, args.learning_rate, args.lr_warmup)
                for pg in optimizer.param_groups:
                    pg['lr'] = newlr

                print('Warmup lr set at {}'.format(newlr))

            # Initialize main learning rate
            if step == args.lr_warmup:
                for pg in optimizer.param_groups:
                    pg['lr'] = args.learning_rate

            batch, finished_epoch, last, ptype = train_iter.next_batch()

            tensors = fs['prep_tensors'](batch, args=args)
            tensors['epoch'] = epoch
            tensors['problem'] = ptype

            # Get rid of gradients from previous step
            model.zero_grad()

            # Forward pass
            pred = model(tensors)

            # Process tensors
            returns_func = prob_dict[ptype]['returns']

            # Process returns
            if returns_func is not None:
                pred, tensors = returns_func(tensors, pred, ptype)

            # Loss and backwards pass
            loss = get_loss(loss_fs_sts[ptype], loss_fns[ptype], pred, tensors)

            # Weight the loss for this problem
            loss = loss * prob_dict[ptype]['loss_weight']

            loss.backward()

            # Gradient clipping
            if args.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(clip_params, args.max_grad_norm)

            optimizer.step()

            # Keep running track of train losses
            train_losses[ptype].append(loss.data.item())

            global_step += 1

            if step % train_iter.viz_number == 0 or finished_epoch:

                print('Time:', time.time() - st)
                st = time.time()

                train_loss = {}

                for prob in args.problems:
                    print(prob)

                    train_loss[prob] = np.average(train_losses[prob])
                    train_losses[prob] = []

                    val_rt = run_dataset(model, prob, iterators[prob]['val'], loss_fns[prob], loss_fs_sts[prob], args=args, last=last, prob_dict=prob_dict)
                    print('Step: {} Train Loss: {:.4f} Val Loss: {:.4f}'.format(step, train_loss[prob], val_rt['loss']))

                    use_accuracy = (prob == main_prob) and ('accuracy' in pr_dict[prob].keys())

                    if use_accuracy:
                        val_acc = get_accuracy(val_rt['acc_target'], val_rt['preds'], val_rt['f1_weights'], val_rt['f1_target'], last, prob, args)
                        print('Val Accuracy: {:.4f}'.format(val_acc['accuracy']))

                        if val_acc['f1'] is not None:
                            print('Val F1: {:.4f}'.format(val_acc['f1']))

                    # Save the model if we improve the val loss/accuracy
                    if prob == main_prob:
                        if use_accuracy:
                            if val_acc[args.acc_key] >= best_val_acc:
                                best_val_acc = val_acc[args.acc_key]
                                if args.save:
                                    torch.save(model.state_dict(), args.save_path + args.save_id)


                        else:
                            if val_rt['loss'] <= best_val_loss:
                                best_val_loss = val_rt['loss']
                                if args.save:
                                    torch.save(model.state_dict(), args.save_path + args.save_id)

            if finished_epoch:
                break

    return None