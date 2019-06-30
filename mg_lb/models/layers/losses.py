import torch


class pt_cross_entropy(torch.nn.Module):
    """
    Wrapper for the official pytorch cross entropy loss, which checks data size is correct
    """
    def __init__(self):
        super(pt_cross_entropy, self).__init__()

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, predictions, targets):

        assert targets.size(0) == predictions.size(0), 'Targets and predictions should be the same length'
        assert len(targets.size()) == 2, 'Targets should be 2d'

        # Reduce targets to 2d
        targets = targets.squeeze(1)

        # Pytorch loss functions have predictions as first arg and targets as second
        loss = self.loss(predictions, targets)

        return loss


class pt_cluster_loss(torch.nn.Module):
    """
    Calculate the loss from grouping words in same cluster together
    """
    def __init__(self, mse):
        super(pt_cluster_loss, self).__init__()

        if mse:
            self.loss = pt_mse()
        else:
            self.loss = pt_mae()

    def forward(self, previous, weights, boolean_mask):
        """
        Args
        previous: target weights. [batch_size, seq_len, num_cluster_levels, 2]
        weights: predicted weights. [batch_size, seq_len, num_cluster_levels, 2]
        boolean_mask: [batch_size, seq_len, num_cluster_levels]
        """
        # [total_num_ents, 2]
        weights = weights[boolean_mask]

        # [total_num_ents, 2]
        previous = previous[boolean_mask]

        # The loss from the groupings/weights
        weights_loss = self.loss(weights, previous)

        return weights_loss


def get_ce_weights(lookup, np_weight, o_weight):
    """
    Weights for cross entropy loss when doing NER
    """
    wixs = torch.ones([len(lookup)])

    if np_weight is not None:
        # Reduce weight on NP labels
        for k, v in lookup.items():
            if '-NP' in k:
                wixs[v] = np_weight

    if o_weight is not None:
        # Reduce weight on 'O' labels
        for k, v in lookup.items():
            if k == 'O':
                wixs[v] = o_weight

    return wixs


class named_ent_loss(torch.nn.Module):
    """
    Calculate loss for named entity recognition including clustering of ents into single vectors
    """
    def __init__(self, mse, cluster_weight, np_weight, o_weight, lookup):
        super(named_ent_loss, self).__init__()

        self.cluster_loss = pt_cluster_loss(mse)
        self.cluster_weight = cluster_weight

        wixs = get_ce_weights(lookup, np_weight, o_weight)

        self.loss = torch.nn.CrossEntropyLoss(weight=wixs)

    def forward(self, predictions, targets, previous, weights, boolean_mask):
        """
        Args
        predictions: [total_num_ents, num_ent_types]
        targets: [total_num_ents, 1]
        previous: [batch_size, seq_len, num_cluster_levels, 2]
        weights: [batch_size, seq_len, num_cluster_levels, 2]
        boolean_mask: [batch_size, seq_len, num_cluster_levels]
        """
        # Cross entropy loss. [total_num_ents]
        ce_loss = self.loss(predictions, targets.squeeze(1))

        # Cluster loss
        cluster_loss = self.cluster_loss(previous, weights, boolean_mask)

        return ce_loss + self.cluster_weight * cluster_loss


class pt_mse(torch.nn.Module):
    """
    MSE loss with optional masking
    """
    def __init__(self):
        super(pt_mse, self).__init__()

    def forward(self, predictions, targets, mask=None, average=True):

        if predictions is None:
            return None

        assert targets.size() == predictions.size(), 'Targets and predictions should be the same shape'

        difference = targets - predictions

        if mask is not None:
            assert average is True, 'Shouldn\'t apply both mask weighting and weighting in NN model'
            assert mask.size() == predictions.size(), 'Mask should be same size as predictions'

            difference = mask * difference

            loss = (difference ** 2).sum() / mask.sum()

        else:
            loss = difference ** 2
            if average:
                loss = loss.mean()

        return loss


class pt_mae(torch.nn.Module):
    """
    MAE loss with optional masking
    """
    def __init__(self):
        super(pt_mae, self).__init__()

    def forward(self, predictions, targets, mask=None, average=True):

        assert targets.size() == predictions.size(), 'Targets and predictions should be the same shape'

        difference = targets - predictions

        if mask is not None:
            assert average is True, 'Shouldn\'t apply both mask weighting and weighting in NN model'
            assert mask.size() == predictions.size(), 'Mask should be same size as predictions'

            difference = mask * difference

            loss = (torch.abs(difference)).sum() / mask.sum()

        else:
            loss = torch.abs(difference)
            if average:
                loss = loss.mean()

        return loss


pt_losses = {
    'mse': pt_mse,
    'mae': pt_mae,
    'named_ent_loss': named_ent_loss,
    'cluster_loss': pt_cluster_loss
}