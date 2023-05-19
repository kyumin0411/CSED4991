import numpy as np
import torch
import torch.nn as nn
import pdb

def make_one_hot(labels, num_classes, device):
    '''
    Converts an integer label to a one-hot values.
    Parameters
    ----------
        labels : N x H x W, where N is batch size.(torch.Tensor)
        num_classes : int
        device: torch.device information
    -------
    Returns
        target : torch.Tensor on given device
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    # pdb.set_trace()
    labels=labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    one_hot = one_hot.to(device)
    target = one_hot.scatter_(1, labels.data, 1) 
    return target


def difference_of_logits(logits, labels, labels_infhot = None):
    if labels_infhot is None:
        labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))

    class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other_logits = (logits - labels_infhot).amax(dim=1)
    return class_logits - other_logits
