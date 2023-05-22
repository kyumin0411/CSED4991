import numpy as np
import torch
import torch.nn as nn
import pdb
import scipy.ndimage

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


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def __str__(self):
        acc_global, accs, ious = self.compute()
        return ('Row correct: ' + '|'.join(f'{acc:>5.2%}' for acc in accs.tolist()) + '\n'
                                                                                      f'IoUs       : ' + '|'.join(
            f'{iou:>5.2%}' for iou in ious.tolist()) + '\n'
                                                       f'Pixel Acc. : {acc_global.item():.2%}\n'
                                                       f'mIoU       : {ious.nanmean().item():.2%}')


def generate_target(y_test, target_class = 13, width = 256, height = 256):
    
    y_target = y_test

    dilated_image = scipy.ndimage.binary_dilation(y_target[0, target_class, :, :], iterations=6).astype(y_test.dtype)

    for i in range(width):
        for j in range(height):
            y_target[0, target_class, i, j] = dilated_image[i,j]

    for i in range(width):
        for j in range(height):
            potato = np.count_nonzero(y_target[0,:,i,j])
            if (potato > 1):
                x = np.where(y_target[0, : ,i, j] > 0)
                k = x[0]
                #print("{}, {}, {}".format(i,j,k))
                if k[0] == target_class:
                    y_target[0,k[1],i,j] = 0.
                else:
                    y_target[0, k[0], i, j] = 0.

    return y_target