import warnings
from distutils.version import LooseVersion
from typing import Callable, Dict, Optional, Union
from configs.test_config_kyumin import get_arguments
from model.refinenetlw import rf_lw101
from dataset.cityscapes_dataset import cityscapesDataSet
import os.path as osp
import os
import numpy as np
import pdb
from collections import OrderedDict
import pickle
import torch
from torch import Tensor, nn
from torch.utils import data
from tqdm import tqdm
from torch.autograd import grad

from util import ConfusionMatrix
from functools import partial

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def lp_distances(x1: Tensor, x2: Tensor, p: Union[float, int] = 2, dim: int = 1) -> Tensor:
    return (x1 - x2).flatten(dim).norm(p=p, dim=dim)


l0_distances = partial(lp_distances, p=0)
l1_distances = partial(lp_distances, p=1)
l2_distances = partial(lp_distances, p=2)
linf_distances = partial(lp_distances, p=float('inf'))

_default_metrics = OrderedDict([
    ('linf', linf_distances),
    ('l0', l0_distances),
    ('l1', l1_distances),
    ('l2', l2_distances),
])


class ForwardCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_samples_called = 0

    def __call__(self, module, input) -> None:
        self.num_samples_called += len(input[0])


class BackwardCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_samples_called = 0

    def __call__(self, module, grad_input, grad_output) -> None:
        self.num_samples_called += len(grad_output[0])

def difference_of_logits(logits: Tensor, labels: Tensor, labels_infhot: Optional[Tensor] = None) -> Tensor:
    if labels_infhot is None:
        labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))

    class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other_logits = (logits - labels_infhot).amax(dim=1)
    return class_logits - other_logits

def DAG_Attack(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        interp,
        masks: Tensor = None,
        targeted: bool = False,
        adv_threshold: float = 0.99,
        max_iter: int = 200,
        γ: float = 0.5,
        p: float = float('inf'),
        callback = None) -> Tensor:
    """DAG attack from https://arxiv.org/abs/1703.08603"""
    pdb.set_trace()
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(-1, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1

    # Setup variables
    r = torch.zeros_like(inputs)

    # Init trackers
    best_adv_percent = torch.zeros(batch_size, device=device)
    adv_found = torch.zeros_like(best_adv_percent, dtype=torch.bool)
    best_adv = inputs.clone()

    for i in range(max_iter):
        pdb.set_trace()
        active_inputs = ~adv_found
        inputs_ = inputs[active_inputs]
        r_ = r.clone()[active_inputs]
        r_.requires_grad_(True)

        adv_inputs_ = (inputs_ + r_).clamp(0, 1)
        logits_feature5 = model(adv_inputs_)[5]
        logits = interp(logits_feature5)

        if i == 0:
            num_classes = logits.size(1)
            if masks is None:
                masks = labels < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)
            masked_labels = labels * masks
            labels_infhot = torch.zeros_like(logits.detach()).scatter(1, masked_labels.unsqueeze(1), float('inf'))

        dl = multiplier * difference_of_logits(logits, labels=masked_labels[active_inputs],
                                               labels_infhot=labels_infhot[active_inputs])
        pixel_is_adv = dl < 0
        pdb.set_trace()
        active_masks = masks[active_inputs]
        adv_percent = (pixel_is_adv & active_masks).flatten(1).sum(dim=1) / masks_sum[active_inputs]
        is_adv = adv_percent >= adv_threshold
        adv_found[active_inputs] = is_adv
        best_adv[active_inputs] = torch.where(batch_view(is_adv), adv_inputs_.detach(), best_adv[active_inputs])

        if callback:
            callback.accumulate_line('dl', i, dl[active_masks].mean(), title=f'DAG (p={p}, γ={γ}) - DL')
            callback.accumulate_line(f'L{p}', i, r.flatten(1).norm(p=p, dim=1).mean(), title=f'DAG (p={p}, γ={γ}) - Norm')
            callback.accumulate_line('adv%', i, adv_percent.mean(), title=f'DAG (p={p}, γ={γ}) - Adv percent')

            if (i + 1) % (max_iter // 20) == 0 or (i + 1) == max_iter:
                callback.update_lines()

        if is_adv.all():
            break
        pdb.set_trace()
        # r_.requires_grad_(True)
        loss = (dl[~is_adv] * active_masks[~is_adv]).relu()
        r_grad = grad(loss.sum(), r_, only_inputs=True, retain_graph=True, allow_unused=True)[0]
        if(r_grad != None):
            print("r_grad is not None!")
        else:
            print("r_grad is None")
        r_grad.div_(batch_view(r_grad.flatten(1).norm(p=p, dim=1).clamp_min_(1e-8)))
        r_.data.sub_(r_grad, alpha=γ)

        r[active_inputs] = r_

    if callback:
        callback.update_lines()

    return best_adv

def run_attack(model,
               loader,
               target: Optional[Union[int, Tensor]] = None,
               metrics: Dict[str, Callable] = _default_metrics,
               return_adv: bool = True) -> dict:
    pdb.set_trace()
    device = next(model.parameters()).device
    targeted = True if target is not None else False
    loader_length = len(loader)
    # image_list = getattr(loader.sampler.data_source, 'dataset', loader.sampler.data_source).images

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    forward_counter, backward_counter = ForwardCounter(), BackwardCounter()
    model.register_forward_pre_hook(forward_counter)
    if LooseVersion(torch.__version__) >= LooseVersion('1.8'):
        model.register_full_backward_hook(backward_counter)
    else:
        model.register_backward_hook(backward_counter)
    forwards, backwards = [], []  # number of forward and backward calls per sample

    times, accuracies, apsrs, apsrs_orig = [], [], [], []
    distances = {k: [] for k in metrics.keys()}

    pdb.set_trace()
    if return_adv:
        images, adv_images = [], []

    for i, (image, label, size, name) in enumerate(tqdm(loader, ncols=80, total=loader_length)):
        pdb.set_trace()
        if return_adv:
            images.append(image.clone())

        interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')

        image, label = image.to(device), label.to(device).squeeze(1).long()
        if targeted:
            if isinstance(target, Tensor):
                attack_label = target.to(device).expand(image.shape[0], -1, -1)
            elif isinstance(target, int):
                attack_label = torch.full_like(label, fill_value=target)
        else:
            attack_label = label


        pdb.set_trace()
        logits_feature5 = model(image)[5]
        logits=interp(logits_feature5)
        if i == 0:
            num_classes = logits.size(1)
            confmat_orig = ConfusionMatrix(num_classes=num_classes)
            confmat_adv = ConfusionMatrix(num_classes=num_classes)

        mask = label < num_classes
        mask_sum = mask.flatten(1).sum(dim=1)
        pred = logits.argmax(dim=1)
        accuracies.extend(((pred == label) & mask).flatten(1).sum(dim=1).div(mask_sum).cpu().tolist())
        confmat_orig.update(label, pred)

        if targeted:
            target_mask = attack_label < logits.size(1)
            target_sum = target_mask.flatten(1).sum(dim=1)
            apsrs_orig.extend(((pred == attack_label) & target_mask).flatten(1).sum(dim=1).div(target_sum).cpu().tolist())
        else:
            apsrs_orig.extend(((pred != label) & mask).flatten(1).sum(dim=1).div(mask_sum).cpu().tolist())
        pdb.set_trace()
        forward_counter.reset(), backward_counter.reset()
        start.record()
        adv_image = DAG_Attack(model=model, inputs=image,interp=interp, labels=attack_label, targeted=targeted)
        # performance monitoring
        end.record()
        torch.cuda.synchronize()
        times.append((start.elapsed_time(end)) / 1000)  # times for cuda Events are in milliseconds
        forwards.append(forward_counter.num_samples_called)
        backwards.append(backward_counter.num_samples_called)
        forward_counter.reset(), backward_counter.reset()
        pdb.set_trace()
        if adv_image.min() < 0 or adv_image.max() > 1:
            warnings.warn('Values of produced adversarials are not in the [0, 1] range -> Clipping to [0, 1].')
            adv_image.clamp_(min=0, max=1)

        if return_adv:
            adv_images.append(adv_image.cpu().clone())
        pdb.set_trace()
        adv_logits_feature5 = model(adv_image)[5]
        adv_logits = interp(adv_logits_feature5)
        adv_pred = adv_logits.argmax(dim=1)
        confmat_adv.update(label, adv_pred)
        if targeted:
            apsrs.extend(((adv_pred == attack_label) & target_mask).flatten(1).sum(dim=1).div(target_sum).cpu().tolist())
        else:
            apsrs.extend(((adv_pred != label) & mask).flatten(1).sum(dim=1).div(mask_sum).cpu().tolist())

        for metric, metric_func in metrics.items():
            distances[metric].extend(metric_func(adv_image, image).detach().cpu().tolist())

    pdb.set_trace()
    acc_global, accs, ious = confmat_orig.compute()
    adv_acc_global, adv_accs, adv_ious = confmat_adv.compute()

    data = {
        # 'image_names': image_list[:len(apsrs)],
        'targeted': targeted,
        'accuracy': accuracies,
        'acc_global': acc_global.item(),
        'adv_acc_global': adv_acc_global.item(),
        'ious': ious.cpu().tolist(),
        'adv_ious': adv_ious.cpu().tolist(),
        'apsr_orig': apsrs_orig,
        'apsr': apsrs,
        'times': times,
        'num_forwards': forwards,
        'num_backwards': backwards,
        'distances': distances,
    }
    pdb.set_trace()
    if return_adv:
        shapes = [img.shape for img in images]
        if len(set(shapes)) == 1:
            images = torch.cat(images, dim=0)
            adv_images = torch.cat(adv_images, dim=0)
        data['images'] = images
        data['adv_images'] = adv_images

    return data


if __name__ == "__main__":
    print("start adversarial_attack.py")

    args = get_arguments()
    restore = torch.load(args.restore_from)

    print("restoring loading from previous model is done")

    model = rf_lw101(num_classes=args.num_classes)

    model.load_state_dict(restore['state_dict'])
    start_iter = 0

    n_classes = args.num_classes

    save_dir_adversarial = osp.join(f'./result_adversarial', args.file_name)
    
    if not os.path.exists(save_dir_adversarial):
        os.makedirs(save_dir_adversarial)

    model.eval()
    print("evaluate model")

    device = torch.device("cuda:0")

    model = model.to(device)
    testloader = data.DataLoader(cityscapesDataSet(args.data_dir_city, args.data_city_list, crop_size = (2048, 1024), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                            batch_size=1, shuffle=False, pin_memory=True)


    adversarial_examples = run_attack(model, testloader)

    print("after creating adversarial_examples")
    
    pdb.set_trace()
    # save adversarial examples([adversarial examples, labels])
    with open('../data/adversarial_example', 'wb') as fp:
        pickle.dump(adversarial_examples, fp)
