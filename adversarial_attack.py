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
from torch.autograd import grad, Variable

from util import ConfusionMatrix, make_one_hot, generate_target
from functools import partial
import random
from dag_medical import DAG

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
        label, 
        adv_label,
        interp,
        device='cuda:0',
        masks: Tensor = None,
        targeted: bool = False,
        adv_threshold: float = 0.99,
        max_iter: int = 200,
        gamma: float = 0.5,
        p: float = float('inf'),
        callback = None) -> Tensor:
    """DAG attack from https://arxiv.org/abs/1703.08603"""
    pdb.set_trace()
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(-1, *[1] * (inputs.ndim - 1))
    multiplier = -1 if targeted else 1

    # Setup variables
    r = torch.zeros_like(inputs)

    # Init trackers
    best_adv_percent = torch.zeros(batch_size, device=device)   # [0.6]
    adv_found = torch.zeros_like(best_adv_percent, dtype=torch.bool)
    # adv_found = torch.zeros_like(inputs, dtype=torch.bool)
    pixel_adv_found = torch.zeros_like(label, dtype=torch.bool)
    best_adv = inputs.clone()
    image = inputs.clone()

    image.requires_grad_(True)
    masks = masks.to(device)
    masks_sum = masks.flatten(1).sum(dim=1)

    for i in range(max_iter):
        pdb.set_trace()
        logits_feature5 = model(image)[5]
        logits = interp(logits_feature5)
        _,predictions=torch.max(logits,1)
        pixel_is_adv_label = (predictions != labels)
        predictions=make_one_hot(predictions,logits.shape[1],device)

        adv_log = torch.mul(logits, adv_label)
        clean_log = torch.mul(logits, label)

        r_m = adv_log - clean_log

        r_m_sum = r_m.sum()
        r_m_sum.requires_grad_(True)
        r_m_grad = grad(r_m_sum, image, retain_graph=True)[0]
        
        r_m_grad.div_(batch_view(r_m_grad.flatten(1).norm(p=p, dim=1).clamp_min_(1e-8)))
        r.data.sub_(r_m_grad, alpha=gamma)

        # r.data.add_(r_m_grad)

        image = (image + r).clamp(0, 1)

        # pixel_is_adv = r_m < 0
        pdb.set_trace()
        pixel_is_adv = (predictions != label)
        # pixel_adv_found.logical_or_(pixel_is_adv)
        adv_percent = (pixel_is_adv & masks).flatten(1).sum(dim=1) / masks_sum
        
        is_adv = adv_percent >= adv_threshold
        best_adv = torch.where(batch_view(is_adv), image.detach(), best_adv)



        pdb.set_trace()
        if adv_percent >= adv_threshold:
            break

        if callback:
            callback.accumulate_line('dl', i, r_m.mean(), title=f'DAG (p={p}, gamma={gamma}) - DL')
            callback.accumulate_line(f'L{p}', i, r.flatten(1).norm(p=p, dim=1).mean(), title=f'DAG (p={p}, gamma={gamma}) - Norm')
            callback.accumulate_line('adv%', i, adv_percent.mean(), title=f'DAG (p={p}, gamma={gamma}) - Adv percent')

            if (i + 1) % (max_iter // 20) == 0 or (i + 1) == max_iter:
                callback.update_lines()

    if callback:
        callback.update_lines()

    return best_adv

def run_attack(model,
               loader,
               device,
               target: Optional[Union[int, Tensor]] = None,
               metrics: Dict[str, Callable] = _default_metrics,
               return_adv: bool = True) -> dict:
    # pdb.set_trace()
    targeted = True if target is not None else False
    loader_length = len(loader)
    # image_list = getattr(loader.sampler.data_source, 'dataset', loader.sampler.data_source).images

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # forward_counter, backward_counter = ForwardCounter(), BackwardCounter()
    # model.register_forward_pre_hook(forward_counter)
    # if LooseVersion(torch.__version__) >= LooseVersion('1.8'):
    #     model.register_full_backward_hook(backward_counter)
    # else:
    #     model.register_backward_hook(backward_counter)
    forwards, backwards = [], []  # number of forward and backward calls per sample

    times, accuracies, apsrs, apsrs_orig = [], [], [], []
    distances = {k: [] for k in metrics.keys()}

    # pdb.set_trace()
    if return_adv:
        images, adv_images = [], []

    # for i, (image, label, size, name) in enumerate(tqdm(loader, ncols=80, total=loader_length)):
    for index, batch in enumerate(testloader):
        image, label, size, name = batch
        image = image.clone().detach().float()
        # pdb.set_trace()
        if return_adv:
            images.append(image)

        interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')
        image = Variable(image).to(device)
        # label = label.to(device).squeeze(1).long()
        
        mask = label < n_classes
        mask_sum = mask.flatten(1).sum(dim=1)
        label = label * mask

        label = label.clone().detach().float()
        label = label.to(device)        

        # if targeted:
        #     if isinstance(target, Tensor):
        #         attack_label = target.to(device).expand(image.shape[0], -1, -1)
        #     elif isinstance(target, int):
        #         attack_label = torch.full_like(label, fill_value=target)
        # else:
        #     attack_label = label

        label_oh = make_one_hot(label.long(),n_classes, device)

        unique_label = torch.unique(label)
        target_class = int(random.choice(unique_label[1:]).item())

        adv_target=generate_target(label_oh.cpu().numpy(), target_class = target_class)
        adv_target=torch.from_numpy(adv_target).float()
        
        # pdb.set_trace()
        adv_target=adv_target.to(device)
        # adversarial_image = DAG(model=model,
        #           model_name="FIFO",
        #           image_name=name,
        #           image=image,
        #           ground_truth=label_oh,
        #           adv_target=adv_target,
        #           interp=interp,
        #           verbose=True,
        #           pure_label=None)
        adv_image = DAG_Attack(model=model, label=label_oh, labels=label, masks=mask,
                               adv_label = adv_target, inputs=image,interp=interp, targeted=targeted)
       
        pdb.set_trace()
        logits_feature5 = model(image)[5]
        logits=interp(logits_feature5)
        if index == 0:
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


    adversarial_examples = run_attack(model, testloader, device)

    print("after creating adversarial_examples")
    
    pdb.set_trace()
    # save adversarial examples([adversarial examples, labels])
    with open('../data/adversarial_example', 'wb') as fp:
        pickle.dump(adversarial_examples, fp)
