from typing import Optional

import torch
from adv_lib.utils.losses import difference_of_logits
from adv_lib.utils.visdom_logger import VisdomLogger
from torch import Tensor, nn
from torch.autograd import grad
import pdb


def DAG(model,
        image,
        ground_truth,
        interp,
        model_name,
        image_name,
        masks = None,
        # targeted = False,
        adv_threshold: float = 0.99,
        num_iterations: int = 200,
        gamma: float = 0.5,
        p: float = float('inf'),
        device='cuda:0',
        verbose=False,
        callback: Optional[VisdomLogger] = None) -> Tensor:
    """DAG attack from https://arxiv.org/abs/1703.08603"""
    # device = inputs.device
    pdb.set_trace()
    orig_image=image
    batch_size = len(image)
    batch_view = lambda tensor: tensor.view(-1, *[1] * (image.ndim - 1))
    # multiplier = -1 if targeted else 1
    multiplier = 1

    # Setup variables
    r = torch.zeros_like(image)

    pdb.set_trace()
    # Init trackers
    best_adv_percent = torch.zeros(batch_size, device=device)
    adv_found = torch.zeros_like(best_adv_percent, dtype=torch.bool)
    best_adv = image.clone()

    for i in range(num_iterations):
        pdb.set_trace()
        active_inputs = ~adv_found
        inputs_ = image[active_inputs]
        r_ = r[active_inputs]
        r_.requires_grad_(True)

        pdb.set_trace()
        adv_inputs_ = (inputs_ + r_).clamp(0, 1)
        logits_feature5 = model(adv_inputs_)[5]
        logits=interp(logits_feature5)

        pdb.set_trace()
        if i == 0:
            num_classes = logits.size(1)
            if masks is None:
                masks = ground_truth < num_classes
            masks_sum = masks.flatten(1).sum(dim=1)
            masked_labels = ground_truth * masks
            labels_infhot = torch.zeros_like(logits.detach()).scatter(1, masked_labels.unsqueeze(1), float('inf'))

        dl = multiplier * difference_of_logits(logits, labels=masked_labels[active_inputs],
                                               labels_infhot=labels_infhot[active_inputs])
        pdb.set_trace()
        pixel_is_adv = dl < 0

        active_masks = masks[active_inputs]
        adv_percent = (pixel_is_adv & active_masks).flatten(1).sum(dim=1) / masks_sum[active_inputs]
        is_adv = adv_percent >= adv_threshold
        adv_found[active_inputs] = is_adv
        best_adv[active_inputs] = torch.where(batch_view(is_adv), adv_inputs_.detach(), best_adv[active_inputs])

        pdb.set_trace()
        if callback:
            callback.accumulate_line('dl', i, dl[active_masks].mean(), title=f'DAG (p={p}, gamma={gamma}) - DL')
            callback.accumulate_line(f'L{p}', i, r.flatten(1).norm(p=p, dim=1).mean(), title=f'DAG (p={p}, gamma={gamma}) - Norm')
            callback.accumulate_line('adv%', i, adv_percent.mean(), title=f'DAG (p={p}, gamma={gamma}) - Adv percent')

            if (i + 1) % (num_iterations // 20) == 0 or (i + 1) == num_iterations:
                callback.update_lines()

        if is_adv.all():
            break

        pdb.set_trace()
        loss = (dl[~is_adv] * active_masks[~is_adv]).relu()
        r_grad = grad(loss.sum(), r_, only_inputs=True)[0]
        r_grad.div_(batch_view(r_grad.flatten(1).norm(p=p, dim=1).clamp_min_(1e-8)))
        r_.data.sub_(r_grad, alpha=gamma)

        r[active_inputs] = r_

    if verbose:
        print("image number : ", image_number)
        print("original condition : ", orig_condition.float().sum())
        print("adversarial condition : ", condition1.float().sum())
        print("condition is ", cond.sum())

    data_path = "../data/adversarial/" + model_name + "_" + image_name[0].split('/')[1]


    # if adversarial_image!=None:
    #     np_arr = np.array(adversarial_image, dtype=np.uint8)
    #     img = PIL.Image.fromarray(np_arr)
    #     img.save(data_path)
        # with open(data_path, 'wb') as fp:
        #     pickle.dump(adversarial_image, fp)

    if callback:
        callback.update_lines()

    return best_adv