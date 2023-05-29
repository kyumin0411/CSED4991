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

from util import make_one_hot, generate_target
from functools import partial
import random
import PIL

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

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
        model_name,
        image_name,
        labels: Tensor,
        label, 
        adv_label,
        interp,
        adv_percent_file,
        device='cuda:0',
        masks: Tensor = None,
        adv_threshold: float = 0.99,
        max_iter: int = 200,
        gamma: float = 0.5,
        p: float = float('inf'),
        callback = None) -> Tensor:
    """DAG attack from https://arxiv.org/abs/1703.08603"""
    batch_view = lambda tensor: tensor.view(-1, *[1] * (inputs.ndim - 1))

    # Setup variables
    r_perturb = torch.zeros_like(inputs)

    # Init trackers
    best_adv = inputs.clone()
    image = inputs.clone()

    image.requires_grad_(True)
    masks = masks.to(device)
    masks_sum = masks.flatten(1).sum(dim=1)

    for iter in range(max_iter):
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
        r_perturb.data.sub_(r_m_grad, alpha=gamma)

        image = (image + r_perturb).clamp(0, 1)

        adv_percent = (pixel_is_adv_label & masks).flatten(1).sum(dim=1) / masks_sum

        if iter==0:
            adv_percent_file.write("1st iterate adv_percent : %f, " %(adv_percent))
       
        best_adv = torch.where(pixel_is_adv_label & masks, image.detach(), best_adv)
 
        if adv_percent >= adv_threshold:
            print('adversed at ',iter,' iterate. ',adv_percent,' %')
            break

    adv_percent_file.write("%d iterate adv_percent : %f \n" %(iter, adv_percent))

    data_path = "../data/adversarial/FIFO_adversarial_attack/adv_image/" + model_name + "_" + image_name[0].split('/')[1]
    original_data_path = "../data/adversarial/FIFO_adversarial_attack/orig_image/" + 'original' + "_" + image_name[0].split('/')[1]

    np_arr = np.array(inputs.cpu(), dtype=np.uint8)
    image_mean = torch.mean(inputs, dim=0)
    image_np = image_mean.cpu().numpy()
    image_np.transpose(1,2,0)
    image_transpose= image_np.transpose(1,2,0)
    repaired_image = image_transpose + IMG_MEAN
    np_arr= np.array(repaired_image, dtype=np.uint8)
    np_arr_RGB = np_arr[:,:,::-1]
    img = PIL.Image.fromarray(np_arr_RGB)
    img.save(original_data_path)

    adversarial_image = inputs + r_perturb

    np_arr = np.array(adversarial_image.cpu(), dtype=np.uint8)
    image_mean = torch.mean(adversarial_image, dim=0)
    image_np = image_mean.cpu().numpy()
    image_np.transpose(1,2,0)
    image_transpose= image_np.transpose(1,2,0)
    repaired_image = image_transpose + IMG_MEAN
    np_arr= np.array(repaired_image, dtype=np.uint8)
    np_arr_RGB = np_arr[:,:,::-1]
    img = PIL.Image.fromarray(np_arr_RGB)
    img.save(data_path)

    return

def run_attack(model,
               loader,
               device,
               return_adv: bool = True) -> dict:

    if return_adv:
        images, adv_images = [], []
    adv_percent_file = open("../data/adversarial/FIFO_adversarial_attack/adv_percent.txt", "w")
    testloader_iteration = 0

    for index, batch in enumerate(loader):
        image, label, size, name = batch

        if return_adv:
            images.append(image)

        interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')
        image = Variable(image).to(device)
        
        image = image.clone().detach().float()
        label_path = "../data/adversarial/FIFO_adversarial_attack/cropped_label/" + name[0].split('/')[1][:-4] + ".npy"
        label_np = label.numpy()
        np.save(label_path, label_np)
        
        label = label.clone().detach().float()
        label = label.to(device)    

        mask = label < n_classes
        mask = mask.to(device)    

        label = label * mask

        label_oh = make_one_hot(label.long(),n_classes, device)

        unique_label = torch.unique(label)
        target_class = int(random.choice(unique_label[1:]).item())

        adv_target=generate_target(label_oh.cpu().numpy(), target_class = target_class)
        adv_target=torch.from_numpy(adv_target).float()
        
        # pdb.set_trace()
        adv_target=adv_target.to(device)
        DAG_Attack(model=model, label=label_oh, labels=label, masks=mask,
                               model_name = "FIFO", image_name= name, adv_percent_file= adv_percent_file,
                               adv_label = adv_target, inputs=image,interp=interp)
       
        testloader_iteration += 1
        print(testloader_iteration , " image is adversed.")

    adv_percent_file.close()
    return


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


    run_attack(model=model, loader=testloader, device=device)

    print("after creating adversarial_examples")
    