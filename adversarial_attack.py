import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import sys
import os
import os.path as osp
from torch.utils.data.dataset import Dataset
from torch.utils import data
from scipy.stats import rice
from dag import DAG
from dag_utils import generate_target, generate_target_swap
from util import make_one_hot
from configs.test_config_kyumin import get_arguments

from model.refinenetlw import rf_lw101
from dataset.cityscapes_dataset import cityscapesDataSet

from optparse import OptionParser

#from model import UNet, SegNet, DenseNet
# from skimage.measure import compare_ssim as ssim

BATCH_SIZE = 10
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def DAG_Attack(model, testloader):
    print("DAG Attack Starts")
    # Hyperparamter for DAG 
    num_iterations=20
    gamma=0.5
    num=15    
    
    device = torch.device('cuda:0')
    model = model.to(device)
    
    adversarial_examples = []

    for index, batch in enumerate(testloader):
        image, label, size, name = batch

        image = image.unsqueeze(0)
        pure_label = label.squeeze(0).numpy()

        image , label = image.clone().detach().requires_grad_(True).float(), label.clone().detach().float()
        image , label = image.to(device), label.to(device)

        # Change labels from [batch_size, height, width] to [batch_size, num_classes, height, width]

        label_oh = make_one_hot(label.long(),n_classes, device)

        unique_label = torch.unique(label)
        target_class = int(random.choice(unique_label[1:]).item())

        adv_target=generate_target(label_oh.cpu().numpy(), target_class = target_class)
        adv_target=torch.from_numpy(adv_target).float()

        adv_target=adv_target.to(device)

        _, _, _, _, _, image_iteration=DAG(model=model,
                  image=image,
                  ground_truth=label_oh,
                  adv_target=adv_target,
                  num_iterations=num_iterations,
                  gamma=gamma,
                  no_background=True,
                  background_class=0,
                  device=device,
                  verbose=False)

        if len(image_iteration) >= 1:

            adversarial_examples.append([image_iteration[-1],
                                         pure_label])

        del image_iteration


    print('total {} {} images are generated'.format(len(adversarial_examples)))
    
    return adversarial_examples

if __name__ == "__main__":
    print("start adversarial_attack.py")

    args = get_arguments()

    print("parsing arguments is done")

    restore = torch.load(args.restore_from)

    print("restoring loading from previous model is done")

    model = rf_lw101(num_classes=args.num_classes)

    print("generating model is done")

    model.load_state_dict(restore['state_dict'])
    start_iter = 0

    print("loading the model is done")

    
    n_classes = args.num_classes

    save_dir_adversarial = osp.join(f'./result_adversarial', args.file_name)
    
    if not os.path.exists(save_dir_adversarial):
        os.makedirs(save_dir_adversarial)

    print("making save directory path is done")

    model.eval()


    print("evaluating model is done")

    testloader = data.DataLoader(cityscapesDataSet(args.data_dir_city, args.data_city_list, crop_size = (2048, 1024), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                            batch_size=1, shuffle=False, pin_memory=True)

    print("generate data loader")
    print("test image of data loader")

    for index, batch in enumerate(testloader):
        print(index)
        image, label, size, name = batch
        # print (image)

    print("before DAG_Attack")

    adversarial_examples = DAG_Attack(model, testloader)

    print("after creating adversarial_examples")
        
    # save adversarial examples([adversarial examples, labels])
    with open('../data/adversarial_example', 'wb') as fp:
        pickle.dump(adversarial_examples, fp)
