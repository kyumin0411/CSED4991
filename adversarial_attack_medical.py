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
import pdb
from torch.utils.data.dataset import Dataset
from torch.utils import data
from scipy.stats import rice
from dag_medical import DAG
from dag_utils import generate_target, generate_target_swap
from util import make_one_hot
from configs.test_config_kyumin import get_arguments

from model.refinenetlw import rf_lw101
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.paired_cityscapes import Pairedcityscapes

from optparse import OptionParser
from torch.autograd import Variable

#from model import UNet, SegNet, DenseNet
# from skimage.measure import compare_ssim as ssim

BATCH_SIZE = 10
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def DAG_Attack(model, testloader, num_classes):
    print("DAG Attack Starts")
    # Hyperparamter for DAG 

    num_iterations=500
    gamma=0.5
    num=15    

    adversarial_examples = []
    
    # pdb.set_trace()

    for index, batch in enumerate(testloader):
        image, label, size, name = batch
        label[label==255] = 0
        
        # pdb.set_trace()
        # image = image.unsqueeze(0)
        # image = image.to(device)
        pure_label = label.squeeze(0).numpy()

        image = image.clone().detach().float()
        label = label.clone().detach().float()
        label = label.to(device)
        image = Variable(image).to(device)
        image.requires_grad_()

        # Change labels from [batch_size, height, width] to [batch_size, num_classes, height, width]

        label_oh = make_one_hot(label.long(),n_classes, device)

        unique_label = torch.unique(label)
        target_class = int(random.choice(unique_label[1:]).item())

        adv_target=generate_target(label_oh.cpu().numpy(), target_class = target_class)
        adv_target=torch.from_numpy(adv_target).float()

        adv_target=adv_target.to(device)

        interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')

        pdb.set_trace()
        _, _, _, _, _, adversarial_image=DAG(model=model,
                  model_name="FIFO",
                  image_name=name,
                  image=image,
                  ground_truth=label_oh,
                  adv_target=adv_target,
                  num_iterations=num_iterations,
                  gamma=gamma,
                  interp=interp,
                  no_background=True,
                  background_class=0,
                  device=device,
                  verbose=True,
                  pure_label=pure_label
                  )

        if adversarial_image!=None:

            adversarial_examples.append([adversarial_image,
                                            pure_label])

        del image_iteration


    print('total {} {} images are generated'.format(len(adversarial_examples)))
    
    return adversarial_examples

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
    # pdb.set_trace()
    
    testloader = data.DataLoader(cityscapesDataSet(args.data_dir_city, args.data_city_list, crop_size = (2048, 1024), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                            batch_size=1, shuffle=False, pin_memory=True)


    adversarial_examples = DAG_Attack(model, testloader, n_classes)

    print("after creating adversarial_examples")
    
    pdb.set_trace()
    # save adversarial examples([adversarial examples, labels])
    with open('../data/adversarial_example', 'wb') as fp:
        pickle.dump(adversarial_examples, fp)
