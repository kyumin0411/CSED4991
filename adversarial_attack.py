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
from torch.utils.data.dataset import Dataset
from torch.utils import data
#from model import UNet, SegNet, DenseNet

#from dataset import SampleDataset
from scipy.stats import rice
# from skimage.measure import compare_ssim as ssim
from dag import DAG
from dag_utils import generate_target, generate_target_swap
from util import make_one_hot
from configs.test_config_kyumin import get_arguments
from model.refinenetlw import rf_lw101

from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.Foggy_Zurich_test import foggyzurichDataSet
from dataset.foggy_driving import foggydrivingDataSet

from optparse import OptionParser

BATCH_SIZE = 10
RESTORE_FROM = 'without_pretraining'
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

class SampleDataset(Dataset):
    
    def __init__(self, root_dir):
    
        self.data_path = root_dir

        self.images = []
        self.labels = []
        
        # data form [images, labels]
        with open (self.data_path, 'rb') as fp:
            data = pickle.load(fp)
        
        for i in range(len(data)):
            
            self.images.append(data[i][0])
            self.labels.append(data[i][1])
             

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        image = self.images[index]
        labels = self.labels[index]
        
        torch_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        image = torch_transform(image)
        labels = torch_transform(labels)
            
        return (image, labels)

#def load_data(args):
    
    # data_path = args.data_path
    #data_path = './dataset/cityscapes_list/clear_lindau.txt'
    #n_classes = args.classes
    # data_width = args.width
    # data_height = args.height
    
    # generate loader
    #test_dataset = SampleDataset(data_path)
    
    #test_loader = DataLoader(
    #    test_dataset,
    #    batch_size=BATCH_SIZE,
    #    num_workers=4,
    #)
    
    #print('test_dataset : {}, test_loader : {}'.format(len(test_dataset), len(test_loader)))
    
    
    #return test_dataset, test_loader

# generate Rician noise examples
# Meausre the difference between original and adversarial examples by using structural Similarity (SSIM). 
# The adversarial examples which has SSIM value from 0.97 to 0.99 can be passed.
# SSIM adapted from https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
def Rician(test_dataset):
    
    def generate_rician(image):
    
        ssim_noise = 0

        if torch.is_tensor(image):
            image = image.numpy()

        rician_image = np.zeros_like(image)

        while ssim_noise <= 0.97 or ssim_noise >= 0.99:
            b = random.uniform(0, 1)
            rv = rice(b)
            rician_image = rv.pdf(image)
            ssim_noise =  ssim(image[0], rician_image[0], data_range=rician_image[0].max() - rician_image[0].min())

        #print('ssim : {:.2f}'.format(ssim_noise))

        return rician_image

    adversarial_examples = []
    
    for batch_idx in range(len(test_dataset)):
        
        image, labels = test_dataset.__getitem__(batch_idx)

        rician_image = generate_rician(image)

        #print("image {} save".format(batch_idx))

        adversarial_examples.append([rician_image[0],labels.squeeze(0).numpy()])

    print('total {} Rician noise images are generated'.format(len(adversarial_examples)))
    
    return adversarial_examples


def DAG_Attack(model, test_dataset, args):
    
    # Hyperparamter for DAG 
    
    num_iterations=20
    gamma=0.5
    num=15
    
    gpu = args.gpu
    
    # set device configuration
    device_ids = []
    
    if gpu == 'gpu' :
        
        if not torch.cuda.is_available() :
            print("No cuda available")
            raise SystemExit
            
        device = torch.device(args.device1)
        
        device_ids.append(args.device1)
        
        if args.device2 != -1 :
            device_ids.append(args.device2)
            
        if args.device3 != -1 :
            device_ids.append(args.device3)
        
        if args.device4 != -1 :
            device_ids.append(args.device4)
        
    
    else :
        device = torch.device("cpu")
    
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids = device_ids)
        
    model = model.to(device)
    
    adversarial_examples = []
    
    for batch_idx in range(len(test_dataset)):
        image, label = test_dataset.__getitem__(batch_idx)

        image = image.unsqueeze(0)
        pure_label = label.squeeze(0).numpy()

        image , label = image.clone().detach().requires_grad_(True).float(), label.clone().detach().float()
        image , label = image.to(device), label.to(device)

        # Change labels from [batch_size, height, width] to [batch_size, num_classes, height, width]
        label_oh=make_one_hot(label.long(),n_classes,device)

        #if args.attacks == 'DAG_A':

        #    adv_target = torch.zeros_like(label_oh)

        #elif args.attacks == 'DAG_B':

        #    adv_target=generate_target_swap(label_oh.cpu().numpy())
        #    adv_target=torch.from_numpy(adv_target).float()

        #elif args.attacks == 'DAG_C':
            
            # choice one randome particular class except background class(0)
        unique_label = torch.unique(label)
        target_class = int(random.choice(unique_label[1:]).item())

        adv_target=generate_target(label_oh.cpu().numpy(), target_class = target_class)
        adv_target=torch.from_numpy(adv_target).float()

        #else :
        #    print("wrong adversarial attack types : must be DAG_A, DAG_B, or DAG_C")
        #    raise SystemExit


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

    args = get_arguments()
    
    if args.restore_from == RESTORE_FROM:
        start_iter = 0
        model = rf_lw101(num_classes=args.num_classes)

    else:
        restore = torch.load(args.restore_from)
        model = rf_lw101(num_classes=args.num_classes)

        model.load_state_dict(restore['state_dict'])
        start_iter = 0

    
    model.eval()
    device = torch.device('cuda:0')
    model.to(device)

    testloader1 = data.DataLoader(foggyzurichDataSet(args.data_dir_eval, args.data_list_eval, crop_size=(1152, 648), mean=IMG_MEAN),
                                    batch_size=1, shuffle=False, pin_memory=True)
    
    # if args.attacks == 'Rician':
        
    #     adversarial_examples = Rician(test_dataset)
        
    #     if args.attack_path is None:
            
    #         adversarial_path = 'data/' + args.attacks + '.pickle'
            
    #     else:
            
    #         adversarial_path = args.attack_path
        
    # else:
        

    adversarial_examples = DAG_Attack(model, testloader1, args)

        
    # save adversarial examples([adversarial examples, labels])
    with open('./data/adversarial_example', 'wb') as fp:
        pickle.dump(adversarial_examples, fp)
