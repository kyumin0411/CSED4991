import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import pdb
from torch.utils import data
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

class adversarialDataSet(data.Dataset):

    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))
        for name in self.img_ids:
            # pdb.set_trace()
            image_name = name.split('/')[1]
            img_file = osp.join(self.root, "orig_image/%s" % ("Cityscape_" + image_name))
            label_file = osp.join(self.root, "cropped_label/%s" % (image_name[:-4]+'.npy'))
            #pdb.set_trace()
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_name
            })
        

    def __len__(self):
        return len(self.files)
    

    def _apply_transform(self, image, lbl, scale=(0.7, 1.3), crop_size=600):
        (W, H) = image.size[:2]
        if isinstance(scale, tuple):
            scale = random.random() * 0.6 + 0.7

        tsfrms = []
        tsfrms.append(transforms.Resize((int(H * scale), int(W * scale))))
        tsfrms = transforms.Compose(tsfrms)

        return tsfrms(image), tsfrms(lbl)
    
    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def __getitem__(self, index):
        datafiles = self.files[index]
        pdb.set_trace()
        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.load(datafiles["label"])
        label = torch.from_numpy(label)
        # label = Image.open(datafiles["label"])
        name = datafiles["name"]

        image = np.asarray(image, np.float32)

        # pdb.set_trace()
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label, np.array(size), name


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
