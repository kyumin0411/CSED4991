from configs.test_config_kyumin import get_arguments
from model.refinenetlw import rf_lw101
from dataset.adversarial_dataset import adversarialDataSet
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
import pdb

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


if __name__ == "__main__":
    print("start adversarial_evaluate.py")

    args = get_arguments()
    restore = torch.load(args.restore_from)

    print("restoring loading from previous model is done")

    model = rf_lw101(num_classes=args.num_classes)

    model.load_state_dict(restore['state_dict'])
    start_iter = 0

    n_classes = args.num_classes

    model.eval()

    device = torch.device("cuda:0")

    model = model.to(device)
    testloader = data.DataLoader(adversarialDataSet(args.adversarial_data_dir, args.data_city_list, crop_size = (600, 600), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                            batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(600, 600), mode='bilinear')

    testloader_iteration = 0

    for index, batch in enumerate(testloader):
        image, label, size, name = batch
        # pdb.set_trace()
        # mask = label < n_classes  
        # label = label * mask
        label = label.numpy()
        label = label[0][0]
        # output_feature5 = model(Variable(image).cuda(args.gpu))[5]
        # output = interp(output_feature5)

        # output = torch.mean(output, dim=0)
        # output = output.cpu().detach().numpy()
        # output = output.transpose(1,2,0)
        # output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        # output_col = colorize_mask(output)
        # output_col = colorize_mask(label)

        # name = name[0].split('/')[-1]
        color_path = "../data/adversarial/Cityscape_adversarial_attack/color_image/" + "original_colored_" + name
        output_col.save(color_path)
        testloader_iteration += 1
        print(testloader_iteration, " is colored.")
           