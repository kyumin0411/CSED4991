from configs.test_config_kyumin import get_arguments
import numpy as np
from PIL import Image
from os.path import join
import warnings
import pdb

warnings.filterwarnings("ignore")

def fast_hist(a, b, n):
    # import pdb; pdb.set_trace()
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int)+ b[k], minlength=n ** 2).reshape(n, n) #


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

def compute_mIoU(root,list_path):
    label = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "light",
    "sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motocycle",
    "bicycle"]

    label2train=[
    [0, 255],
    [1, 255],
    [2, 255],
    [3, 255],
    [4, 255],
    [5, 255],
    [6, 255],
    [7, 0],
    [8, 1],
    [9, 255],
    [10, 255],
    [11, 2],
    [12, 3],
    [13, 4],
    [14, 255],
    [15, 255],
    [16, 255],
    [17, 5],
    [18, 255],
    [19, 6],
    [20, 7],
    [21, 8],
    [22, 9],
    [23, 10],
    [24, 11],
    [25, 12],
    [26, 13],
    [27, 14],
    [28, 15],
    [29, 255],
    [30, 255],
    [31, 16],
    [32, 17],
    [33, 18],
    [-1, 255]]    

    num_classes = 19
    hist = np.zeros((num_classes, num_classes))
    # pdb.set_trace()
    img_ids = [i_id.strip() for i_id in open(list_path)]

    for name in img_ids:
        image_name = name.split('/')[1]
        pred_file = join(root, "Cityscape_adversarial_attack/Cityscape_color_image/%s" % ("Cityscape_colored_" + image_name))
        gt_file = join(root, "Cityscape_adversarial_attack/color_image_handle_image/%s" % ("original_colored_" + image_name))

        pred = np.array(Image.open(pred_file))
        label = np.array(Image.open(gt_file))
        
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_file, pred_file[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    # pdb.set_trace()
    mIoUs = per_class_iu(hist)

    print('Evaluation on Cityscapes lindau 40')
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    miou = float(str(round(np.nanmean(mIoUs) * 100, 2)))
    return miou


def miou(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir, args.dataset)

if __name__ == "__main__":
    # print("start adversarial_mIoU.py")

    args = get_arguments()

    print("Calculate mIoU")
    print("--- Model : Cityscape")
    print("--- Inputs : original")
    mIoU = compute_mIoU(args.adversarial_data_dir, args.data_city_list)
    # print ("result mIoU : ", mIoU)