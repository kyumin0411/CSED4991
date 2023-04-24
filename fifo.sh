#!/bin/sh

git pull

conda activate fifo

python adversarial_attack.py --file-name "adversarial_cityscapes" --restore-from FIFO_final_model.pth