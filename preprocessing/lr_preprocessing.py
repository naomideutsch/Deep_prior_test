

import argparse
import os
from PIL import Image
import numpy as np






def lr_preprocessing(images_dir, output_dir, size):
    images_paths = os.listdir(images_dir)
    for name in images_paths:
        print(os.path.join(images_dir, name))
        image = Image.open(os.path.join(images_dir, name))
        lr_image = image.resize(size=size)
        lr_image.save(os.path.join(output_dir, name))









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, required=True)
    parser.add_argument('--lr-imgs-dir', type=str, required=True)
    parser.add_argument('--lr-img-size', type=int, nargs=2, default=(128, 128))
    args = parser.parse_args()

    if not os.path.exists(args.lr_imgs_dir):
        os.makedirs(args.lr_imgs_dir)

    lr_preprocessing(args.img_dir, args.lr_imgs_dir, args.lr_img_size)