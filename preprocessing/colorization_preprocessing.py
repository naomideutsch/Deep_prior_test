import argparse
import os
from PIL import Image
import numpy as np
import utils
import tensorflow as tf


def gray_preprocessing(args):
    images_paths = os.listdir(args.img_dir)
    for name in images_paths:
        image = Image.open(os.path.join(args.img_dir, name))
        tf_image = tf.convert_to_tensor(np.array(image).astype(np.float32))
        gray_image = utils.convert_to_gray(tf_image[None])[0]
        blurred = Image.fromarray(np.uint8(tf.Session().run(gray_image)))
        blurred.save(os.path.join(args.output_dir, name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    gray_preprocessing(args)
