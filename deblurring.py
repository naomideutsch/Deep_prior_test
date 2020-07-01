

import argparse
import pickle
import os
import imageio
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from preprocessing import utils

import os
import sys
file_dir = os.path.dirname("stylegan/")
sys.path.append(file_dir)


import dnnlib
import dnnlib.tflib as tflib
import config

from perceptual_model import PerceptualModel

STYLEGAN_MODEL_URL = '/content/drive/My Drive/Colab Notebooks/styleGAN_weights/karras2019stylegan-ffhq-1024x1024.pkl'


def get_gradient_reg(image):
    dx, dy = tf.image.image_gradients(image)
    sx = tf.reduce_mean(tf.pow(dx,2))
    sy = tf.reduce_mean(tf.pow(dy,2))
    return sx + sy

def get_l2_reg(image):
    print(tf.nn.l2_loss(image))
    return tf.reduce_mean(tf.nn.l2_loss(image))

def get_reg_by_name(name):
    if name == "grad":
        return lambda image: get_gradient_reg(image)
    if name == "l2":
        return lambda image: get_l2_reg(image)



def optimize_latent_codes(args):
    tflib.init_tf()

    reg = get_reg_by_name(args.reg)

    blur_func = utils.get_blur(args.kernel_size, args.sigma, type=args.kernel_type)

    with open(STYLEGAN_MODEL_URL, "rb") as f:
        _G, _D, Gs = pickle.load(f)

    latent_code = tf.get_variable(
		name='latent_code', shape=(1, 18, 512), dtype='float32', initializer=tf.initializers.zeros()
	)

    generated_img = Gs.components.synthesis.get_output_for(latent_code, randomize_noise=False)
    generated_img = tf.transpose(generated_img, [0, 2, 3, 1])
    generated_img = ((generated_img + 1) / 2) * 255
    generated_img = tf.image.resize_images(generated_img, tuple(args.input_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    generated_blurred_img = blur_func(generated_img)
    generated_img_for_display = tf.saturate_cast(generated_img, tf.uint8)

    blr_img = tf.placeholder(tf.float32, [None, args.input_size[0], args.input_size[1], 3])

    perceptual_model = PerceptualModel(img_size=args.input_size)
    generated_img_features = perceptual_model(generated_blurred_img)
    target_img_features = perceptual_model(blr_img)



    loss_op = tf.reduce_mean(tf.abs(generated_img_features - target_img_features))

    if reg != None:
        loss_op += args.beta * reg(generated_img)




    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(loss_op, var_list=[latent_code])

    sess = tf.get_default_session()

    img_names = sorted(os.listdir(args.blurred_imgs_dir))
    for img_name in img_names:
        img = imageio.imread(os.path.join(args.blurred_imgs_dir, img_name))

        sess.run(tf.variables_initializer([latent_code] + optimizer.variables()))

        progress_bar_iterator = tqdm(
            iterable=range(args.total_iterations),
            bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}',
            desc=img_name
        )

        for i in progress_bar_iterator:
            loss, _ = sess.run(
                fetches=[loss_op, train_op],
                feed_dict={
                    blr_img: img[np.newaxis, ...]
                }
            )

            progress_bar_iterator.set_postfix_str('loss=%.2f' % loss)

        deblurred_imgs, latent_codes = sess.run(
            fetches=[generated_img_for_display, latent_code],
            feed_dict={
                blr_img: img[np.newaxis, ...]
            }
        )

        imageio.imwrite(os.path.join(args.deblurred_imgs_dir, img_name), deblurred_imgs[0])
        np.savez(file=os.path.join(args.latents_dir, img_name + '.npz'), latent_code=latent_codes[0])






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blurred-imgs-dir', type=str, required=True)
    parser.add_argument('--deblurred-imgs-dir', type=str, required=True)
    parser.add_argument('--latents-dir', type=str, required=True)

    parser.add_argument('--input-size', type=int, nargs=2, default=(128, 128))
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--reg', default="l2")



    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--kernel-type', default="gauss")

    parser.add_argument('--sigma', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--total-iterations', type=int, default=1000)

    args = parser.parse_args()


    os.makedirs(args.deblurred_imgs_dir, exist_ok=True)
    os.makedirs(args.latents_dir, exist_ok=True)

    optimize_latent_codes(args)
