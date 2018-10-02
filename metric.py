import tensorflow as tf


def psnr(x, y):
    return tf.image.psnr(a=x, b=y, max_val=1.)


def ssim(x, y):
    return tf.image.ssim(img1=x, img2=y, max_val=1.)


def mse(x, y):
    return tf.metrics.mean_squared_error(labels=x, predictions=y)
