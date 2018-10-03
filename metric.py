import tensorflow as tf


def psnr(x, y, m_val=255):
    return tf.image.psnr(a=x, b=y, max_val=m_val)


def ssim(x, y, m_val=255):
    return tf.image.ssim(img1=x, img2=y, max_val=m_val)


def mse(x, y):
    return tf.metrics.mean_squared_error(labels=x, predictions=y)
