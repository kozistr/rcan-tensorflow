import tensorflow as tf
import numpy as np

from config import get_config


# Configuration
config, _ = get_config()

SEED = config.seed

tf.set_random_seed(SEED)

# ---------------------------------------------------------------------------------------------
# Initializer & Regularizer

w_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
b_init = tf.zeros_initializer()

reg = config.l2_reg
w_reg = tf.contrib.layers.l2_regularizer(reg)


# ---------------------------------------------------------------------------------------------
# Functions


def adaptive_global_average_pool_2d(x):
    """
    In the paper, using gap which output size is 1, so i just gap func :)
    :param x: 4d-tensor, (batch_size, height, width, channel)
    :return: 4d-tensor, (batch_size, 1, 1, channel)
    """
    c = x.get_shape()[-1]
    return tf.reshape(tf.reduce_mean(x, axis=[1, 2]), (-1, 1, 1, c))


def conv2d(x, f=64, k=3, s=1, pad='SAME', use_bias=True, reuse=None, name='conv2d'):
    """
    :param x: input
    :param f: filters
    :param k: kernel size
    :param s: strides
    :param pad: padding
    :param use_bias: using bias or not
    :param reuse: reusable
    :param name: scope name
    :return: output
    """
    return tf.layers.conv2d(inputs=x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=w_init,
                            kernel_regularizer=w_reg,
                            bias_initializer=b_init,
                            padding=pad,
                            use_bias=use_bias,
                            reuse=reuse,
                            name=name)


def pixel_shuffle(x, scaling_factor):
    # pixel_shuffle
    # (batch_size, h, w, c * r^2) to (batch_size, h * r, w * r, c)
    sf = scaling_factor

    _, h, w, c = x.get_shape()
    c //= sf ** 2

    x = tf.split(x, scaling_factor, axis=-1)
    x = tf.concat(x, 2)

    x = tf.reshape(x, (-1, h * scaling_factor, w * scaling_factor, c))
    return x


def mean_shift(x, rgb_mean, f=3, k=1, s=1, pad='SAME', name='mean_shift'):
    with tf.variable_scope(name):
        weight = tf.get_variable(shape=[k, k, f, f], initializer=tf.constant_initializer(np.eye(f)),
                                 trainable=False, name='ms_weight')
        bias = tf.get_variable(shape=[f], initializer=tf.constant_initializer(rgb_mean),
                               trainable=False, name='ms_bias')

        x = tf.nn.conv2d(x, weight, strides=[1, s, s, 1], padding=pad, name='ms_conv2d')
        x = tf.nn.bias_add(x, bias)
        return x


# ---------------------------------------------------------------------------------------------
# Gradients (for supporting multi-gpu in tensorflow)


def average_gradients(grads):
    average_grads = []
    for grad_and_vars in zip(*grads):
        grads = [tf.expand_dims(g, axis=0) for g, _ in grad_and_vars]

        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)
    return average_grads
