import tensorflow as tf


SEED = 1337

tf.set_random_seed(SEED)

w_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
b_init = tf.zeros_initializer()

reg = 5e-4
w_reg = tf.contrib.layers.l2_regularizer(reg)


# ---------------------------------------------------------------------------------------------
# Functions


def adaptive_global_average_pool_2d(x):
    """
    In the paper, using gap which output size is 1, so i just gap func :)
    :param x: 4d-tensor, (batch_size, height, width, channel)
    :return: 2d-tensor, (batch_size, 1)
    """
    return tf.reduce_mean(x, axis=[1, 2, 3])


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


def pixel_shuffle(x, f, scaling_factor):
    x = conv2d(x, f)
    return x


# ---------------------------------------------------------------------------------------------
# Activation Functions


def relu(x):
    return tf.nn.relu


def leaky_relu(x, alpha=.2):
    return tf.nn.leaky_relu(x, alpha)


def elu(x):
    return tf.nn.elu(x)
