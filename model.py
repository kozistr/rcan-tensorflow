import tensorflow as tf
import tfutil


class RCAN:

    def __init__(self,
                 sess,                 # Tensorflow session
                 batch_size=16,        # batch size
                 reduction=16,         # reduction rate at CA layer
                 eps=1.1e-5,
                 ):
        self.sess = sess
        self.batch_size = batch_size

        self.reduction = reduction

        self._eps = eps

    @staticmethod
    def channel_attention(x, f, reduction, name):
        """
        Channel Attention (CA) Layer
        :param x: input layer
        :param f: conv2d filter size
        :param reduction: conv2d filter reduction rate
        :param name: scope name
        :return: output layer
        """
        with tf.variable_scope("CA-%s" % name):
            x_gap = tfutil.adaptive_global_average_pool_2d(x)

            x = tfutil.conv2d(x_gap, f=f // reduction, k=1, pad='VALID')
            x = tf.nn.relu(x)

            x = tfutil.conv2d(x, f=f, k=1, pad='VALID')
            x = tf.nn.sigmoid(x)
            return x_gap * x

    def residual_channel_attention_block(self, x, f, kernel_size, reduction, use_bn, name):
        with tf.variable_scope("RCAB-%s" % name):
            x = tfutil.conv2d(x, f=f, k=kernel_size, pad='VALID')
            x = tf.layers.batch_normalization(epsilon=self._eps) if use_bn else x
            x = tf.nn.relu(x)

            x = tfutil.conv2d(x, f=f, k=kernel_size, pad='VALID')
            res = tf.layers.batch_normalization(epsilon=self._eps) if use_bn else x

            x = self.channel_attention(x, f, reduction, name)
            return res + x

    @staticmethod
    def residual_group(x, f, reduction, name):
        pass
