import tensorflow as tf
import tfutil


class RCAN:

    def __init__(self,
                 sess,                 # Tensorflow session
                 batch_size=16,        # batch size
                 reduction=16,         # reduction rate at CA layer
                 ):
        self.sess = sess
        self.batch_size = batch_size

        self.reduction = reduction

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
        with tf.variable_scope("channel_attention-%s" % name):
            x_gap = tfutil.adaptive_global_average_pool_2d(x)

            x = tfutil.conv2d(x_gap, f=f // reduction, k=1, pad='VALID')
            x = tf.nn.relu(x)

            x = tfutil.conv2d(x, f=f, k=1, pad='VALID')
            x = tf.nn.sigmoid(x)
            return x_gap * x
