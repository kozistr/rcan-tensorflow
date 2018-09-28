import tensorflow as tf
import tfutil


class RCAN:

    def __init__(self,
                 sess,                 # Tensorflow session
                 batch_size=16,        # batch size
                 n_res_blocks=4,       # number of residual block
                 res_scale=1,          # scaling factor of res block
                 reduction=16,         # reduction rate at CA layer
                 eps=1.1e-5,
                 ):
        self.sess = sess
        self.batch_size = batch_size

        self.n_res_blocks = n_res_blocks
        self.res_scale = res_scale

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
            x = tfutil.conv2d(x, f=f, k=kernel_size, pad='VALID', name="conv2d-1")
            x = tf.layers.batch_normalization(epsilon=self._eps, name="bn-1") if use_bn else x
            x = tf.nn.relu(x)

            x = tfutil.conv2d(x, f=f, k=kernel_size, pad='VALID', name="conv2d-2")
            res = tf.layers.batch_normalization(epsilon=self._eps, name="bn-2") if use_bn else x

            x = self.channel_attention(x, f, reduction, name="RCAB-%s" % name)
            return self.res_scale * res + x

    def residual_group(self, x, f, kernel_size, reduction, use_bn, name):
        with tf.variable_scope("RG-%s" % name):
            for i in range(self.n_res_blocks):
                x = self.residual_channel_attention_block(x, f, kernel_size, reduction, use_bn, name=str(i))

            res = tfutil.conv2d(x, f=f, k=kernel_size, pad='VALID')
            return res + x


