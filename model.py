import tensorflow as tf
import numpy as np
import tfutil


class RCAN:

    def __init__(self,
                 sess,                               # tensorflow session
                 batch_size=16,                      # batch size
                 n_channel=3,                        # number of image channel, 3 for RGB, 1 for gray-scale
                 img_scaling_factor=4,               # image scale factor to up
                 lr_img_size=(96, 96),               # input image size for LR
                 hr_img_size=(384, 384),             # input image size for HR
                 n_res_blocks=20,                    # number of residual block
                 n_res_groups=10,                    # number of residual group
                 res_scale=1,                        # scaling factor of res block
                 n_filters=64,                       # number of conv2d filter size
                 kernel_size=3,                      # number of conv2d kernel size
                 act=tf.nn.relu,                     # activation function
                 use_bn=False,                       # using batch_norm or not
                 reduction=16,                       # reduction rate at CA layer
                 rgb_mean=(0.4488, 0.4371, 0.4040),  # RGB mean, for DIV2K DataSet
                 rgb_std=(1., 1., 1.),               # RGB std, for DIV2K DataSet
                 optimizer='adam',                   # name of optimizer
                 lr=1e-4,                            # learning rate
                 lr_decay=.5,                        # learning rate decay factor
                 lr_decay_step=2e5,                  # learning rate decay step
                 momentum=.9,                        # SGD momentum value
                 beta1=.9,                           # Adam beta1 value
                 beta2=.999,                         # Adam beta2 value
                 opt_eps=1e-8,                       # Adam epsilon value
                 eps=1.1e-5,                         # epsilon
                 ):
        self.sess = sess
        self.batch_size = batch_size
        self.n_channel = n_channel
        self.img_scale = img_scaling_factor
        self.lr_img_size = lr_img_size + (self.n_channel,)
        self.hr_img_size = hr_img_size + (self.n_channel,)

        self.n_res_blocks = n_res_blocks
        self.n_res_groups = n_res_groups
        self.res_scale = res_scale

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.act = act
        self.use_bn = use_bn
        self.reduction = reduction

        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_step = lr_decay_step
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.opt_eps = opt_eps

        self._eps = eps

        self.opt = None
        self.loss = None

        # tensor placeholder for input
        self.x_lr = tf.placeholder(tf.float32, shape=[None] + self.lr_img_size, name='x-lr-img')
        self.x_hr = tf.placeholder(tf.float32, shape=[None] + self.hr_img_size, name='x-hr-img')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        # RCAN model
        self.model = self.residual_channel_attention_network(x=self.x_lr,
                                                             f=self.n_filters,
                                                             kernel_size=self.kernel_size,
                                                             reduction=self.reduction,
                                                             use_bn=self.use_bn,
                                                             scale=self.img_scale)

        # build a network
        self.build_model()

    def image_pre_process(self, x):
        r, g, b = tf.split(x, 3, 3)
        bgr = tf.concat([b - self.rgb_mean[0],
                         g - self.rgb_mean[1],
                         r - self.rgb_mean[2]], axis=3)
        return bgr

    def image_post_process(self, x):
        b, g, r = tf.split(x, 3, 3)
        rgb = tf.concat([r + self.rgb_mean[2],
                         g + self.rgb_mean[1],
                         b + self.rgb_mean[0]], axis=3)
        return rgb

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

    def image_scaling(self, x, f, scale_factor):
        """
        :param x: image
        :param scale_factor: scale factor
        :return:
        """
        if scale_factor == 3:
            x = tfutil.pixel_shuffle(x, f * 9, 3)
        elif scale_factor & (scale_factor - 1) == 0:  # is it 2^n?
            log_scale_factor = int(np.log2(scale_factor))
            for i in range(log_scale_factor):
                x = tfutil.pixel_shuffle(x, f * 4, 2)
        else:
            raise NotImplementedError("[-] Not supported scaling factor (%d)" % scale_factor)
        return x

    def residual_channel_attention_network(self, x, f, kernel_size, reduction, use_bn, scale):
        with tf.variable_scope("Residual_Channel_Attention_Network"):
            x = self.image_pre_process(x)

            # head
            x = tfutil.conv2d(x, f=f, k=kernel_size, pad='VALID', name="conv2d-head")
            head = tf.nn.relu(x)

            # body
            x = head
            for i in range(self.n_res_groups):
                x = self.residual_group(x, f, kernel_size, reduction, use_bn, name=str(i))

            x = tfutil.conv2d(x, f=f, k=kernel_size, pad='VALID', name="conv2d-body")
            body = tf.nn.relu(x)
            body += head

            # tail
            x = body
            x = self.image_scaling(x, f, scale)
            x = tfutil.conv2d(x, f=f, k=kernel_size, pad='VALID', name="conv2d-tail")
            tail = tf.nn.relu(x)

            x = self.image_post_process(tail)
            return x

    def build_model(self):
        # l1 loss
        self.loss = tf.reduce_mean(tf.abs(self.model - self.x_hr))
