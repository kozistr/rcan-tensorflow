import tensorflow as tf
import numpy as np
import tfutil
import metric


class RCAN:

    def __init__(self,
                 sess,                               # tensorflow session
                 batch_size=16,                      # batch size
                 n_channel=3,                        # number of image channel, 3 for RGB, 1 for gray-scale
                 img_scaling_factor=4,               # image scale factor to up
                 lr_img_size=(48, 48),               # input patch image size for LR
                 hr_img_size=(192, 192),             # input patch image size for HR
                 n_res_blocks=20,                    # number of residual block
                 n_res_groups=10,                    # number of residual group
                 res_scale=1,                        # scaling factor of res block
                 n_filters=64,                       # number of conv2d filter size
                 kernel_size=3,                      # number of conv2d kernel size
                 activation='relu',                  # activation function
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
                 tf_log="./model/",                  # path saved tensor summary/model
                 n_gpu=1,                            # number of GPU
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
        self.activation = activation
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

        self.tf_log = tf_log

        self.n_gpu = n_gpu

        self.act = None
        self.opt = None
        self.train_op = None
        self.loss = None
        self.output = None

        self.psnr = None
        self.ssim = None

        self.saver = None
        self.best_saver = None
        self.merged = None
        self.writer = None

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # tensor placeholder for input
        self.x_lr = tf.placeholder(tf.float32, shape=(None,) + self.lr_img_size, name='x-lr-img')
        self.x_hr = tf.placeholder(tf.float32, shape=(None,) + self.hr_img_size, name='x-hr-img')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        # setting stuffs
        self.setup()

        # build a network
        self.build_model()

    def setup(self):
        # Activation Function Setting
        if self.activation == 'relu':
            self.act = tf.nn.relu
        elif self.activation == 'leaky_relu':
            self.act = tf.nn.leaky_relu
        elif self.activation == 'elu':
            self.act = tf.nn.elu
        else:
            raise NotImplementedError("[-] Not supported activation function (%s)" % self.activation)

        # Optimizer
        if self.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr,
                                              beta1=self.beta1, beta2=self.beta2, epsilon=self.opt_eps)
        elif self.optimizer == 'sgd':  # gonna use mm opt actually
            self.opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum)
            # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        else:
            raise NotImplementedError("[-] Not supported optimizer (%s)" % self.optimizer)

    def image_processing(self, x, sign, name):
        with tf.variable_scope(name):
            rgb_mean = tf.convert_to_tensor((sign * self.rgb_mean[0] * 255.,
                                             sign * self.rgb_mean[1] * 255.,
                                             sign * self.rgb_mean[2] * 255.),
                                            dtype=tf.float32)
            x = tfutil.mean_shift(x, rgb_mean=rgb_mean)
            return x

    def channel_attention(self, x, f, reduction, name):
        """
        Channel Attention (CA) Layer
        :param x: input layer
        :param f: conv2d filter size
        :param reduction: conv2d filter reduction rate
        :param name: scope name
        :return: output layer
        """
        with tf.variable_scope("CA-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            x = tfutil.adaptive_global_average_pool_2d(x)

            x = tfutil.conv2d(x, f=f // reduction, k=1, name="conv2d-1")
            x = self.act(x)

            x = tfutil.conv2d(x, f=f, k=1, name="conv2d-2")
            x = tf.nn.sigmoid(x)
            return skip_conn * x

    def residual_channel_attention_block(self, x, f, kernel_size, reduction, use_bn, name, is_train=True):
        with tf.variable_scope("RCAB-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            x = tfutil.conv2d(x, f=f, k=kernel_size, name="conv2d-1")
            x = tf.layers.BatchNormalization(epsilon=self._eps, trainable=is_train, name="bn-1")(x) if use_bn else x
            x = self.act(x)

            x = tfutil.conv2d(x, f=f, k=kernel_size, name="conv2d-2")
            x = tf.layers.BatchNormalization(epsilon=self._eps, trainable=is_train, name="bn-2")(x) if use_bn else x

            x = self.channel_attention(x, f, reduction, name="RCAB-%s" % name)
            return skip_conn + self.res_scale * x

    def residual_group(self, x, f, kernel_size, reduction, use_bn, name, is_train=True):
        with tf.variable_scope("RG-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            for i in range(self.n_res_blocks):
                x = self.residual_channel_attention_block(x, f, kernel_size, reduction, use_bn, name=str(i),
                                                          is_train=is_train)

            x = tfutil.conv2d(x, f=f, k=kernel_size)
            return skip_conn + x

    def up_scaling(self, x, f, scale_factor, name):
        """
        :param x: image
        :param f: conv2d filter
        :param scale_factor: scale factor
        :param name: scope name
        :return:
        """
        with tf.variable_scope(name):
            if scale_factor == 3:
                x = tfutil.conv2d(x, f * 9, k=1, name='conv2d-image_scaling-0')
                x = tfutil.pixel_shuffle(x, 3)
            elif scale_factor & (scale_factor - 1) == 0:  # is it 2^n?
                log_scale_factor = int(np.log2(scale_factor))
                for i in range(log_scale_factor):
                    x = tfutil.conv2d(x, f * 4, k=1, name='conv2d-image_scaling-%d' % i)
                    x = tfutil.pixel_shuffle(x, 2)
            else:
                raise NotImplementedError("[-] Not supported scaling factor (%d)" % scale_factor)
            return x

    def residual_channel_attention_network(self, x, f, kernel_size, reduction, use_bn, scale,
                                           is_train=True, reuse=False, gpu_idx=0):
        with tf.variable_scope("Residual_Channel_Attention_Network-gpu%d" % gpu_idx, reuse=reuse):
            x = self.image_processing(x, sign=-1, name='pre-processing')

            # 1. head
            head = tfutil.conv2d(x, f=f, k=kernel_size, name="conv2d-head")

            # 2. body
            x = head
            for i in range(self.n_res_groups):
                x = self.residual_group(x, f, kernel_size, reduction, use_bn, name=str(i), is_train=is_train)

            body = tfutil.conv2d(x, f=f, k=kernel_size, name="conv2d-body")
            body += head

            # 3. tail
            x = self.up_scaling(body, f, scale, name='up-scaling')
            tail = tfutil.conv2d(x, f=self.n_channel, k=kernel_size, name="conv2d-tail")  # (-1, 384, 384, 3)

            x = self.image_processing(tail, sign=1, name='post-processing')
            return x

    def build_model(self):
        # RCAN model
        self.output = self.residual_channel_attention_network(x=self.x_lr,
                                                              f=self.n_filters,
                                                              kernel_size=self.kernel_size,
                                                              reduction=self.reduction,
                                                              use_bn=self.use_bn,
                                                              scale=self.img_scale,
                                                              is_train=self.is_train
                                                              )
        self.output = tf.clip_by_value(self.output, 0, 255)

        # l1 loss
        self.loss = tf.reduce_mean(tf.abs(self.output - self.x_hr))

        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        # metrics
        self.psnr = tf.reduce_mean(metric.psnr(self.output, self.x_hr, m_val=255))
        self.ssim = tf.reduce_mean(metric.ssim(self.output, self.x_hr, m_val=255))

        # summaries
        tf.summary.scalar("loss/l1_loss", self.loss)
        tf.summary.scalar("metric/psnr", self.psnr)
        tf.summary.scalar("metric/ssim", self.ssim)
        tf.summary.scalar("misc/lr", self.lr)

        # merge summary
        self.merged = tf.summary.merge_all()

        # model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.best_saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.tf_log, self.sess.graph)
