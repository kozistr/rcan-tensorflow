from dataset import Div2KDataSet as DataSet
from dataset import DataIterator
from config import get_config

import tensorflow as tf
import numpy as np
import argparse
import time

import model
import util


# Argument
parser = argparse.ArgumentParser()
parser.add_argument('--resize_to', type=int, default=0)
args = parser.parse_args()

resize_to = args.resize_to

# Configuration
config, _ = get_config()

np.random.seed(config.seed)
tf.set_random_seed(config.seed)


def main():
    start_time = time.time()  # Clocking start

    # Div2K - Track 1: Bicubic downscaling - x4 DataSet load
    """
    ds = DataSet(ds_path=config.data_dir,
                 ds_name="X4",
                 use_save=True,
                 save_type="to_h5",
                 save_file_name=config.data_dir + "DIV2K",
                 use_img_scale=True)
    """
    ds = DataSet(ds_hr_path=config.data_dir + "DIV2K-hr.h5",
                 ds_lr_path=config.data_dir + "DIV2K-lr.h5",
                 use_img_scale=True)

    hr, lr = ds.hr_images, ds.lr_images  # [0, 1] scaled images

    lr_shape = (ds.lr_height, ds.lr_width, ds.channel)
    hr_shape = (ds.hr_height, ds.hr_width, ds.channel)

    lr = np.reshape(lr, (-1,) + lr_shape)
    hr = np.reshape(hr, (-1,) + hr_shape)

    if not resize_to == 0:
        import cv2

        lr_shape = (resize_to, resize_to, ds.channel)
        hr_shape = (resize_to * config.image_scaling_factor, resize_to * config.image_scaling_factor, ds.channel)

        new_lr = np.zeros((ds.n_images,) + lr_shape, dtype=np.float32)
        new_hr = np.zeros((ds.n_images,) + hr_shape, dtype=np.float32)
        for idx in range(ds.n_images):
            new_lr[idx] = cv2.resize(lr[idx], lr_shape[:-1], cv2.INTER_CUBIC)
            new_hr[idx] = cv2.resize(hr[idx], hr_shape[:-1], cv2.INTER_LINEAR)

        hr = new_hr
        lr = new_lr

    print("[+] Loaded LR image ", lr_shape)
    print("[+] Loaded HR image ", hr_shape)

    # DataIterator
    di = DataIterator(lr, hr, config.batch_size)

    # sample LR image
    rnd = np.random.randint(0, ds.n_images)
    sample_x_lr = np.reshape(lr[rnd], (1,) + lr_shape)

    util.img_save(img=np.reshape(sample_x_lr, lr_shape), path=config.output_dir + "/sample_lr.png",
                  use_inverse=True)

    # gpu config
    gpu_config = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_config)

    with tf.Session(config=tf_config) as sess:
        rcan_model = model.RCAN(sess=sess,
                                lr_img_size=lr_shape[:-1],
                                hr_img_size=hr_shape[:-1],
                                batch_size=config.batch_size,
                                img_scaling_factor=config.image_scaling_factor,
                                n_res_blocks=config.n_res_blocks,
                                n_res_groups=config.n_res_groups,
                                res_scale=config.res_scale,
                                n_filters=config.filter_size,
                                kernel_size=config.kernel_size,
                                activation=config.activation,
                                use_bn=config.use_bn,
                                reduction=config.reduction,
                                optimizer=config.optimizer,
                                lr=config.lr,
                                lr_decay=config.lr_decay,
                                lr_decay_step=config.lr_decay_step,
                                momentum=config.momentum,
                                beta1=config.beta1,
                                beta2=config.beta2,
                                opt_eps=config.opt_epsilon,
                                tf_log=config.summary,
                                n_gpu=config.n_gpu,
                                )

        # Initializing
        sess.run(tf.global_variables_initializer())

        # Load model & Graph & Weights
        global_step = 0

        ckpt = tf.train.get_checkpoint_state(config.summary)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            rcan_model.saver.restore(sess, ckpt.model_checkpoint_path)

            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print("[+] global step : %d" % global_step, " successfully loaded")
        else:
            print('[-] No checkpoint file found')

        # config params
        lr = config.lr if global_step < config.lr_decay_step \
            else config.lr * (config.lr_decay * (global_step // config.lr_decay_step))

        rcan_model.global_step.assign(tf.constant(global_step))
        start_epoch = global_step // (ds.n_images // config.batch_size)

        best_loss = 2e2
        for epoch in range(start_epoch, config.epochs):
            for x_lr, x_hr in di.iterate():
                # training
                _, loss = sess.run([rcan_model.opt, rcan_model.loss],
                                   feed_dict={
                                       rcan_model.x_lr: x_lr,
                                       rcan_model.x_hr: x_hr,
                                       rcan_model.lr: lr,
                                       rcan_model.is_train: True,
                                   })

                if global_step % config.logging_step == 0:
                    print("[+] %d epochs %d steps" % (epoch, global_step), "loss : {:.8f}".format(loss))

                    # summary
                    summary = sess.run(rcan_model.merged,
                                       feed_dict={
                                           rcan_model.x_lr: x_lr,
                                           rcan_model.x_hr: x_hr,
                                           rcan_model.lr: lr,
                                           rcan_model.is_train: False,
                                       })
                    rcan_model.writer.add_summary(summary, global_step)

                    # output
                    output = sess.run(rcan_model.output,
                                      feed_dict={
                                          rcan_model.x_lr: sample_x_lr,
                                          rcan_model.lr: lr,
                                          rcan_model.is_train: False,
                                      })
                    output = np.reshape(output, rcan_model.hr_img_size)
                    util.img_save(img=output, path=config.output_dir + "/%d.png" % global_step,
                                  use_inverse=True)

                    # model save
                    rcan_model.saver.save(sess, config.summary, global_step)

                    if loss < best_loss:
                        print("[*] improved {:.8f} to {:.8f}".format(best_loss, loss))
                        rcan_model.best_saver.save(sess, config.summary, global_step)
                        best_loss = loss

                # lr schedule
                if global_step and global_step % config.lr_decay_step == 0:
                    lr *= config.lr_decay

                # increase global step
                rcan_model.global_step.assign_add(tf.constant(1))
                global_step += 1

    end_time = time.time() - start_time  # Clocking end

    # Elapsed time
    print("[+] Elapsed time {:.8f}s".format(end_time))


if __name__ == "__main__":
    main()
