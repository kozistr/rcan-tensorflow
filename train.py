from dataset import Div2KDataSet as DataSet
from dataset import DataIterator
from config import get_config

import tensorflow as tf
import numpy as np
import argparse
import time
import os

import model
import util


# Argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_from', type=str, default='img', choices=['img', 'h5'])
args = parser.parse_args()

data_from = args.data_from

# Configuration
config, _ = get_config()

np.random.seed(config.seed)
tf.set_random_seed(config.seed)


def main():
    start_time = time.time()  # Clocking start

    # Div2K - Track 1: Bicubic downscaling - x4 DataSet load
    if data_from == 'img':
        ds = DataSet(ds_path=config.data_dir,
                     ds_name="X4",
                     use_save=True,
                     save_type="to_h5",
                     save_file_name=config.data_dir + "DIV2K",
                     use_img_scale=False,
                     n_patch=config.patch_size)
    else:  # .h5 files
        ds = DataSet(ds_hr_path=config.data_dir + "DIV2K-hr.h5",
                     ds_lr_path=config.data_dir + "DIV2K-lr.h5",
                     use_img_scale=False,
                     n_patch=config.patch_size)

    if config.patch_size > 0:
        hr, lr = ds.patch_hr_images, ds.patch_lr_images  # [0, 1] scaled images
    else:
        hr, lr = ds.hr_images, ds.lr_images

    lr_shape = lr.shape[1:]
    hr_shape = hr.shape[1:]

    print("[+] Loaded LR patch image ", lr.shape)
    print("[+] Loaded HR patch image ", hr.shape)

    # setup directory
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)

    # sample LR image
    if config.patch_size > 0:
        patch = int(np.sqrt(config.patch_size))

        rnd = np.random.randint(0, ds.n_images)

        sample_lr = lr[config.patch_size * rnd:config.patch_size * (rnd + 1), :, :, :]
        sample_lr = np.reshape(sample_lr, (config.patch_size,) + lr_shape)  # (16,) + lr_shape

        sample_hr = hr[config.patch_size * rnd:config.patch_size * (rnd + 1), :, :, :]
        sample_hr = np.reshape(sample_hr, (config.patch_size,) + hr_shape)  # (16,) + hr_shape

        util.img_save(img=util.merge(sample_lr, (patch, patch)),
                      path=config.output_dir + "/sample_lr.png",
                      use_inverse=False,
                      )
        util.img_save(img=util.merge(sample_hr, (patch, patch)),
                      path=config.output_dir + "/sample_hr.png",
                      use_inverse=False,
                      )
    else:
        rnd = np.random.randint(0, ds.n_images)

        sample_lr = lr[rnd]
        sample_lr = np.reshape(sample_lr, lr_shape)  # lr_shape

        sample_hr = hr[rnd]
        sample_hr = np.reshape(sample_hr, hr_shape)  # hr_shape

        util.img_save(img=sample_lr,
                      path=config.output_dir + "/sample_lr.png",
                      use_inverse=False,
                      )
        util.img_save(img=sample_hr,
                      path=config.output_dir + "/sample_hr.png",
                      use_inverse=False,
                      )

        # scaling into lr [0, 1]
        sample_lr /= 255.

    # DataIterator
    di = DataIterator(lr, hr, config.batch_size)

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

        best_loss = 1e8
        for epoch in range(start_epoch, config.epochs):
            for x_lr, x_hr in di.iterate():
                # scaling into lr [0, 1] # hr [0, 255]
                x_lr = np.true_divide(x_lr, 255., casting='unsafe')

                # training
                _, loss, psnr, ssim = sess.run([rcan_model.train_op, rcan_model.loss, rcan_model.psnr, rcan_model.ssim],
                                               feed_dict={
                                                   rcan_model.x_lr: x_lr,
                                                   rcan_model.x_hr: x_hr,
                                                   rcan_model.lr: lr,
                                               })

                if global_step % config.logging_step == 0:
                    print("[+] %d epochs %d steps" % (epoch, global_step),
                          "loss : {:.8f} PSNR : {:.4f} SSIM : {:.4f}".format(loss, psnr, ssim))

                    # summary & output
                    summary, output = sess.run([rcan_model.merged, rcan_model.output],
                                               feed_dict={
                                                   rcan_model.x_lr: sample_lr,
                                                   rcan_model.x_hr: sample_hr,
                                                   rcan_model.lr: lr,
                                               })
                    rcan_model.writer.add_summary(summary, global_step)

                    util.img_save(img=util.merge(output, (patch, patch)),
                                  path=config.output_dir + "/%d.png" % global_step,
                                  use_inverse=False)

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
