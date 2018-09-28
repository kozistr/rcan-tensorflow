from dataset import Div2KDataSet as DataSet
from dataset import DataIterator
from config import get_config

import tensorflow as tf
import time

import model
import util

# Configuration
config, _ = get_config()


def main():
    start_time = time.time()  # Clocking start

    # Div2K - Track 1: Bicubic downscaling - x4 DataSet load
    ds = DataSet(ds_path=config.data_dir,
                 ds_name="X4",
                 use_save=True,
                 save_type="to_h5",
                 save_file_name=config.data_dir + "DIV2K",
                 use_img_scale=False)
    """
    ds = DataSet(ds_hr_path=config.data_dir + "DIV2K-hr.h5",
                 ds_lr_path=config.data_dir + "DIV2K-lr.h5",
                 use_img_scale=False)
    """

    hr, lr = ds.hr_images, ds.lr_images  # [0, 255] scaled images

    print("[+] Loaded HR image ", hr.shape)
    print("[+] Loaded LR image ", lr.shape)

    sample_x_lr = None

    # DataIterator
    di = DataIterator(lr, hr, config.batch_size)

    # gpu config
    gpu_config = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_config)

    with tf.Session(config=tf_config) as sess:
        rcan_model = model.RCAN(sess=sess,
                                batch_size=config.batch_size,
                                img_scaling_factor=config.image_scaling_factor,
                                n_res_blocks=config.n_res_blocks,
                                n_res_groups=config.n_res_groups,
                                res_scale=config.n_res_scale,
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
                # training
                _, loss = sess.run([rcan_model.opt, rcan_model.loss],
                                   feed_dict={
                                       rcan_model.x_lr: x_lr,
                                       rcan_model.x_hr: x_hr,
                                       rcan_model.lr: lr,
                                   })

                if global_step % config.logging_step == 0:
                    print("[+] %d epochs %d steps" % (epoch, global_step), "loss : {.8f}".format(loss))

                    # summary
                    summary = sess.run(rcan_model.merged,
                                       feed_dict={
                                           rcan_model.x_lr: x_lr,
                                           rcan_model.x_hr: x_hr,
                                           rcan_model.lr: lr,
                                       })
                    rcan_model.writer.add_summary(summary, global_step)

                    # output
                    output = sess.run(rcan_model.output,
                                      feed_dict={
                                          rcan_model.x_lr: sample_x_lr,
                                          rcan_model.lr: lr,
                                      })
                    util.img_save(img=output, path=config.output_dir + "/%d.png" % global_step)

                    # model save
                    rcan_model.saver.save(sess, config.summary, global_step)

                    if loss < best_loss:
                        print("[*] improved {.8f} to {.8f}".format(best_loss, loss))
                        rcan_model.best_saver.save(sess, config.summary, global_step)
                        best_loss = loss

                # increase global step
                rcan_model.global_step.assign_add(tf.constant(1))

                # lr schedule
                if global_step and global_step % config.lr_decay_step == 0:
                    lr *= config.lr_decay

    end_time = time.time() - start_time  # Clocking end

    # Elapsed time
    print("[+] Elapsed time {:.8f}s".format(end_time))


if __name__ == "__main__":
    main()
