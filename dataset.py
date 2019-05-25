from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import h5py
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from util import split
from config import get_config
from multiprocessing import Pool


# Configuration
config, _ = get_config()

seed = config.seed


class DataSetLoader:

    @staticmethod
    def get_extension(ext):
        if ext in ['jpg', 'png']:
            return 'img'
        elif ext == 'tfr':
            return 'tfr'
        elif ext == 'h5':
            return 'h5'
        elif ext == 'npy':
            return 'npy'
        else:
            raise ValueError("[-] There's no supporting file... [%s] :(" % ext)

    @staticmethod
    def get_img(path, size=(64, 64), interp=cv2.INTER_CUBIC):
        img = cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1]  # BGR to RGB
        if img.shape[:1] == size:
            return img
        else:
            return cv2.resize(img, size, interp)

    @staticmethod
    def parse_tfr_tf(record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenFeature([], tf.string)})
        data = tf.decode_raw(features['data'], tf.uint8)
        return tf.reshape(data, features['shape'])

    @staticmethod
    def parse_tfr_np(record):
        ex = tf.train.Example()
        ex.ParseFromString(record)
        shape = ex.features.feature['shape'].int64_list.value
        data = ex.features.feature['data'].bytes_list.value[0]
        return np.fromstring(data, np.uint8).reshape(shape)

    @staticmethod
    def img_scaling(img, scale='0,1'):
        if scale == '0,1':
            try:
                img /= 255.
            except TypeError:  # ufunc 'true divide' output ~
                img = np.true_divide(img, 255.0, casting='unsafe')
        elif scale == '-1,1':
            try:
                img = (img / 127.5) - 1.
            except TypeError:
                img = np.true_divide(img, 127.5, casting='unsafe') - 1.
        else:
            raise ValueError("[-] Only '0,1' or '-1,1' please - (%s)" % scale)

        return img

    def __init__(self, path, size=None, name='to_tfr', use_save=False, save_file_name='',
                 buffer_size=4096, n_threads=8,
                 use_image_scaling=False, image_scale='0,1', img_save_method=cv2.INTER_LINEAR, debug=True):

        self.op = name.split('_')
        self.debug = debug

        try:
            assert len(self.op) == 2
        except AssertionError:
            raise AssertionError("[-] Invalid Target Types :(")

        self.size = size

        try:
            assert self.size
        except AssertionError:
            raise AssertionError("[-] Invalid Target Sizes :(")

        # To-DO
        # Supporting 4D Image
        self.height = size[0]
        self.width = size[1]
        self.channel = size[2]

        self.path = path

        try:
            assert os.path.exists(self.path)
        except AssertionError:
            raise AssertionError("[-] Path(%s) does not exist :(" % self.path)

        self.buffer_size = buffer_size
        self.n_threads = n_threads

        if os.path.isfile(self.path):
            self.file_list = [self.path]
            self.file_ext = self.path.split('.')[-1]
            self.file_names = [self.path]
        else:
            self.file_list = sorted(os.listdir(self.path))
            self.file_ext = self.file_list[0].split('.')[-1]
            self.file_names = glob(self.path + '/*')
        self.raw_data = np.ndarray([], dtype=np.uint8)  # (N, H * W * C)

        if self.debug:
            print("[*] Detected Path            is [%s]" % self.path)
            print("[*] Detected File Extension  is [%s]" % self.file_ext)
            print("[*] Detected First File Name is [%s] (%d File(s))" % (self.file_names[0], len(self.file_names)))

        self.types = ('img', 'tfr', 'h5', 'npy')  # Supporting Data Types
        self.op_src = self.get_extension(self.file_ext)
        self.op_dst = self.op[1]

        try:
            chk_src, chk_dst = False, False
            for t in self.types:
                if self.op_src == t:
                    chk_src = True
                if self.op_dst == t:
                    chk_dst = True
            assert chk_src and chk_dst
        except AssertionError:
            raise AssertionError("[-] Invalid Operation Types (%s, %s) :(" % (self.op_src, self.op_dst))

        self.img_save_method = img_save_method

        if self.op_src == self.types[0]:
            self.load_img()
        elif self.op_src == self.types[1]:
            self.load_tfr()
        elif self.op_src == self.types[2]:
            self.load_h5()
        elif self.op_src == self.types[3]:
            self.load_npy()
        else:
            raise NotImplementedError("[-] Not Supported Type :(")

        # Random Shuffle
        order = np.arange(self.raw_data.shape[0])
        np.random.RandomState(seed).shuffle(order)
        self.raw_data = self.raw_data[order]

        # Clip [0, 255]
        self.raw_data = np.rint(self.raw_data).clip(0, 255).astype(np.uint8)

        self.use_save = use_save
        self.save_file_name = save_file_name

        if self.use_save:
            try:
                assert self.save_file_name
            except AssertionError:
                raise AssertionError("[-] Empty save-file name :(")

            if self.op_dst == self.types[0]:
                self.convert_to_img()
            elif self.op_dst == self.types[1]:
                self.tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
                self.tfr_writer = tf.python_io.TFRecordWriter(self.save_file_name + ".tfrecords", self.tfr_opt)
                self.convert_to_tfr()
            elif self.op_dst == self.types[2]:
                self.convert_to_h5()
            elif self.op_dst == self.types[3]:
                self.convert_to_npy()
            else:
                raise NotImplementedError("[-] Not Supported Type :(")

        self.use_image_scaling = use_image_scaling
        self.img_scale = image_scale

        if self.use_image_scaling:
            self.raw_data = self.img_scaling(self.raw_data, self.img_scale)

    def load_img(self):
        self.raw_data = np.zeros((len(self.file_list), self.height * self.width * self.channel),
                                 dtype=np.uint8)

        for i, fn in tqdm(enumerate(self.file_names)):
            self.raw_data[i] = self.get_img(fn, (self.height, self.width), self.img_save_method).flatten()
            if self.debug:  # just once
                print("[*] Image Shape   : ", self.raw_data[i].shape)
                print("[*] Image Size    : ", self.raw_data[i].size)
                print("[*] Image MIN/MAX :  (%d, %d)" % (np.min(self.raw_data[i]), np.max(self.raw_data[i])))
                self.debug = False

    def load_tfr(self):
        self.raw_data = tf.data.TFRecordDataset(self.file_names, compression_type='', buffer_size=self.buffer_size)
        self.raw_data = self.raw_data.map(self.parse_tfr_tf, num_parallel_calls=self.n_threads)

    def load_h5(self, size=0, offset=0):
        init = True

        for fl in self.file_list:  # For multiple .h5 files
            with h5py.File(fl, 'r') as hf:
                data = hf['images']
                full_size = len(data)

                if size == 0:
                    size = full_size

                n_chunks = int(np.ceil(full_size / size))
                if offset >= n_chunks:
                    print("[*] Looping from back to start.")
                    offset %= n_chunks
                if offset == n_chunks - 1:
                    print("[-] Not enough data available, clipping to end.")
                    data = data[offset * size:]
                else:
                    data = data[offset * size:(offset + 1) * size]

                data = np.array(data, dtype=np.uint8)
                print("[+] ", fl, " => Image size : ", data.shape)

                if init:
                    self.raw_data = data
                    init = False

                    if self.debug:  # just once
                        print("[*] Image Shape   : ", self.raw_data[0].shape)
                        print("[*] Image Size    : ", self.raw_data[0].size)
                        print("[*] Image MIN/MAX :  (%d, %d)" % (np.min(self.raw_data[0]), np.max(self.raw_data[0])))
                        self.debug = False

                    continue
                else:
                    self.raw_data = np.concatenate((self.raw_data, data))

    def load_npy(self):
        self.raw_data = np.rollaxis(np.squeeze(np.load(self.file_names), axis=0), 0, 3)

        if self.debug:  # just once
            print("[*] Image Shape   : ", self.raw_data[0].shape)
            print("[*] Image Size    : ", self.raw_data[0].size)
            print("[*] Image MIN/MAX :  (%d, %d)" % (np.min(self.raw_data[0]), np.max(self.raw_data[0])))
            self.debug = False

    def convert_to_img(self):
        def to_img(i):
            cv2.imwrite('imgHQ%05d.png' % i, cv2.COLOR_BGR2RGB)
            return True

        raw_data_shape = self.raw_data.shape  # (N, H * W * C)

        try:
            assert os.path.exists(self.save_file_name)
        except AssertionError:
            print("[-] There's no %s :(" % self.save_file_name)
            print("[*] Make directory at %s... " % self.save_file_name)
            os.mkdir(self.save_file_name)

        ii = [i for i in range(raw_data_shape[0])]

        pool = Pool(self.n_threads)
        print(pool.map(to_img, ii))

    def convert_to_tfr(self):
        for data in self.raw_data:
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=data.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tostring()]))
            }))
            self.tfr_writer.write(ex.SerializeToString())

    def convert_to_h5(self):
        with h5py.File(self.save_file_name, 'w') as f:
            f.create_dataset("images", data=self.raw_data)

    def convert_to_npy(self):
        np.save(self.save_file_name, self.raw_data)


class Div2KDataSet:

    def __init__(self, hr_height=768, hr_width=768, lr_height=192, lr_width=192, channel=3,
                 n_patch=16, use_split=False, split_rate=0.1, random_state=42, n_threads=8,
                 ds_path=None, ds_name=None, use_img_scale=True,
                 ds_hr_path=None, ds_lr_path=None,
                 use_save=False, save_type='to_h5', save_file_name=None, debug=False):

        """
        # General Settings
        :param hr_height: input HR image height, default 768
        :param hr_width: input HR image width, default 768
        :param lr_height: input LR image height, default 192
        :param lr_width: input LR image width, default 192
        :param channel: input image channel, default 3 (RGB)
        - in case of Div2K - ds x4, image size is 768 x 768 x 3 (HWC).

        # Pre-Processing Option
        :param n_patch: patch size to crop, default 16
        :param split_rate: image split rate (into train & test), default 0.1
        :param random_state: random seed for shuffling, default 42
        :param n_threads: the number of threads for multi-threading, default 8

        # DataSet Option
        :param ds_path: DataSet's Path, default None
        :param ds_name: DataSet's Name, default None
        :param use_img_scale: using img scaling?, default False
        :param ds_hr_path: DataSet High Resolution path
        :param ds_lr_path: DataSet Low Resolution path
        :param use_save: saving into another file format
        :param save_type: file format to save
        :param save_file_name: file name to save
        :param debug: debugging messages, default False
        """

        self.hr_height = hr_height
        self.hr_width = hr_width
        self.lr_height = lr_height
        self.lr_width = lr_width
        self.channel = channel
        self.hr_shape = (self.hr_height, self.hr_width, self.channel)
        self.lr_shape = (self.lr_height, self.lr_width, self.channel)

        self.n_patch = n_patch
        self.use_split = use_split
        self.split_rate = split_rate
        self.random_state = random_state
        self.num_threads = n_threads  # change this value to the fitted value for ur system

        """
        Expected ds_path : div2k/...
        Expected ds_name : X4
        """
        self.ds_path = ds_path
        self.ds_name = ds_name
        self.ds_hr_path = ds_hr_path
        self.ds_lr_path = ds_lr_path

        try:
            assert self.ds_path
        except AssertionError:
            try:
                assert self.ds_hr_path and self.ds_lr_path
            except AssertionError:
                raise AssertionError("[-] DataSet's path is required!")

        self.use_save = use_save
        self.save_type = save_type
        self.save_file_name = save_file_name
        self.debug = debug

        try:
            if self.use_save:
                assert self.save_file_name
            else:
                self.save_file_name = ""
        except AssertionError:
            raise AssertionError("[-] save-file/folder-name is required!")

        self.n_images = 800
        self.n_images_val = 100

        self.use_img_scaling = use_img_scale

        if self.ds_path:  # like .h5 or .tfr # will be in the same folder
            self.ds_hr_path = self.ds_path + "/DIV2K_train_HR/"
            self.ds_lr_path = self.ds_hr_path

        self.hr_images = DataSetLoader(path=self.ds_hr_path,
                                       size=self.hr_shape,
                                       use_save=self.use_save,
                                       name=self.save_type,
                                       save_file_name=self.save_file_name + "-hr.h5",
                                       use_image_scaling=self.use_img_scaling,
                                       image_scale='0,1',
                                       img_save_method=cv2.INTER_LINEAR).raw_data  # numpy arrays
        self.patch_hr_images = None

        self.lr_images = DataSetLoader(path=self.ds_lr_path,
                                       size=self.lr_shape,
                                       use_save=self.use_save,
                                       name=self.save_type,
                                       save_file_name=self.save_file_name + "-lr.h5",
                                       use_image_scaling=self.use_img_scaling,
                                       image_scale='0,1',
                                       img_save_method=cv2.INTER_CUBIC).raw_data  # numpy arrays
        self.patch_lr_images = None

        if self.n_patch > 0:
            patch_size = int(np.sqrt(self.n_patch))

            self.patch_hr_images = np.zeros((self.n_images * self.n_patch,
                                             self.hr_height // patch_size, self.hr_width // patch_size, self.channel),
                                            dtype=np.uint8)

            self.patch_lr_images = np.zeros((self.n_images * self.n_patch,
                                             self.lr_height // patch_size, self.lr_width // patch_size, self.channel),
                                            dtype=np.uint8)

            for i in tqdm(range(self.n_images)):
                hr_patches = split(np.reshape(self.hr_images[i, :], self.hr_shape), self.n_patch)
                lr_patches = split(np.reshape(self.lr_images[i, :], self.lr_shape), self.n_patch)

                for n_ps in range(self.n_patch):
                    self.patch_hr_images[i * self.n_patch + n_ps] = hr_patches[n_ps]
                    self.patch_lr_images[i * self.n_patch + n_ps] = lr_patches[n_ps]

                if self.debug:
                    import matplotlib.pyplot as plt

                    fig = plt.figure()
                    for j in range(self.n_patch):
                        fig.add_subplot(4, 4, j + 1)
                        plt.imshow(self.patch_hr_images[j, :, :, :])
                    plt.show()

                    fig = plt.figure()
                    for j in range(self.n_patch):
                        fig.add_subplot(4, 4, j + 1)
                        plt.imshow(self.patch_lr_images[j, :, :, :])
                    plt.show()

                    self.debug = False


class DataIterator:

    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_examples = num_examples = x.shape[0]
        self.num_batches = num_examples // batch_size
        self.pointer = 0

        assert (self.batch_size <= self.num_examples)

    def next_batch(self):
        start = self.pointer
        self.pointer += self.batch_size

        if self.pointer > self.num_examples:
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)

            self.x = self.x[perm, :, :, :]
            self.y = self.y[perm, :, :, :]

            start = 0
            self.pointer = self.batch_size

        end = self.pointer
        return self.x[start:end, :, :, :], self.y[start:end, :, :, :]

    def iterate(self):
        for step in range(self.num_batches):
            yield self.next_batch()
