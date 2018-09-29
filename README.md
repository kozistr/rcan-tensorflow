# rcan-tensorflow
Image Super-Resolution Using Very Deep Residual Channel Attention Networks Implementation in Tensorflow

[ECCV 2018 paper](https://arxiv.org/pdf/1807.02758.pdf)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/kozistr/rcan-tensorflow.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kozistr/rcan-tensorflow/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/kozistr/rcan-tensorflow.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kozistr/rcan-tensorflow/context:python)

## Introduction
This repo contains my implementation of RCAN (Residual Channel Attention Networks).

## Dependencies
* Python
* Tensorflow 1.x
* tqdm
* h5py
* scipy
* cv2

## DataSet
* DIV2K DataSet (X4 bicubic)

## Usage
### training
    # hyper-paramters in config.py, you can edit them!
    $ python3 train.py
### testing
    $ python3 test.py --src_image ./sample.png --dst_image sample-upscaled.png

## Results
* OOM on my machine :(... I can't test my code, but maybe code runs fine.

## Author
HyeongChan Kim / [@kozistr](http://kozistr.tech)
