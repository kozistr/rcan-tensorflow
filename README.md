# rcan-tensorflow
Image Super-Resolution Using Very Deep Residual Channel Attention Networks Implementation in Tensorflow

[ECCV 2018 paper](https://arxiv.org/pdf/1807.02758.pdf)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/kozistr/rcan-tensorflow.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kozistr/rcan-tensorflow/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/kozistr/rcan-tensorflow.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kozistr/rcan-tensorflow/context:python)

OOM on my machine :(... I can't test my code, but maybe code runs fine.

## Introduction
This repo contains my implementation of RCAN (Residual Channel Attention Networks).

## Dependencies
* Python
* Tensorflow 1.x
* tqdm
* h5py

## DataSet
* DIV2K DataSet

## Usage
### training
    $ python3 train.py --checkpoint [pretrained model]
### testing
    $ python3 test.py

## Results


## Author
HyeongChan Kim / [@kozistr](http://kozistr.tech)
