# Introduction

This is the source code of our TIFS 2024 paper "DMA: Dual Modality-Aware Alignment for Visible-Infrared Person Re-Identification". Please cite the following paper if you use our code.

Zhenyu Cui, Jiahuan Zhou, and Yuxin Peng, "DMA: Dual Modality-Aware Alignment for Visible-Infrared Person Re-Identification", IEEE Transactions on Information Forensics and Security (TIFS), 2024.


# Dependencies

- Python 3.7

- cudatoolkit 11.3

- cudnn 8.4

- PyTorch 1.9.1


# Data Preparation

- Download the [SYSU-MM01](https://github.com/wuancong/SYSU-MM01) dataset and the [RegDB](http://dm.dongguk.edu/link.html) dataset, and place them to `/home/cuizhenyu/Dataset_VIReID/` folders.


# Usage

- Start training by executing the following commands.

1. For SYSU-MM01 dataset:

    Train: `python train.py --dataset sysu -sche 80 140 -v 1 -maxe 200`

    Test: `python test.py --dataset sysu --model_path ./save_model/sysu_v1/model_best.t`

2. For RegDB dataset: `python train.py --dataset regdb -sche 80 140 -v 1 -maxe 200`

For any questions, feel free to contact us (cuizhenyu@stu.pku.edu.cn).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl/home/) for more information about our papers, source codes, and datasets.