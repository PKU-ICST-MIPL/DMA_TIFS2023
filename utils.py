import sys
import errno
import os.path as osp
from torch.backends import cudnn
import random
import torch
import torch.nn as nn
import numpy as np
import math
import transforms as T
from torch.nn import functional as F
import os
import re
import shutil
from tqdm import tqdm
from pynvml import *

def use_gpu(used_percentage=0.1):
    nvmlInit()
    gpu_num = nvmlDeviceGetCount()
    out = ""
    for i in range(gpu_num):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_percentage_real = info.used / info.total
        if used_percentage_real < used_percentage:
            out = str(i)
    nvmlShutdown()
    if gpu_num == "":
        print ("No suitable GPU for training.")
    return out

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')
    def __del__(self):
        self.close()
    def __enter__(self):
        pass
    def __exit__(self, *args):
        self.close()
    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
    def close(self):
        if self.file is not None:
            self.file.close()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if epoch <= args.warmup_epoch:
        lr = lr * epoch  / args.warmup_epoch
    for scheduler_epoch in args.scheduler_epoch:
        if epoch > scheduler_epoch:
            lr = lr * 0.1
    optimizer.param_groups[0]['lr'] = lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = args.dlr * lr
    return lr

def get_transform(args):
    transform_train = T.Compose([
        T.Resize((args.img_h, args.img_w)),
        T.Pad(10), T.RandomCrop((args.img_h, args.img_w)),
        T.RandomHorizontalFlip(), T.ToTensor(), T.RandomErasing(p=0.5), T.RandomColoring(p=0.5), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((args.img_h, args.img_w)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_train, transform_test

def get_transform_naive(args):
    transform_train = T.Compose([
        T.Resize((args.img_h, args.img_w)),
        T.Pad(10), T.RandomCrop((args.img_h, args.img_w)),
        T.RandomHorizontalFlip(), T.ToTensor(), T.RandomErasing(p=0.5),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((args.img_h, args.img_w)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_train, transform_test

# def get_grad_norm(parameters, norm_type=2):
#     if isinstance(parameters, torch.Tensor):
#         parameters = [parameters]
#     parameters = list(filter(lambda p: p.grad is not None, parameters))
#     norm_type = float(norm_type)
#     total_norm = 0
#     for p in parameters:
#         param_norm = p.grad.data.norm(norm_type)
#         total_norm += param_norm.item() ** norm_type
#     total_norm = total_norm ** (1. / norm_type)
#     return total_norm

def pre_process_sysu(dir="/home/cuizhenyu/Dataset_VIReID/SYSU-MM01/"):
    if os.path.isdir(dir + 'train_rgb/') and os.path.isdir(dir + 'gallery_rgb/') \
            and os.path.isdir(dir + 'train_ir/') and os.path.isdir(dir + 'query_ir/'):
        return
    if not os.path.isdir(dir + 'train_rgb/'):
        os.makedirs(dir + 'train_rgb/')
    if not os.path.isdir(dir + 'gallery_rgb/'):
        os.makedirs(dir + 'gallery_rgb/')
    if not os.path.isdir(dir + 'train_ir/'):
        os.makedirs(dir + 'train_ir/')
    if not os.path.isdir(dir + 'query_ir/'):
        os.makedirs(dir + 'query_ir/')

    rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    ir_cameras = ['cam3', 'cam6']

    file_path_train = os.path.join(dir, 'exp/train_id.txt')
    file_path_val = os.path.join(dir, 'exp/val_id.txt')
    file_path_test = os.path.join(dir, 'exp/test_id.txt')

    with open(file_path_train, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_train = ["%04d" % x for x in ids]

    with open(file_path_val, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_val = ["%04d" % x for x in ids]

    with open(file_path_test, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_test = ["%04d" % x for x in ids]
    # combine train and val split
    id_train.extend(id_val)

    train_rgb = []
    train_ir = []
    gallery_rgb = []
    query_ir = []

    for id in sorted(id_train):
        for cam in rgb_cameras:
            img_dir = os.path.join(dir, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                train_rgb.extend(new_files)
        for cam in ir_cameras:
            img_dir = os.path.join(dir, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                train_ir.extend(new_files)
    for id in sorted(id_test):
        for cam in rgb_cameras:
            img_dir = os.path.join(dir, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                gallery_rgb.extend(new_files)
        for cam in ir_cameras:
            img_dir = os.path.join(dir, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                query_ir.extend(new_files)

    for img_path in tqdm(train_rgb, desc='train_rgb'):
        pattern = re.compile(r'cam(\d+)/(\d+)/(\d+)')
        camid, pid, imgid = pattern.search(img_path).groups()
        path = dir + 'train_rgb/' + pid + '_c' + camid + '_' + imgid + '.jpg'
        shutil.copyfile(img_path, path)
    for img_path in tqdm(train_ir, desc='train_ir'):
        pattern = re.compile(r'cam(\d+)/(\d+)/(\d+)')
        camid, pid, imgid = pattern.search(img_path).groups()
        path = dir + 'train_ir/' + pid + '_c' + camid + '_' + imgid + '.jpg'
        shutil.copyfile(img_path, path)
    for img_path in tqdm(gallery_rgb, desc='gallery_rgb'):
        pattern = re.compile(r'cam(\d+)/(\d+)/(\d+)')
        camid, pid, imgid = pattern.search(img_path).groups()
        path = dir + 'gallery_rgb/' + pid + '_c' + camid + '_' + imgid + '.jpg'
        shutil.copyfile(img_path, path)
    for img_path in tqdm(query_ir, desc='query_ir'):
        pattern = re.compile(r'cam(\d+)/(\d+)/(\d+)')
        camid, pid, imgid = pattern.search(img_path).groups()
        path = dir + 'query_ir/' + pid + '_c' + camid + '_' + imgid + '.jpg'
        shutil.copyfile(img_path, path)


