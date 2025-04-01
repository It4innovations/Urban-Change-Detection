import numpy as np
from torchvision import utils
import torchvision.transforms.functional as TF
import os
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.data import Subset
import torch.nn.functional as F
from torch.nn import MaxPool1d,AvgPool1d
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import init

import math
from typing import Iterable, Set, Tuple
import mlflow
from prettytable import PrettyTable
import random
import glob
import sys
import time
import random
#import cv2
from PIL import Image
from PIL import ImageFilter
import PIL
import tifffile
import functools
from einops import rearrange
import re
import shutil

from mlflow.models.signature import infer_signature


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

### utils.py
def get_loader(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='CDDataset'):
    dataConfig = DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split,
                             img_size=img_size, is_train=is_train,
                             label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)

    return dataloader


def get_loaders(args):

    data_name = args.data_name
    # dataConfig = data_config.DataConfig().get_data_config(data_name)
    dataConfig = DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'CDDataset':
        training_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=args.img_size,is_train=True,
                                 label_transform=label_transform)
        val_set = CDDataset(root_dir=root_dir, split=split_val,
                                 img_size=args.img_size,is_train=False,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    return dataloaders

def get_loader_predict(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='CDDataset'):
    dataConfig = DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    if dataset == 'CDDataset':
        data_set = CDDataset_predict(root_dir=root_dir, split=split, img_size=img_size, is_train=is_train, label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)
    
    return dataloader

def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

### data_config.py

class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = './datasets/levir_cd'
        elif data_name == 'test':
            self.root_dir = './datasets/test'
        elif data_name == 'WHU':
            self.root_dir = './datasets/whu_cd'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self

### torchutils.py

def visualize_imgs(*imgs):

    import matplotlib.pyplot as plt
    nums = len(imgs)
    if nums > 1:
        fig, axs = plt.subplots(1, nums)
        for i, image in enumerate(imgs):
            axs[i].imshow(image, cmap='jet')
    elif nums == 1:
        fig, ax = plt.subplots(1, nums)
        for i, image in enumerate(imgs):
            ax.imshow(image, cmap='jet')
        plt.show()
    plt.show()

def minmax(tensor):
    assert tensor.ndim >= 2
    shape = tensor.shape
    tensor = tensor.view([*shape[:-2], shape[-1]*shape[-2]])
    min_, _ = tensor.min(-1, keepdim=True)
    max_, _ = tensor.max(-1, keepdim=True)
    return min_, max_

def norm_tensor(tensor,min_=None,max_=None, mode='minmax'):

    assert tensor.ndim >= 2
    shape = tensor.shape
    tensor = tensor.view([*shape[:-2], shape[-1]*shape[-2]])
    if mode == 'minmax':
        if min_ is None:
            min_, _ = tensor.min(-1, keepdim=True)
        if max_ is None:
            max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    elif mode == 'thres':
        N = tensor.shape[-1]
        thres_a = 0.001
        top_k = round(thres_a*N)
        max_ = tensor.topk(top_k, dim=-1, largest=True)[0][..., -1]
        max_ = max_.unsqueeze(-1)
        min_ = tensor.topk(top_k, dim=-1, largest=False)[0][..., -1]
        min_ = min_.unsqueeze(-1)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)

    elif mode == 'std':
        mean, std = torch.std_mean(tensor, [-1], keepdim=True)
        tensor = (tensor - mean)/std
        min_, _ = tensor.min(-1, keepdim=True)
        max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    elif mode == 'exp':
        tai = 1
        tensor = torch.nn.functional.softmax(tensor/tai, dim=-1, )
        min_, _ = tensor.min(-1, keepdim=True)
        max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    else:
        raise NotImplementedError
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.view(shape)


def visulize_features(features, normalize=False):

    from torchvision.utils import make_grid
    assert features.ndim == 4
    b,c,h,w = features.shape
    features = features.view((b*c, 1, h, w))
    if normalize:
        features = norm_tensor(features)
    grid = make_grid(features)
    visualize_tensors(grid)

def visualize_tensors(*tensors):

    import matplotlib.pyplot as plt
    # from misc.torchutils import tensor2np
    images = []
    for tensor in tensors:
        assert tensor.ndim == 3 or tensor.ndim==2
        if tensor.ndim ==3:
            assert tensor.shape[0] == 1 or tensor.shape[0] == 3
        images.append(tensor2np(tensor))
    nums = len(images)
    if nums>1:
        fig, axs = plt.subplots(1, nums)
        for i, image in enumerate(images):
            axs[i].imshow(image, cmap='jet')
        plt.show()
    elif nums == 1:
        fig, ax = plt.subplots(1, nums)
        for i, image in enumerate(images):
            ax.imshow(image, cmap='jet')
        plt.show()


def np_to_tensor(image):

    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[2]==3:
                image = np.transpose(image,[2,0,1])
        elif image.ndim == 2:
            image = np.newaxis(image, 0)
        image = torch.from_numpy(image)
        return image.unsqueeze(0)


def seed_torch(seed=2019):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res

class ChannelMaxPool(MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        pooled =  F.max_pool1d(input, self.kernel_size, self.stride,
                        self.padding, self.dilation, self.ceil_mode,
                        self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,c,w,h)

class ChannelAvePool(AvgPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        pooled = F.avg_pool1d(input, self.kernel_size, self.stride,
                        self.padding)
        _, _, c = pooled.size()
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,c,w,h)

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):

    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

def balanced_cross_entropy(input, target, weight=None,ignore_index=255):

    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    pos = (target==1).float()
    neg = (target==0).float()
    pos_num = torch.sum(pos) + 0.0000001
    neg_num = torch.sum(neg) + 0.0000001
    target_pos = target.float()
    target_pos[target_pos!=1] = ignore_index  # ÄºÅ¼ËÃ§â€¢Ä„Ã¤Â¸Å¤Ã¤Â¸ÅŸÄ‡Â­ÅÄ‡Â Â·Ä‡Å›Â¬Ã§Å¡â€žÄºÅšÅŸÄºÅºÅº
    target_neg = target.float()
    target_neg[target_neg!=0] = ignore_index  # ÄºÅ¼ËÃ§â€¢Ä„Ã¤Â¸Å¤Ã¤Â¸ÅŸÄÂ´ÅºÄ‡Â Â·Ä‡Å›Â¬Ã§Å¡â€žÄºÅšÅŸÄºÅºÅº


    loss_pos = cross_entropy(input, target_pos,weight=weight,reduction='sum',ignore_index=ignore_index)
    loss_neg = cross_entropy(input, target_neg,weight=weight,reduction='sum',ignore_index=ignore_index)
    loss = 0.5 * loss_pos / pos_num + 0.5 * loss_neg / neg_num
    return loss

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'poly':
        max_step = opt.niter+opt.niter_decay
        power = 0.9
        def lambda_rule(epoch):
            current_step = epoch + opt.epoch_count
            lr_l = (1.0 - current_step / (max_step+1)) ** float(power)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def mul_cls_acc(preds, targets, topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        bs, C = targets.shape
        _, pred = preds.topk(maxk, 1, True, True)
        pred += 1  
        correct = torch.zeros([bs, maxk]).long()  
        if preds.device != torch.device(type='cpu'):
            correct = correct.cuda()
        for i in range(C):
            label = i + 1
            target = targets[:, i] * label
            correct = correct + pred.eq(target.view(-1, 1).expand_as(pred)).long()
        n = (targets == 1).long().sum(1)  
        res = []
        for k in topk:
            acc_k = correct[:, :k].sum(1).float() / n.float()  
            acc_k = acc_k.sum()/bs
            res.append(acc_k)
    return res


def cls_accuracy(output, target, topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, init_step=0, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = init_step
        print(self.global_step)
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class PolyAdamOptimizer(torch.optim.Adam):
    def __init__(self, params, lr, betas, max_step, momentum=0.9):
        super().__init__(params, lr, betas)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)
        self.global_step += 1

class SGDROptimizer(torch.optim.SGD):

    def __init__(self, params, steps_per_epoch, lr=0, weight_decay=0, epoch_start=1, restart_mult=2):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.local_step = 0
        self.total_restart = 0

        self.max_step = steps_per_epoch * epoch_start
        self.restart_mult = restart_mult

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.local_step >= self.max_step:
            self.local_step = 0
            self.max_step *= self.restart_mult
            self.total_restart += 1

        lr_mult = (1 + math.cos(math.pi * self.local_step / self.max_step))/2 / (self.total_restart + 1)

        for i in range(len(self.param_groups)):
            self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.local_step += 1
        self.global_step += 1


def split_dataset(dataset, n_splits):

    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out


def decode_seg(label_mask, toTensor=False):

    if not isinstance(label_mask, np.ndarray):
        if isinstance(label_mask, torch.Tensor):  # get the data from a variable
            image_tensor = label_mask.data
        else:
            return label_mask
        label_mask = image_tensor[0][0].cpu().numpy()

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3),dtype=np.float)
    r = label_mask % 6
    g = (label_mask % 36) // 6
    b = label_mask // 36
    rgb[:, :, 0] = r / 6
    rgb[:, :, 1] = g / 6
    rgb[:, :, 2] = b / 6
    if toTensor:
        rgb = torch.from_numpy(rgb.transpose([2,0,1])).unsqueeze(0)

    return rgb


def tensor2im(input_image, imtype=np.uint8, normalize=True):
    
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        if image_numpy.shape[0] == 3:  # if RGB
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            if normalize:
                image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2np(input_image, if_normalize=True):
    
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array

    else:
        image_numpy = input_image
    if image_numpy.ndim == 2:
        return image_numpy
    elif image_numpy.ndim == 3:
        C, H, W = image_numpy.shape
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if C == 1:
            image_numpy = image_numpy[:, :, 0]
        if if_normalize and C == 3:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
            #  add to prevent extreme noises in visual images
            image_numpy[image_numpy<0]=0
            image_numpy[image_numpy>255]=255
            image_numpy = image_numpy.astype(np.uint8)
    return image_numpy


import ntpath
def save_visuals(visuals, img_dir, name, save_one=True, iter='0'):
 
    # save images to the disk
    for label, image in visuals.items():
        N = image.shape[0]
        if save_one:
            N = 1
        for j in range(N):
            name_ = ntpath.basename(name[j])
            name_ = name_.split(".")[0]
            image_numpy = tensor2np(image[j], if_normalize=True).astype(np.uint8)
            img_path = os.path.join(img_dir, iter+'_%s_%s.png' % (name_, label))
            save_image(image_numpy, img_path)

### pyutils.py

def seed_random(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def mkdir(path):
   
    if not os.path.exists(path):
        os.makedirs(path)


def get_paths(image_folder_path, suffix='*.png'):
    
    paths = sorted(glob.glob(os.path.join(image_folder_path, suffix)))
    return paths


def get_paths_from_list(image_folder_path, list):
    out = []
    for item in list:
        path = os.path.join(image_folder_path,item)
        out.append(path)
    return sorted(out)

### metric_tool.py

###################       metrics      ###################
class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict



def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']

### logger_tool.py

class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log_path = outfile
        now = time.strftime("%c")
        self.write('================ (%s) ================\n' % now)

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, mode='a') as f:
            f.write(message)

    def write_dict(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v)
        self.write(message)

    def write_dict_str(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %s ' % (k, v)
        self.write(message)

    def flush(self):
        self.terminal.flush()


class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)


    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))

    def str_estimated_remaining(self):
        return str(self.est_remaining/3600) + 'h'

    def estimated_remaining(self):
        return self.est_remaining/3600

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out

### imutils.py

def cv_rotate(image, angle, borderValue):
    
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    if isinstance(borderValue, int):
        values = (borderValue, borderValue, borderValue)
    else:
        values = borderValue
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=values)


def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))


def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def pil_rotate(img, degree, default_value):
    if isinstance(default_value, tuple):
        values = (default_value[0], default_value[1], default_value[2], 0)
    else:
        values = (default_value, default_value, default_value,0)
    img = Image.fromarray(img)
    if img.mode =='RGB':
        # set img padding == default_value
        img2 = img.convert('RGBA')
        rot = img2.rotate(degree, expand=1)
        fff = Image.new('RGBA', rot.size, values)  # Ã§ÂÂ°Äâ€°Ë›
        out = Image.composite(rot, fff, rot)
        img = out.convert(img.mode)

    else:
        # set label padding == default_value
        img2 = img.convert('RGBA')
        rot = img2.rotate(degree, expand=1)
        # a white image same size as rotated image
        fff = Image.new('RGBA', rot.size, values)
        # create a composite image using the alpha layer of rot as a mask
        out = Image.composite(rot, fff, rot)
        img = out.convert(img.mode)

    return np.asarray(img)


def random_resize_long_image_list(img_list, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = img_list[0].shape[:2]
    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w
    out = []
    for img in img_list:
        out.append(pil_rescale(img, scale, 3) )
    return out


def random_resize_long(img, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = img.shape[:2]

    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w

    return pil_rescale(img, scale, 3)


def random_scale_list(img_list, scale_range, order):
    
    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    if isinstance(img_list, tuple):
        assert img_list.__len__() == 2
        img1 = []
        img2 = []
        for img in img_list[0]:
            img1.append(pil_rescale(img, target_scale, order[0]))
        for img in img_list[1]:
            img2.append(pil_rescale(img, target_scale, order[1]))
        return (img1, img2)
    else:
        out = []
        for img in img_list:
            out.append(pil_rescale(img, target_scale, order))
        return out


def random_scale(img, scale_range, order):

    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    if isinstance(img, tuple):
        return (pil_rescale(img[0], target_scale, order[0]), pil_rescale(img[1], target_scale, order[1]))
    else:
        return pil_rescale(img, target_scale, order)


def random_rotate_list(img_list, max_degree, default_values):
    degree = random.random() * max_degree
    if isinstance(img_list, tuple):
        assert img_list.__len__() == 2
        img1 = []
        img2 = []
        for img in img_list[0]:
            assert isinstance(img, np.ndarray)
            img1.append((pil_rotate(img, degree, default_values[0])))
        for img in img_list[1]:
            img2.append((pil_rotate(img, degree, default_values[1])))
        return (img1, img2)
    else:
        out = []
        for img in img_list:
            out.append(pil_rotate(img, degree, default_values))
        return out


def random_rotate(img, max_degree, default_values):
    degree = random.random() * max_degree
    if isinstance(img, tuple):
        return (pil_rotate(img[0], degree, default_values[0]),
                pil_rotate(img[1], degree, default_values[1]))
    else:
        return pil_rotate(img, degree, default_values)


def random_lr_flip_list(img_list):

    if bool(random.getrandbits(1)):
        if isinstance(img_list, tuple):
            assert img_list.__len__()==2
            img1=list((np.fliplr(m) for m in img_list[0]))
            img2=list((np.fliplr(m) for m in img_list[1]))

            return (img1, img2)
        else:
            return list([np.fliplr(m) for m in img_list])
    else:
        return img_list


def random_lr_flip(img):

    if bool(random.getrandbits(1)):
        if isinstance(img, tuple):
            return tuple([np.fliplr(m) for m in img])
        else:
            return np.fliplr(img)
    else:
        return img


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def random_crop_list(images_list, cropsize, default_values):

    if isinstance(images_list, tuple):
        imgsize = images_list[0][0].shape[:2]
    elif isinstance(images_list, list):
        imgsize = images_list[0].shape[:2]
    else:
        raise RuntimeError('do not support the type of image_list')
    if isinstance(default_values, int): default_values = (default_values,)

    box = get_random_crop_box(imgsize, cropsize)
    if isinstance(images_list, tuple):
        assert images_list.__len__()==2
        img1 = []
        img2 = []
        for img in images_list[0]:
            f = default_values[0]
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            img1.append(cont)
        for img in images_list[1]:
            f = default_values[1]
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            img2.append(cont)
        return (img1, img2)
    else:
        out = []
        for img in images_list:
            f = default_values
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype) * f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            out.append(cont)
        return out


def random_crop(images, cropsize, default_values):

    if isinstance(images, np.ndarray): images = (images,)
    if isinstance(default_values, int): default_values = (default_values,)

    imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, default_values):

        if len(img.shape) == 3:
            cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
        else:
            cont = np.ones((cropsize, cropsize), img.dtype)*f
        cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
        new_images.append(cont)

    if len(new_images) == 1:
        new_images = new_images[0]

    return new_images


def top_left_crop(img, cropsize, default_value):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[:ch, :cw] = img[:ch, :cw]

    return container


def center_crop(img, cropsize, default_value=0):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    sh = h - cropsize
    sw = w - cropsize

    if sw > 0:
        cont_left = 0
        img_left = int(round(sw / 2))
    else:
        cont_left = int(round(-sw / 2))
        img_left = 0

    if sh > 0:
        cont_top = 0
        img_top = int(round(sh / 2))
    else:
        cont_top = int(round(-sh / 2))
        img_top = 0

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        img[img_top:img_top+ch, img_left:img_left+cw]

    return container


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


def pil_blur(img, radius):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius)))


def random_blur(img):
    radius = random.random()
    # print('add blur: ', radius)
    if isinstance(img, list):
        out = []
        for im in img:
            out.append(pil_blur(im, radius))
        return out
    elif isinstance(img, np.ndarray):
        return pil_blur(img, radius)
    else:
        print(img)
        raise RuntimeError("do not support the input image type!")


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(np.array(image_numpy,dtype=np.uint8))
    image_pil.save(image_path)


def im2arr(img_path, mode=1, dtype=np.uint8):
    
    if mode==1:
        img = PIL.Image.open(img_path)
        arr = np.asarray(img, dtype=dtype)
    else:
        arr = tifffile.imread(img_path)
        if arr.ndim == 3:
            a, b, c = arr.shape
            if a < b and a < c:  
                arr = arr.transpose([1,2,0])
    return arr

### trainer.py

class CDTrainer():

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        #print(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                     momentum=0.9,
                                     weight_decay=5e-4)

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs
        self.acc = 0
        self.miou = 0
        self.mf1 = 0
        self.iou_0 = 0
        self.iou_1 = 0
        self.F1_0 = 0
        self.F1_1 = 0
        self.precision_0 = 0
        self.precision_1 = 0
        self.recall_0 = 0
        self.recall_1 = 0

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        self.load_checkpoints = args.load_checkpoints
        self.continue_computation = args.continue_computation
        
        
        # define the loss functions
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        #if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
        if (os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)) & (self.continue_computation == False) & (len(self.load_checkpoints) == 0)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
        elif ((os.path.exists(self.load_checkpoints)) & (len(self.load_checkpoints) > 0)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(self.load_checkpoints,
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            if(self.continue_computation == True):
                self.epoch_to_start = checkpoint['epoch_id'] + 1
                self.best_val_acc = checkpoint['best_val_acc']
                self.best_epoch_id = checkpoint['best_epoch_id']
                self.continue_computation == False
                
                src_dir = os.path.dirname(self.load_checkpoints)
                dst_dir = self.checkpoint_dir
                # Copy the last model found
                shutil.copyfile(os.path.join(src_dir, 'last_ckpt.pt'), os.path.join(dst_dir, 'last_ckpt.pt'))
                # Copy the best model found
                shutil.copyfile(os.path.join(src_dir, 'best_ckpt.pt'), os.path.join(dst_dir, 'best_ckpt.pt'))
                # Copy the train acc curve
                shutil.copyfile(os.path.join(src_dir, 'train_acc.npy'), os.path.join(dst_dir, 'train_acc.npy'))
                # Copy the val acc curve
                shutil.copyfile(os.path.join(src_dir, 'val_acc.npy'), os.path.join(dst_dir, 'val_acc.npy'))
            
            else:
                self.epoch_to_start = 0
                self.best_val_acc = 0
                self.best_epoch_id = 0
                
                src_dir = os.path.dirname(self.load_checkpoints)
                dst_dir = self.checkpoint_dir
                shutil.copyfile(os.path.join(src_dir, 'best_ckpt.pt'), os.path.join(dst_dir, 'best_ckpt.pt'))

                #### update learning rate scheduler and loss function
                # Learning rate and Beta1 for Adam optimizers
                self.lr = args.lr

                # define optimizers
                self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                            momentum=0.9,
                                            weight_decay=5e-4)

                # define lr schedulers
                self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

                
                # define the loss functions
                if args.loss == 'ce':
                    self._pxl_loss = cross_entropy
                elif args.loss == 'bce':
                    self._pxl_loss = losses.binary_ce
                else:
                    raise NotImplemented(args.loss)
                #####################################


            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch
            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
            self.load_checkpoints = ''
        elif ((self.continue_computation == True) & (len(self.load_checkpoints) == 0)):
            print("Please provide model file to load and continue computation.")
            exit()
        elif ((self.continue_computation == True) and not (os.path.exists(self.load_checkpoints)) and (len(self.load_checkpoints) > 0)):
            print("Checkpoint file does not exist.")
            exit()
        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))
                
        single_image = torch.randn(3, 256, 512, dtype=torch.float64).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        single_image = single_image.to(device) 
        
        with torch.no_grad():  # No gradient tracking needed
            prediction = self.net_G(single_image)
            
        single_image_np = single_image.cpu().numpy()
        prediction_np = prediction.cpu().detach().numpy()
        
        signature_ = infer_signature(single_image_np, prediction_np)
        mlflow.pytorch.log_model(self.net_G, "urbanisation_model", signature=signature_)
                

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     imps*self.batch_size, est,
                     self.G_loss.item(), running_acc)
            self.logger.write(message)


        if np.mod(self.batch_id, 500) == 1:
            #print('***'); print(self.batch['input_image'].shape); print('***')
            width = self.batch['input_image'].shape[-2]
            vis_input, vis_input2 = make_numpy_grid(de_norm(self.batch['input_image'][:, :, :, :width])), make_numpy_grid(de_norm(self.batch['input_image'][:, :, :, width:]))
        
            #vis_input = make_numpy_grid(de_norm(self.batch['A']))
            #vis_input2 = make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = make_numpy_grid(self._visualize_pred())

            vis_gt = make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                              str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores(); #print('scores: '); print(scores)
        self.epoch_acc = scores['mf1']
        self.acc = scores['acc']
        self.miou = scores['miou']
        self.mf1 = scores['mf1']
        self.iou_0 = scores['iou_0']
        self.iou_1 = scores['iou_1']
        self.F1_0 = scores['F1_0']
        self.F1_1 = scores['F1_1']
        self.precision_0 = scores['precision_0']
        self.precision_1 = scores['precision_1']
        self.recall_0 = scores['recall_0']
        self.recall_1 = scores['recall_1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')
                        
    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()


    def _forward_pass(self, batch):
        self.batch = batch
        #img_in1 = batch['A'].to(self.device)
        #img_in2 = batch['B'].to(self.device)
        img_in = batch['input_image'].to(self.device)
        #self.G_pred = self.net_G(img_in1, img_in2)
        #print('---'); print(img_in.shape); print('---')
#        torch.save(img_in, 'img_test')
        self.G_pred = self.net_G(img_in)


    def _backward_G(self):
        gt = self.batch['L'].to(self.device).long()
        self.G_loss = self._pxl_loss(self.G_pred, gt)
        self.G_loss.backward()


    def train_models(self):
        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass(batch)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._timer_update()

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()
            
            # save training accuracy
            
            mlflow.log_metric("training_acc", self.acc, step=self.epoch_id)
            mlflow.log_metric("training_miou", self.miou, step=self.epoch_id)
            mlflow.log_metric("training_mf1", self.mf1, step=self.epoch_id)
            mlflow.log_metric("training_iou_0", self.iou_0, step=self.epoch_id)
            mlflow.log_metric("training_iou_1", self.iou_1, step=self.epoch_id)
            mlflow.log_metric("training_F1_0", self.F1_0, step=self.epoch_id)
            mlflow.log_metric("training_F1_1", self.F1_1, step=self.epoch_id)
            mlflow.log_metric("training_precision_0", self.precision_0, step=self.epoch_id)
            mlflow.log_metric("training_precision_1", self.precision_1, step=self.epoch_id)
            mlflow.log_metric("training_recall_0", self.recall_0, step=self.epoch_id)
            mlflow.log_metric("training_recall_1", self.recall_1, step=self.epoch_id)

            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            # save validation accuracy
            mlflow.log_metric("validation_acc", self.acc, step=self.epoch_id)
            mlflow.log_metric("validation_miou", self.miou, step=self.epoch_id)
            mlflow.log_metric("validation_mf1", self.mf1, step=self.epoch_id)
            mlflow.log_metric("validation_iou_0", self.iou_0, step=self.epoch_id)
            mlflow.log_metric("validation_iou_1", self.iou_1, step=self.epoch_id)
            mlflow.log_metric("validation_F1_0", self.F1_0, step=self.epoch_id)
            mlflow.log_metric("validation_F1_1", self.F1_1, step=self.epoch_id)
            mlflow.log_metric("validation_precision_0", self.precision_0, step=self.epoch_id)
            mlflow.log_metric("validation_precision_1", self.precision_1, step=self.epoch_id)
            mlflow.log_metric("validation_recall_0", self.recall_0, step=self.epoch_id)
            mlflow.log_metric("validation_recall_1", self.recall_1, step=self.epoch_id)
            self._update_checkpoints()
            
            """
            if self.epoch_acc == self.best_val_acc:
                
                single_image = torch.randn(3, 256, 512, dtype=torch.float64).unsqueeze(0)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                single_image = single_image.to(device) 
                
                with torch.no_grad():  # No gradient tracking needed
                    prediction = self.net_G(single_image)
                    
                single_image_np = single_image.cpu().numpy()
                prediction_np = prediction.cpu().detach().numpy()
                
                signature = infer_signature(single_image_np, prediction_np)
                mlflow.pytorch.log_model(self.net_G, "urbanisation_model", signature=signature)
            """
            
### resnet.py

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            dilation = 1
            # raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_2(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=None):
        super(ResNet_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.strides = strides
        if self.strides is None:
            self.strides = [2, 2, 2, 2, 2]

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=self.strides[0], padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=self.strides[2],
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=self.strides[3],
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=self.strides[4],
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_2(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

### networks.py

"""
Helper Functions
##############################################################################
"""


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptionsÄÄ½Å½Äƒâ‚¬â‚¬
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned')

    elif args.net_G == 'base_transformer_pos_s4_dd8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)

    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
# ##############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = resnet18(pretrained=True, replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = resnet34(pretrained=True, replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = resnet50(pretrained=True, replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        x = TF.normalize(x, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        # Split the input into x1 and x2 along the width dimension
        width = x.shape[-1]
        #x1, x2 = x[:, :, :width, :], x[:, :, width:, :]
        x1, x2 = x[:, :, :, :width], x[:, :, :, width:]
        
        # Process each half of the image separately
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        
        # Take the absolute difference between the processed halves
        x = torch.abs(x1 - x2)
        
        # Upsampling as per configuration
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        
        # Classification layer
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        #print('---'); print(x.shape); print('---')
        x = self.resnet.conv1(x)
        #print('***'); print(x.shape); print('***')
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        
        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256
        
        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzierÄÄ½Åšthen downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        #if with_pos is 'learned':
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        #if self.pool_mode is 'max':
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        #elif self.pool_mode is 'ave':
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x):
    
        x = TF.normalize(x, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        
        # Split the input image into x1 and x2 along the width dimension
        width = x.shape[-2]
        x1, x2 = x[:, :, :, :width], x[:, :, :, width:]
        # Process x1 and x2 as in the original forward method

        # Ensure that x1 and x2 are of dtype float32
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
    
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        
        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

### losses.py

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


### help_funcs.py

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x

### evaluator.py


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        #print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

        self.pred_dir = args.checkpoint_dir + '/prediction'
        os.makedirs(self.pred_dir, exist_ok=True)

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        if np.mod(self.batch_id, 100) == 1:
        
            width = self.batch['input_image'].shape[-2]
            vis_input, vis_input2 = make_numpy_grid(de_norm(self.batch['input_image'][:, :, :, :width])), make_numpy_grid(de_norm(self.batch['input_image'][:, :, :, width:]))

            #vis_input = make_numpy_grid(de_norm(self.batch['A']))
            #vis_input2 = make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = make_numpy_grid(self._visualize_pred())

            vis_gt = make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        #img_in1 = batch['A'].to(self.device)
        #img_in2 = batch['B'].to(self.device)
        #self.G_pred = self.net_G(img_in1, img_in2)
        
        img_in = batch['input_image'].to(self.device)
        
        #self.G_pred = self.net_G(img_in1, img_in2)
        #print('---'); print(img_in.shape); print('---')
        self.G_pred = self.net_G(img_in)
        

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()  

    def _save_predictions(self):
        
        preds = self._visualize_pred()
        name = self.batch['name']
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                #self.pred_dir, name[i].replace('.jpg', '.png'))
                self.pred_dir, name[i].replace('.jpg', '.png'))
            pred = pred[0].cpu().numpy()
            save_image(pred, file_name)
    
### CD_dataset.py

#IMG_FOLDER_NAME = "A"
#IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"
INPUT_IMAGE_FOLDER_NAME = "one_input"

IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    #img_name_list = np.loadtxt(dataset_path, dtype=np.str)
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


#def get_img_post_path(root_dir,img_name):
#    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


#def get_img_path(root_dir, img_name):
#    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)

def get_img_input_path(root_dir, img_name):
    return os.path.join(root_dir, INPUT_IMAGE_FOLDER_NAME, img_name)

def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))


class ImageDataset(data.Dataset):
#class ImageDataset(Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        # self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.list + '.txt'
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)

        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
#                with_scale_random_crop=True,
                with_scale_random_crop=False,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        input_image_path = get_img_input_path(self.root_dir, self.img_name_list[index % self.A_size])
        
        combined_image = Image.open(input_image_path)

        # Convert the split images into NumPy arrays
        img = np.asarray(combined_image)

        [img], _ = self.augm.transform([img],[], to_tensor=self.to_tensor)
        
        return {'input_image': img, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):
    
    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        
        input_image_path = get_img_input_path(self.root_dir, self.img_name_list[index % self.A_size])
        
        combined_image = Image.open(input_image_path)

        # Convert the split images into NumPy arrays
        img = np.asarray(combined_image)
        
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])
        
        label = np.array(Image.open(L_path), dtype=np.uint8)
        if self.label_transform == 'norm':
            label = label // 255
        
        [img], [label] = self.augm.transform([img], [label], to_tensor=self.to_tensor)
        
        return {'name': name, 'input_image': img, 'L': label}

class CDDataset_predict(ImageDataset):
    
    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset_predict, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]

        input_image_path = get_img_input_path(self.root_dir, self.img_name_list[index % self.A_size])
        
        combined_image = Image.open(input_image_path)
        
        # Convert the split images into NumPy arrays
        img = np.asarray(combined_image)

        [img], _ = self.augm.transform([img],[], to_tensor=self.to_tensor)
        return {'input_image': img, 'name': name}

### data_utils.py 

def to_tensor_and_norm(imgs, labels):
    # to tensor
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            for img in imgs]
    return imgs, labels


class CDDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [TF.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            #print('***'); print(imgs[0].size); print('***')
            if imgs[0].size != (self.img_size, 2*self.img_size):
                imgs = [TF.resize(img, [self.img_size, 2*self.img_size], interpolation=3)
                        for img in imgs]
            #    print('---'); print(imgs[0].size); print('---')
        else:
            self.img_size = imgs[0].size[0]

        labels = [TF.to_pil_image(img) for img in labels]
        if len(labels) != 0:
            if labels[0].size != (self.img_size, self.img_size):
                labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
                        for img in labels]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0], scale=(0.8, 1.0), ratio=(1, 1))

            imgs = [TF.resized_crop(img, i, j, h, w,
                                    size=(self.img_size, self.img_size),
                                    interpolation=Image.CUBIC)
                    for img in imgs]

            labels = [TF.resized_crop(img, i, j, h, w,
                                      size=(self.img_size, self.img_size),
                                      interpolation=Image.NEAREST)
                      for img in labels]

        if self.with_scale_random_crop:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                    for img in labels]

        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                      for img in labels]
            
            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                    for img in imgs]
                        
        return imgs, labels


def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)

### main_cd

def train(args):
    #dataloaders = utils.get_loaders(args);  
    dataloaders = get_loaders(args);  
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    # from models.evaluator import CDEvaluator
    dataloader = get_loader(args.data_name, img_size=args.img_size,
                            batch_size=args.batch_size, is_train=False,
                            split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()
    
    ###############################
    """
    #print(model.epoch_acc); print(model.best_val_acc);
    if model.epoch_acc > model.best_val_acc:
        data_iter = iter(dataloader)
        images = next(data_iter)
        
        # Get the first image from the batch
        single_image = np.array(images['input_image'][0])  # Shape will be [channels, height, width]
        
        single_image = torch.from_numpy(single_image)  # Convert to tensor

        # Add a batch dimension using unsqueeze
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        single_image = single_image.unsqueeze(0).to(device) 
        
        with torch.no_grad():  # No gradient tracking needed
            prediction = model.net_G(single_image)
            
        single_image_np = single_image.cpu().numpy()
        prediction_np = prediction.cpu().detach().numpy()
        
        signature = infer_signature(single_image_np, prediction_np)
        mlflow.pytorch.log_model(model.net_G, "urbanisation_model", signature=signature)
    """
    ###############################

def predict(args, data):
    
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    
    #os.makedirs(args.output_folder, exist_ok=True)
    #os.makedirs(data.root_dir + "/predict", exist_ok=True)
    
    log_path = os.path.join(data.root_dir + "/predict", 'log_vis.txt')
    
    data_loader = get_loader_predict(args.data_name, img_size=args.img_size, batch_size=args.batch_size, split=args.split, is_train=False)
    
    model = CDEvaluator(args=args, dataloader=data_loader)
    
    #if os.path.exists(os.path.join(data.root_dir + 'best_ckpt.pt')):
    if os.path.exists(args.load_checkpoints):
        model.logger.write('loading best checkpoint...\n')
        # load the entire checkpoint
        
        #checkpoint = torch.load(data.root_dir + 'best_ckpt.pt', map_location=model.device)
        checkpoint = torch.load(args.load_checkpoints, map_location=model.device)

        model.net_G.load_state_dict(checkpoint['model_G_state_dict'])

        model.net_G.to(model.device)

        # update some other states
        model.best_val_acc = checkpoint['best_val_acc']
        model.best_epoch_id = checkpoint['best_epoch_id']

        model.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                            (model.best_val_acc, model.best_epoch_id))
        model.logger.write('\n')

        model.net_G.eval()
        for i, batch in enumerate(data_loader):
            name = batch['name']
            print('process: %s' % name)
            score_map = model._forward_pass(batch)
            model._save_predictions()
        

def create_example_dataset(source_dir, dest_dir, list_name, num_files=20):

    # get filenames
    with open(source_dir + '/list/' + list_name, 'r') as file:
        filenames = [line.strip() for line in file.readlines()]

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir + '/A'):
        os.makedirs(dest_dir + '/A')
    
    if not os.path.exists(dest_dir + '/B'):
        os.makedirs(dest_dir + '/B')
    
    if not os.path.exists(dest_dir + '/label'):
        os.makedirs(dest_dir + '/label')

    if not os.path.exists(dest_dir + '/list'):
        os.makedirs(dest_dir + '/list')

    # Select random filenames
    random_filenames = random.sample(filenames, min(num_files, len(filenames)))
    
    # Copy selected random files to the destination directory
    for filename in random_filenames:
        file_path = os.path.join(source_dir + '/A/', filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, dest_dir + '/A/')

        file_path = os.path.join(source_dir + '/B/', filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, dest_dir + '/B/')

        file_path = os.path.join(source_dir + '/label/', filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, dest_dir + '/label/')

     # Write the selected filenames to a text file
    with open(os.path.join(dest_dir + '/list/', list_name), 'a') as file:
        file.write('\n'.join(random_filenames))


if __name__ == '__main__':
    
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)
    parser.add_argument('--create_example_dataset', action='store_true', help='Create sample dataset from data.')

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    #parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8', type=str,
    #                    help='base_resnet18 | base_transformer_pos_s4 | '
    #                         'base_transformer_pos_s4_dd8 | '
    #                         'base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--loss', default='ce', type=str)

    #parser.add_argument('--load_checkpoints', default=False, type=bool)
    #parser.add_argument('--load_checkpoints', action='store_true', help='Load already computed model (default is disabled).')
    #parser.add_argument('--load_checkpoints_name', type=str, help='Name of the model, that should be loaded.')
    parser.add_argument('--load_checkpoints', type=str, help='Name of the model, that should be loaded.', default='')
    parser.add_argument('--run_no', default=0, type=int)
    parser.add_argument('--continue_computation', action="store_true", help='Start epoch counting from epoch countion of loaded model')

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)
    parser.add_argument('--predict', action='store_true', help='Use model for prediction (default is disabled).')

    args = parser.parse_args()
    args.net_G = 'base_transformer_pos_s4_dd8_dedim8'

    data = DataConfig().get_data_config(data_name=args.data_name)

    #utils.get_device(args)
    get_device(args)
    
    """
    if args.load_checkpoints == False:
        if args.run_no == 0:
            max_number = 0 # Initialize max_number to None
            if os.path.exists(os.path.join('checkpoints', args.project_name)) and os.path.isdir(os.path.join('checkpoints', args.project_name)):
                subdirectories = [d for d in os.listdir(os.path.join('checkpoints', args.project_name)) if os.path.isdir(os.path.join(os.path.join('checkpoints', args.project_name), d))]
                for subdirectory in subdirectories:
                    numbers = [int(match) for match in re.findall(r'\d+', subdirectory)]  # Convert matches to integers
                    if numbers:
                        max_in_subdirectory = max(numbers)
                        if max_in_subdirectory > max_number:
                            max_number = max_in_subdirectory
                #print(str(max_number))
        else:
            max_number = args.run_no-1
            if os.path.exists(os.path.join(os.getcwd(), 'checkpoints', args.project_name, 'run_' + str(max_number+1))) and os.path.isdir(os.path.join(os.getcwd(), 'checkpoints', args.project_name, 'run_' + str(max_number+1))):
                print("This run number already exists. Please provide another run number. Computation stops.")
                exit()
    else:
        if args.run_no == 0:
            print("Please provide number of run to load checkpoints from. Computation stops.")
            exit()
        else:
            max_number = args.run_no-1
            if not (os.path.exists(os.path.join('checkpoints', args.project_name, 'run_' + str(max_number+1))) and os.path.isdir(os.path.join('checkpoints', args.project_name, 'run_' + str(max_number+1)))):
                print("This run number does not exist. Please provide existing run number. Computation stops.")
                exit()
    """
    
    if args.run_no == 0:
        max_number = 0 # Initialize max_number to None
        if os.path.exists(os.path.join('checkpoints', args.project_name)) and os.path.isdir(os.path.join('checkpoints', args.project_name)):
            subdirectories = [d for d in os.listdir(os.path.join('checkpoints', args.project_name)) if os.path.isdir(os.path.join(os.path.join('checkpoints', args.project_name), d))]
            for subdirectory in subdirectories:
                numbers = [int(match) for match in re.findall(r'\d+', subdirectory)]  # Convert matches to integers
                if numbers:
                    max_in_subdirectory = max(numbers)
                    if max_in_subdirectory > max_number:
                        max_number = max_in_subdirectory
            #print(str(max_number))
    else:
        max_number = args.run_no-1
        if os.path.exists(os.path.join(os.getcwd(), 'checkpoints', args.project_name, 'run_' + str(max_number+1))) and os.path.isdir(os.path.join(os.getcwd(), 'checkpoints', args.project_name, 'run_' + str(max_number+1))):
            print("This run number already exists. Please provide another run number. Computation stops.")
            exit()
    
    if len(args.load_checkpoints) > 0:
        if not os.path.exists(args.load_checkpoints):
            print("Model file does not exists. Please provide correct path to model.")
            exit()

    args.project_name = os.path.join(args.project_name, 'run_' + str(max_number+1))
    
    
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    if args.create_example_dataset:
        create_example_dataset(data.root_dir, args.project_name + '/example', 'test.txt', num_files=10)
        create_example_dataset(data.root_dir, args.project_name + '/example', 'train.txt', num_files=10)
        create_example_dataset(data.root_dir, args.project_name + '/example', 'val.txt', num_files=5)

    #with open(args.project_name + '/example/list/test.txt', 'r') as file:
    #    # Read the first line
    #    example_image = file.readline().strip()

    #signature = infer_signature(np.asarray(Image.open(args.project_name + '/example/A/' + example_image)),np.asarray(Image.open(args.project_name + '/example/A/' + example_image)))

#    mlflow.set_tracking_uri("https://69:3fJ7RKAo9CaRanQfG7mkw_gCt_fP5qrxhFqVcl3MTzA@mlflow.developer.workspaces.onda-dev.ai-pipeline.org/")
#    mlflow.set_experiment("Urbanisation")
    with mlflow.start_run():
        # print('mlflow.is_tracking_uri_set():' + str(mlflow.is_tracking_uri_set()))
        # Log tags
        mlflow.set_tag("gpu_ids", args.gpu_ids)
        mlflow.set_tag("project_name", args.project_name)
        mlflow.set_tag("checkpoint_root", args.checkpoint_root)
        
        # Log parameters
        mlflow.log_param("num_workers", args.num_workers)
        mlflow.log_param("dataset", args.dataset)
        mlflow.log_param("data_name", args.data_name)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("split", args.split)
        mlflow.log_param("split_val", args.split_val)
        mlflow.log_param("img_size", args.img_size)
        mlflow.log_param("n_class", args.n_class)
        mlflow.log_param("net_G", args.net_G)
        mlflow.log_param("loss", args.loss)
        mlflow.log_param("optimizer", args.optimizer)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("max_epochs", args.max_epochs)
        mlflow.log_param("learning_rate_policy", args.lr_policy)
        mlflow.log_param("learning_rate_decay_iters", args.lr_decay_iters)
        mlflow.log_param("predict", args.predict)
        mlflow.log_param("load_checkpoints", args.load_checkpoints)
        
#        mlflow.log_artifacts(args.project_name + '/example')
        #mlflow.log_artifacts(data.root_dir)
#        mlflow.log_artifacts(data.root_dir + "/list")
#        mlflow.log_artifacts("datasets/test_list")
        #mlflow.log_artifact(data.root_dir + "/create_levir_cd_dataset.py")
#        mlflow.log_artifact("requirements.txt")
        
        if args.predict == False:
            train(args)
            test(args)
            mlflow.log_artifact(args.checkpoint_dir + "/best_ckpt.pt")
            mlflow.log_artifact(args.checkpoint_dir + "/last_ckpt.pt")
        else:
            predict(args, data)
            mlflow.log_artifact(args.checkpoint_dir)
            
