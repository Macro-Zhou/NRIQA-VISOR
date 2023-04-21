
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from os.path import join
import os
import numpy as np
import random
import math
import scipy.io as scio
import csv
import cv2

def load_img(filepath, C='RGB'):
    with open(filepath, 'rb') as f:
        img = Image.open(f)
        return img.convert(C)

def random_crop(img, org, patchsize=224, gap=8):

    # H, W, C = img.shape
    H, W = img.size
    if H-patchsize-1 > 0:
        start_x = random.randrange(0, H-patchsize-1, gap)
    else:
        start_x = 0
    end_x = start_x + patchsize
    if W-patchsize-1 > 0:
        start_y = random.randrange(0, W-patchsize-1, gap)
    else:
        start_y = 0
    end_y = start_y + patchsize
    crop_box = [start_x, start_y, end_x, end_y]
    region_img = img.crop(crop_box)
    region_org = org.crop(crop_box)
    # region_img = img[start_x:end_x, start_y:end_y, :].copy()  # CV2
    # region_org = org[start_x:end_x, start_y:end_y, :].copy()

    return region_img, region_org


class augmentation(object):

    def __call__(self, img, org, tb_flg=True):
        if random.random() < 0.5:
            # img = img[:, ::-1, :]  # CV2
            # org = org[:, ::-1, :]
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            org = org.transpose(Image.FLIP_LEFT_RIGHT)
        if tb_flg:
            if random.random() < 0.5:
                # img = img[::-1, :, :]
                # org = org[::-1, :, :]
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                org = org.transpose(Image.FLIP_TOP_BOTTOM)

        return img, org


def image_file_name(txtfile, image_dir, org_dir):
    image_filenames = []
    org_filenames = []
    image_scores = []
    with open(txtfile, 'r') as f:
        for line in f:
            image_scores.append(float(line.split()[0]))
            image_filenames.append(image_dir+'//'+line.split()[1])
            org_filenames.append(org_dir+'//'+line.split()[1][:-9]+'.bmp')

    return image_filenames, org_filenames, image_scores


class ReadIQAFolder(data.Dataset):
    def __init__(self, matpath, nexp=0, aug=False, random_crop=False, img_transform=None, status='eval'):
        super(ReadIQAFolder).__init__()

        self.D = scio.loadmat(matpath[0])
        data_dir = matpath[1]
        exp = self.D['index'][nexp][:]

        if status == 'eval':
            colID = exp[int(len(exp)*0.8):]
            print('TEST INDEX:', end=' ')
            print(colID)
        elif status == 'full':
            colID = exp[:]
        else:
            colID = exp[0:int(len(exp)*0.8)]

        self.im_path = []
        self.ref_path = []
        self.scores = []
        self.dtype = []
        self.norm = matpath[2]

        for idx, im_name in enumerate(self.D['im_names']):
            if self.D['ref_ids'][idx].item() in colID:
                if status == 'train':
                    rep = 25
                else:
                    rep = 1
                for _ in range(rep):
                    self.im_path.append(join(data_dir, self.D['im_names'][idx][0].item()))
                    self.ref_path.append(join(data_dir, self.D['ref_names'][idx][0].item()))
                    # self.scores.append(self.D['subjective_scores'][idx].item() / self.norm)
                    self.scores.append(self.D['subjective_scores'][idx].item())
                    # self.dtype.append(self.D['dtype_ids'][idx].item())
                    self.dtype.append(0)
        self.transforms = img_transform
        self.aug = aug
        self.rcrop = random_crop

    def __getitem__(self, item):

        img = load_img(self.im_path[item].replace('\\', '/'))
        org = load_img(self.ref_path[item].replace('\\', '/'))

        if self.aug:
            img, org = augmentation()(img, org)
        if self.rcrop:
            img, org = random_crop(img, org, patchsize=224)

        mos = self.scores[item]
        mos = np.array(mos).astype(np.float32)

        if self.transforms:
            img = self.transforms(img)
            org = self.transforms(org)

        ids = self.dtype[item]

        return img, org, mos, ids

    def __len__(self):
        return len(self.im_path)



