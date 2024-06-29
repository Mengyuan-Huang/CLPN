from torch.utils.data import Dataset
import torch
import glob
import numpy as np
import random
import os
import cv2
from pathlib import Path

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng', '.JPG']

class LoadImages_LOL(Dataset):
    def __init__(self, low_sRGB_path, img_size=(256, 256), augment=False, normalize=False,is_train=True):
        path = str(Path(low_sRGB_path))
        parent = str(Path(path).parent) + os.sep
        if os.path.isfile(path):  # file
            with open(path, 'r') as f:
                f = f.read().splitlines()
                f = [x.replace('./', parent) if x.startswith('./') else x for x in f]
        elif os.path.isdir(path):  # folder
            f = glob.iglob(path + os.sep + '*.*')
        else:
            raise Exception('%s does not exist' % path)

        self.low_sRGB_files = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        self.normal_sRGB_files = [x.replace('low', 'high') for x in self.low_sRGB_files]
        self.crop_shape = img_size
        self.augment = augment
        self.is_train = is_train

    def __len__(self):
        return len(self.low_sRGB_files)

    def __getitem__(self, index):
        low_sRGBs = cv2.imread(self.low_sRGB_files[index], flags=-1)
        normal_sRGBs = cv2.imread(self.normal_sRGB_files[index], flags=-1)
        low_sRGB = low_sRGBs / 255.
        normal_sRGB = normal_sRGBs / 255.

        if self.is_train:
            # 图像裁剪
            low_sRGB, normal_sRGB= crop_image(self.crop_shape,low_sRGB, normal_sRGB)
            if self.augment:
                # 随机翻转
                low_sRGB, normal_sRGB= random_flip(low_sRGB, normal_sRGB)

        low_sRGB = low_sRGB[:, :, ::-1].transpose(2, 0, 1)
        low_sRGB = np.ascontiguousarray(low_sRGB)
        normal_sRGB = normal_sRGB[:, :, ::-1].transpose(2, 0, 1)
        normal_sRGB = np.ascontiguousarray(normal_sRGB)

        return {'low_sRGB':torch.from_numpy(low_sRGB), 'normal_sRGB': torch.from_numpy(normal_sRGB),
                'low_sRGB_files': self.low_sRGB_files[index]}


class LoadImages_ME(Dataset):
    def __init__(self, low_sRGB_path, img_size=(256, 256), augment=False, normalize=False,is_train=True):
        path = str(Path(low_sRGB_path))
        parent = str(Path(path).parent) + os.sep
        if os.path.isfile(path):  # file
            with open(path, 'r') as f:
                f = f.read().splitlines()
                f = [x.replace('./', parent) if x.startswith('./') else x for x in f]
        elif os.path.isdir(path):  # folder
            f = glob.iglob(path + os.sep + '*.*')
        else:
            raise Exception('%s does not exist' % path)

        self.low_sRGB_files = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        normal_sRGB_files = [x.replace('INPUT_IMAGES', 'expert_c_testing_set') for x in self.low_sRGB_files]
        self.normal_sRGB_files = [x[:x.rfind('_')] + '.jpg' for x in normal_sRGB_files]
        self.crop_shape = img_size
        self.augment = augment
        self.is_train = is_train

    def __len__(self):
        return len(self.low_sRGB_files)

    def __getitem__(self, index):
        low_sRGBs = cv2.imread(self.low_sRGB_files[index], flags=-1)
        normal_sRGBs = cv2.imread(self.normal_sRGB_files[index], flags=-1)
        low_sRGB = low_sRGBs / 255.
        normal_sRGB = normal_sRGBs / 255.

        if self.is_train:
            # 图像裁剪
            low_sRGB, normal_sRGB= crop_image(self.crop_shape,low_sRGB, normal_sRGB)
            if self.augment:
                # 随机翻转
                low_sRGB, normal_sRGB= random_flip(low_sRGB, normal_sRGB)

        low_sRGB = low_sRGB[:, :, ::-1].transpose(2, 0, 1)
        low_sRGB = np.ascontiguousarray(low_sRGB)
        normal_sRGB = normal_sRGB[:, :, ::-1].transpose(2, 0, 1)
        normal_sRGB = np.ascontiguousarray(normal_sRGB)

        return {'low_sRGB':torch.from_numpy(low_sRGB), 'normal_sRGB': torch.from_numpy(normal_sRGB),
                'low_sRGB_files': self.low_sRGB_files[index]}

def crop_image(crop_shape, img1, img2):
    nw = random.randint(0, img1.shape[0] - crop_shape[0])
    nh = random.randint(0, img1.shape[1] - crop_shape[1])
    crop_img1 = img1[nw:nw + crop_shape[0], nh:nh + crop_shape[1], :]
    crop_img2 = img2[nw:nw + crop_shape[0], nh:nh + crop_shape[1], :]
    return crop_img1, crop_img2

def random_flip(img1, img2):
    mode = random.randint(0, 7)
    if mode == 0:
        # original
        return img1, img2
    elif mode == 1:
        # flip up and down
        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
    elif mode == 2:
        # rotate counterwise 90 degree 逆时针旋转90
        img1 = np.rot90(img1)
        img2 = np.rot90(img2)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        img1 = np.rot90(img1)
        img2 = np.rot90(img2)
        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
    elif mode == 4:
        # rotate 180 degree
        img1 = np.rot90(img1, k=2)
        img2 = np.rot90(img2, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        img1 = np.rot90(img1, k=2)
        img2 = np.rot90(img2, k=2)
        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
    elif mode == 6:
        # rotate 270 degree
        img1 = np.rot90(img1, k=3)
        img2 = np.rot90(img2, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        img1 = np.rot90(img1, k=3)
        img2 = np.rot90(img2, k=3)
        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
    return img1, img2
