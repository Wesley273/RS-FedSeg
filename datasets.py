# -*- coding: UTF-8 -*-
import os

import albumentations as albu
import cv2
import numpy as np
from torch.utils.data import Dataset


class Region(Dataset):
    """
    分布为NonIID的数据集，一般具有多个不同区域
    
    进行图像读取，图像增强增强和图像预处理

    Args:
        images_dir (str): 该区域图像所在路径\n
        masks_dir (str): 该区域标签所在路径\n
        augmentation (albumentations.Compose): 数据传输管道\n
        preprocessing (albumentations.Compose): 数据预处理\n
    """
    # 数据集中用于图像分割的所有标签类别
    CLASSES = ['target']

    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [255]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # 从标签中提取特定的类别 (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # 图像增强应用
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 图像预处理应用
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Full(Region):
    """ 
    分布为NonIID的数据集，一般具有多个不同区域，这里将其合并为一个完整数据集

    进行图像读取，图像增强增强和图像预处理

    Args:
        images_dirs (list(str)) : 各区域图像所在路径组成的列表\n
        masks_dirs (list(str)) : 各区域标签所在路径组成的列表\n
        augmentation (albumentations.Compose) : 数据传输管道\n
        preprocessing (albumentations.Compose) : 数据预处理\n
    """

    def __init__(
            self,
            images_dirs,
            masks_dirs,
            augmentation=None,
            preprocessing=None,
    ):
        self.images_fps = []
        self.masks_fps = []

        for images_dir, masks_dir in zip(images_dirs, masks_dirs):
            ids = os.listdir(images_dir)
            images_fps = [os.path.join(images_dir, image_id) for image_id in ids]
            masks_fps = [os.path.join(masks_dir, image_id) for image_id in ids]

            self.images_fps.extend(images_fps)
            self.masks_fps.extend(masks_fps)

        # convert str names to class values on masks
        self.class_values = [255]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images_fps)


class DataAug():
    @classmethod
    def augment_train(cls):
        train_transform = [

            albu.HorizontalFlip(p=0.5),

            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            albu.PadIfNeeded(min_height=480, min_width=480),
            albu.RandomCrop(height=320, width=320, always_apply=True),

            albu.GaussNoise(p=0.2),
            albu.Perspective(p=0.5),

            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightnessContrast(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.Sharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)

    @classmethod
    def augment_val(cls):
        """调整图像使得图片的分辨率长宽能被32整除"""
        val_transform = [
            albu.PadIfNeeded(480, 480)
        ]
        return albu.Compose(val_transform)

    @classmethod
    def to_tensor(cls, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    @classmethod
    def preprocessing(cls, preprocessing_fn):
        """进行图像预处理操作

        Args:
            preprocessing_fn (callbale): 数据规范化的函数
                (针对每种预训练的神经网络)
        Return:
            transform: albumentations.Compose
        """

        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=cls.to_tensor, mask=cls.to_tensor),
        ]
        return albu.Compose(_transform)
