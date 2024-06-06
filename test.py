# -*- coding: UTF-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssl

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch

from config import Config
from datasets import BH_POOL, BH_WATERTANK, CamVid
from my_utils.data_augmentation import (augment_train, augment_validation,
                                        preprocessing)

if torch.cuda.is_available():
    DEVICE = torch.device(Config.device)
else:
    DEVICE = torch.device('cpu')

ssl._create_default_https_context = ssl._create_unverified_context

# 图像分割结果的可视化展示


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# ---------------------------------------------------------------
if __name__ == '__main__':

    DATA_DIR = Config.get_data_dir(0)

    # 测试集
    x_test_dir = os.path.join(DATA_DIR, 'test')
    y_test_dir = os.path.join(DATA_DIR, 'testannot')

    pretrained = smp.encoders.get_preprocessing_fn(Config.encoder, Config.encoder_weights)
    # ---------------------------------------------------------------
    # $# 测试训练出来的最佳模型

    # 加载最佳模型
    best_model = torch.load(os.path.join('result', Config.data_name, 'global', 'global_model.pth'))

    # 创建测试数据集
    test_dataset = Config.dataset(
        x_test_dir,
        y_test_dir,
        augmentation=augment_validation(),
        preprocessing=preprocessing(pretrained),
        classes=Config.classes,
    )

    # ---------------------------------------------------------------
    # $# 图像分割结果可视化展示
    # 对没有进行图像处理转化的测试集进行图像可视化展示
    test_dataset_vis = Config.dataset(
        x_test_dir, y_test_dir,
        classes=Config.classes,
    )
    # 从测试集中随机挑选3张图片进行测试
    for i in range(3):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )
