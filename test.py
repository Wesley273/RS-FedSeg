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
from my_utils.data_augmentation import augment_val, preprocessing

if torch.cuda.is_available():
    DEVICE = torch.device(Config.device)
else:
    DEVICE = torch.device('cpu')

ssl._create_default_https_context = ssl._create_unverified_context


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


if __name__ == '__main__':
    DATA_DIR = Config.get_data_dir(client=0)

    # 测试集
    test_dir = os.path.join(DATA_DIR, 'test')
    testannot_dir = os.path.join(DATA_DIR, 'testannot')

    # 加载最佳模型
    best_net = smp.UnetPlusPlus(
        encoder_name=Config.encoder,
        encoder_weights=Config.encoder_weights,
        classes=len(Config.classes),
        activation=Config.activation,
    ).to(DEVICE)
    best_net.load_state_dict(torch.load(os.path.join('result', Config.data_name, 'global', 'global_net.pth')))

    # 创建测试数据集
    test_dataset = Config.dataset(
        test_dir,
        testannot_dir,
        augmentation=augment_val(),
        preprocessing=preprocessing(Config.preprocessing_fn),
        classes=Config.classes,
    )

    # 用没有进行图像处理转化的测试集进行图像可视化展示
    test_dataset_vis = Config.dataset(test_dir, testannot_dir, classes=Config.classes)
    # 从测试集中随机挑选3张图片进行测试
    for i in range(3):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_net.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(image=image_vis, ground_truth_mask=gt_mask, predicted_mask=pr_mask)
