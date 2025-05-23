# -*- coding: UTF-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssl

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from datasets import DataAug, Full

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
    region_data_dirs = [Config.get_data_dir(j) for j in range(1, Config.region_num + 1)]

    # 测试集
    img_dirs = [os.path.join(data_dir, 'img') for data_dir in region_data_dirs]
    mask_dirs = [os.path.join(data_dir, 'mask') for data_dir in region_data_dirs]

    # 加载最佳模型
    best_net = Config.get_net().to(DEVICE)
    best_net.load_state_dict(torch.load(os.path.join(Config.get_result_dir(), 'global', 'global_net.pth'), map_location=DEVICE))

    # 创建测试数据集
    test_dataset = Full(
        img_dirs,
        mask_dirs,
        augmentation=DataAug.augment_val(),
        preprocessing=DataAug.preprocessing(Config.preprocessing_fn)
    )

    # 用没有进行图像处理转化的测试集进行图像可视化展示
    test_dataset_vis = Full(img_dirs, mask_dirs)
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
