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
    DATA_DIR = Config.get_data_dir(0)

    # 加载最佳模型
    a = torch.load(os.path.join('result', Config.data_name, 'global', 'global_net.pth'))
    result_path = os.path.join('result', Config.data_name, 'global')
    torch.save(a.state_dict(), os.path.join(result_path, 'global_net.pth'))
