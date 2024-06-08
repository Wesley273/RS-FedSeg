# -*- coding: UTF-8 -*-
import os

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils

from datasets import Region


class Config:
    # windows训练或linux服务器训练
    platform = 'win'
    # 网络训练参数
    device = 'cuda'
    batch_size = 4
    lr = 1e-4
    num_workers = 0
    epoch = 10
    client_epoch = 5

    # 数据集加载
    data_dict = {'BH_POOL': {'dataset': Region, 'regions': 8},
                 'MINI_BH_POOL': {'dataset': Region, 'regions': 8},
                 'BH_WATERTANK': {'dataset': Region, 'regions': 6},
                 'IAIL': {'dataset': Region, 'regions': 5}}

    data_name = 'IAIL'
    data_dist = None
    dataset = data_dict[data_name]['dataset']
    region_num = data_dict[data_name]['regions']

    # 网络结构与预训练参数
    encoder = 'se_resnext50_32x4d'
    encoder_weights = 'imagenet'
    classes = dataset.CLASSES
    activation = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    # 损失函数与评价指标
    loss = smp_utils.losses.DiceLoss()
    metrics = [smp_utils.metrics.IoU(),
               smp_utils.metrics.Accuracy(),
               smp_utils.metrics.Precision(),
               smp_utils.metrics.Fscore(),
               smp_utils.metrics.Recall()]

    @classmethod
    def get_data_dir(cls, region):
        if cls.platform == 'win':
            data_dir = os.path.join('data', cls.data_name, 'REGION_{}'.format(region))
        if cls.platform == 'linux':
            data_dir = os.path.join('/root', 'autodl-tmp', Config.data_name, 'REGION_{}'.format(region))
        return data_dir

    @classmethod
    def get_result_dir(cls):
        return os.path.join('result', Config.data_name, cls.data_dist)

    @classmethod
    def get_net(cls):
        return smp.Unet(encoder_name=cls.encoder,
                        encoder_weights=cls.encoder_weights,
                        classes=len(cls.classes),
                        activation=cls.activation)
