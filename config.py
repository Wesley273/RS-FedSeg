# -*- coding: UTF-8 -*-
import os

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils

from datasets import NonIID


class Config:
    # 网络训练参数
    device = 'cuda'
    batch_size = 8
    lr = 1e-4
    num_workers = 0
    epoch = 5
    client_epoch = 3

    # 数据集加载
    data_dict = {'BH_POOL': {'dataset': NonIID, 'regions': 8},
                 'MINI_BH_POOL': {'dataset': NonIID, 'regions': 8},
                 'BH_WATERTANK': {'dataset': NonIID, 'regions': 6},
                 'IAIL': {'dataset': NonIID, 'regions': 5}}

    data_name = 'BH_POOL'
    data_type = 'non_iid'
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
    def get_data_dir(cls, client):
        data_dir = os.path.join('data', cls.data_name, 'REGION_{}'.format(client))
        return data_dir

    @classmethod
    def get_result_dir(cls):
        return os.path.join('result', Config.data_name, cls.data_type)

    @classmethod
    def get_net(cls):
        return smp.Unet(encoder_name=cls.encoder,
                        encoder_weights=cls.encoder_weights,
                        classes=len(cls.classes),
                        activation=cls.activation)
