# -*- coding: UTF-8 -*-
import os

from datasets import BH_POOL, BH_WATERTANK, CamVid

dataset_dict = {'CamVid': CamVid, 'BH_POOL': BH_POOL, 'BH_WATERTANK': BH_WATERTANK}


def get_data_dir(data_name, client):
    if data_name == 'CamVid':
        data_dir = os.path.join('data', data_name)
    else:
        data_dir = os.path.join('data', data_name, 'REGION_{}'.format(client))
    return data_dir


class Config:
    # 网络训练参数
    device = 'cuda'
    batch_size = 2
    lr = 0.0001
    lr_low = 1e-5
    num_workers = 4

    # 数据集加载
    data_name = 'BH_WATERTANK'
    dataset = dataset_dict[data_name]
    client = 1
    data_dir = get_data_dir(data_name, client)

    # 网络结构
    encoder = 'se_resnext50_32x4d'
    encoder_weights = 'imagenet'
    classes = dataset.CLASSES
    activation = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
