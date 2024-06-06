# -*- coding: UTF-8 -*-
import os

from datasets import BH_POOL, BH_WATERTANK, CamVid

data_dict = {'CamVid': {'dataset': CamVid, 'regions': 1},
             'BH_POOL': {'dataset': BH_POOL, 'regions': 8},
             'MINI_BH_POOL': {'dataset': BH_POOL, 'regions': 8},
             'BH_WATERTANK': {'dataset': BH_WATERTANK, 'regions': 8}
             }


class Config:
    # 网络训练参数
    device = 'cuda'
    batch_size = 2
    lr = 0.0001
    num_workers = 0
    epoch = 5
    client_epoch = 2

    # 数据集加载
    data_name = 'MINI_BH_POOL'
    dataset = data_dict[data_name]['dataset']
    region_num = data_dict[data_name]['regions']

    # 网络结构
    encoder = 'se_resnext50_32x4d'
    encoder_weights = 'imagenet'
    classes = dataset.CLASSES
    activation = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

    @classmethod
    def get_data_dir(cls, client):
        if Config.data_name == 'CamVid':
            data_dir = os.path.join('data', cls.data_name)
        else:
            data_dir = os.path.join('data', cls.data_name, 'REGION_{}'.format(client))
        return data_dir
