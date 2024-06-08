# -*- coding: UTF-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import ssl

import torch
from segmentation_models_pytorch import utils as smp_utils
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader

from config import Config
from datasets import Full, DataAug

if torch.cuda.is_available():
    DEVICE = torch.device(Config.device)
else:
    DEVICE = torch.device('cpu')

ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == '__main__':
    net = Config.get_net()
    data_dirs = [Config.get_data_dir(j) for j in range(1, Config.region_num + 1)]
    train_dirs = [os.path.join(data_dir, 'train') for data_dir in data_dirs]
    train_annot_dirs = [os.path.join(data_dir, 'trainannot') for data_dir in data_dirs]
    val_dirs = [os.path.join(data_dir, 'val') for data_dir in data_dirs]
    val_annot_dirs = [os.path.join(data_dir, 'valannot') for data_dir in data_dirs]
    # 训练数据集
    train_dataset = Full(
        train_dirs,
        train_annot_dirs,
        augmentation=DataAug.augment_train(),
        preprocessing=DataAug.preprocessing(Config.preprocessing_fn)
    )
    # 加载验证数据集
    val_dataset = Full(
        val_dirs,
        val_annot_dirs,
        augmentation=DataAug.augment_val(),
        preprocessing=DataAug.preprocessing(Config.preprocessing_fn)
    )
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=Config.lr)

    # 创建一个epoch，用于迭代数据样本
    train_epoch = smp_utils.train.TrainEpoch(net, loss=Config.loss, metrics=Config.metrics, optimizer=optimizer, device=DEVICE, verbose=True)
    val_epoch = smp_utils.train.ValidEpoch(net, loss=Config.loss, metrics=Config.metrics, device=DEVICE, verbose=True)

    # 进行Config.epoch轮次迭代的模型训练
    max_score = 0
    train_logs = {}
    val_logs = {}
    for e in range(0, Config.epoch):
        print('#**************** Epoch: {} ****************#'.format(e))
        # with autocast():
        train_logs[e] = train_epoch.run(train_loader)
        val_logs[e] = val_epoch.run(val_loader)

        # 保存训练记录
        result_dir = os.path.join(Config.get_result_dir(), '..', 'un_fed')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        with open(os.path.join(result_dir, 'train_logs.json'), "w") as file:
            json.dump(train_logs, file)
        with open(os.path.join(result_dir, 'val_logs.json'), "w") as file:
            json.dump(val_logs, file)
        # 保存当前轮次模型
        torch.save(net.state_dict(), os.path.join(result_dir, 'latest_net.pth'))
        print('Latest net saved!')
        # 保存最好的模型
        if max_score < val_logs[e]['iou_score']:
            max_score = val_logs[e]['iou_score']
            best_net = net
            torch.save(best_net.state_dict(), os.path.join(result_dir, 'best_net.pth'))
            print('Best net saved!')
