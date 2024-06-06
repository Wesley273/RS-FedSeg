# -*- coding: UTF-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import ssl
from collections import defaultdict

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import utils as smp_utils
from torch.utils.data import DataLoader

from config import Config
from Fed import FedAvg
from my_utils.data_augmentation import (augment_train, augment_validation,
                                        preprocessing)

if torch.cuda.is_available():
    DEVICE = torch.device(Config.device)
else:
    DEVICE = torch.device('cpu')

ssl._create_default_https_context = ssl._create_unverified_context


def local_train(net_local, data_dir, client):

    # 训练集
    x_train_dir = os.path.join(data_dir, 'train')
    y_train_dir = os.path.join(data_dir, 'trainannot')

    # 验证集
    x_valid_dir = os.path.join(data_dir, 'val')
    y_valid_dir = os.path.join(data_dir, 'valannot')

    pretrained_fn = smp.encoders.get_preprocessing_fn(Config.encoder, Config.encoder_weights)

    # 加载训练数据集
    train_dataset = Config.dataset(
        x_train_dir,
        y_train_dir,
        augmentation=augment_train(),
        preprocessing=preprocessing(pretrained_fn),
        classes=Config.classes,
    )

    # 加载验证数据集
    valid_dataset = Config.dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=augment_validation(),
        preprocessing=preprocessing(pretrained_fn),
        classes=Config.classes,
    )

    # 需根据显卡的性能进行设置，batch_size为每次迭代中一次训练的图片数，num_workers为训练时的工作进程数，如果显卡不太行或者显存空间不够，将batch_size调低并将num_workers调为0
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    loss = smp_utils.losses.DiceLoss()
    metrics = [smp_utils.metrics.IoU(threshold=0.5)]

    optimizer = torch.optim.Adam(params=net_local.parameters(), lr=Config.lr)

    # 创建一个epoch，用于迭代数据样本
    train_epoch = smp_utils.train.TrainEpoch(
        net_local,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp_utils.train.ValidEpoch(
        net_local,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # 进行Config.client_epoch轮次迭代的模型训练
    max_score = 0
    train_logs = {}
    val_logs = {}
    result_path = os.path.join('result', Config.data_name, 'client_{}'.format(client))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for i in range(0, Config.client_epoch):
        print('Client epoch: {}'.format(i))
        train_logs[i] = train_epoch.run(train_loader)
        val_logs[i] = valid_epoch.run(valid_loader)

        with open(os.path.join(result_path, 'train_logs.json'), "w") as file:
            json.dump(train_logs, file)
        with open(os.path.join(result_path, 'val_logs.json'), "w") as file:
            json.dump(val_logs, file)

        # 保存当前轮次模型
        torch.save(net_local, os.path.join(result_path, 'latest_model.pth'))
        print('Latest Model saved!')
        # 保存最好的模型
        if max_score < val_logs[i]['iou_score']:
            max_score = val_logs[i]['iou_score']
            best_net_local = net_local
            torch.save(best_net_local, os.path.join(result_path, 'best_model.pth'))
            print('Best Model saved!')

    return best_net_local.state_dict(), train_logs, val_logs


def global_val(data_dir, net_global):

    # 验证集
    x_valid_dir = os.path.join(data_dir, 'val')
    y_valid_dir = os.path.join(data_dir, 'valannot')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(Config.encoder, Config.encoder_weights)

    # 加载验证数据集
    valid_dataset = Config.dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=augment_validation(),
        preprocessing=preprocessing(preprocessing_fn),
        classes=Config.classes,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    loss = smp_utils.losses.DiceLoss()
    metrics = [smp_utils.metrics.IoU(threshold=0.5)]

    # 创建一个epoch，用于迭代数据样本
    valid_epoch = smp_utils.train.ValidEpoch(
        net_global,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    val_logs = {}
    result_path = os.path.join('result', Config.data_name, 'global')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print('Valid net global'.format(i))
    val_logs = valid_epoch.run(valid_loader)
    with open(os.path.join(result_path, 'val_logs.json'), "w") as file:
        json.dump(val_logs, file)
    # 保存当前全局模型
    torch.save(net_global, os.path.join(result_path, 'global_model.pth'))
    print('Global Model saved!')
    return val_logs


if __name__ == '__main__':
    local_w = defaultdict(dict)
    local_train_log = defaultdict(dict)
    local_val_log = defaultdict(dict)
    global_val_log = defaultdict(dict)
    net_global = smp.UnetPlusPlus(
        encoder_name=Config.encoder,
        encoder_weights=Config.encoder_weights,
        classes=len(Config.classes),
        activation=Config.activation,
    )
    # 总联邦学习训练轮次
    for e in range(Config.epoch):
        print('Epoch: {}'.format(e))
        net_global.train()
        # 每个client在本地训练一定轮次
        for i in range(1, Config.region_num + 1):
            print('----Client {} local train----'.format(i))
            local_w[e][i], local_train_log[e][i], local_val_log[e][i] = local_train(net_global, Config.get_data_dir(i), i)
        # 模型聚合
        w_avg = FedAvg(local_w[e])
        net_global.load_state_dict(w_avg)
        net_global.eval()
        global_val_log[e] = global_val(Config.get_data_dir(0), net_global)

    # 保存数据
    result_path = os.path.join('result', Config.data_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(os.path.join(result_path, 'local_train_logs.json'), "w") as file:
        json.dump(local_train_log, file)
    with open(os.path.join(result_path, 'local_val_logs.json'), "w") as file:
        json.dump(local_val_log, file)
    with open(os.path.join(result_path, 'global_val_logs.json'), "w") as file:
        json.dump(global_val_log, file)
