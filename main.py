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
from fed import FedAvg
from my_utils.data_augmentation import (augment_train, augment_val,
                                        preprocessing)

if torch.cuda.is_available():
    DEVICE = torch.device(Config.device)
else:
    DEVICE = torch.device('cpu')

ssl._create_default_https_context = ssl._create_unverified_context


def local_train(local_net, data_dir, client):
    # 训练集
    train_dir = os.path.join(data_dir, 'train')
    trainannot_dir = os.path.join(data_dir, 'trainannot')

    # 验证集
    val_dir = os.path.join(data_dir, 'val')
    valannot_dir = os.path.join(data_dir, 'valannot')

    # 加载训练数据集
    train_dataset = Config.dataset(
        train_dir,
        trainannot_dir,
        augmentation=augment_train(),
        preprocessing=preprocessing(Config.preprocessing_fn),
        classes=Config.classes,
    )

    # 加载验证数据集
    val_dataset = Config.dataset(
        val_dir,
        valannot_dir,
        augmentation=augment_val(),
        preprocessing=preprocessing(Config.preprocessing_fn),
        classes=Config.classes,
    )

    # 需根据显卡的性能进行设置，batch_size为每次迭代中一次训练的图片数，num_workers为训练时的工作进程数，如果显卡不太行或者显存空间不够，将batch_size调低并将num_workers调为0
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    optimizer = torch.optim.Adam(params=local_net.parameters(), lr=Config.lr)

    # 创建一个epoch，用于迭代数据样本
    train_epoch = smp_utils.train.TrainEpoch(local_net, loss=Config.loss, metrics=Config.metrics, optimizer=optimizer, device=DEVICE, verbose=True)
    val_epoch = smp_utils.train.ValidEpoch(local_net, loss=Config.loss, metrics=Config.metrics, device=DEVICE, verbose=True)

    # 进行Config.client_epoch轮次迭代的模型训练
    max_score = 0
    train_logs = {}
    val_logs = {}
    for i in range(0, Config.client_epoch):
        print('Client epoch: {}'.format(i))
        train_logs[i] = train_epoch.run(train_loader)
        val_logs[i] = val_epoch.run(val_loader)

        # 保存训练记录
        result_path = os.path.join('result', Config.data_name, 'client_{}'.format(client))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, 'train_logs.json'), "w") as file:
            json.dump(train_logs, file)
        with open(os.path.join(result_path, 'val_logs.json'), "w") as file:
            json.dump(val_logs, file)
        # 保存当前轮次模型
        torch.save(local_net, os.path.join(result_path, 'latest_net.pth'))
        print('Latest local net saved!')
        # 保存最好的模型
        if max_score < val_logs[i]['iou_score']:
            max_score = val_logs[i]['iou_score']
            best_local_net = local_net
            torch.save(best_local_net, os.path.join(result_path, 'best_net.pth'))
            print('Best local net saved!')

    return best_local_net.state_dict(), train_logs, val_logs


def global_val(data_dir, global_net):
    # 验证集
    val_dir = os.path.join(data_dir, 'val')
    val_annot_dir = os.path.join(data_dir, 'valannot')

    # 加载验证数据集
    valid_dataset = Config.dataset(
        val_dir,
        val_annot_dir,
        augmentation=augment_val(),
        preprocessing=preprocessing(Config.preprocessing_fn),
        classes=Config.classes,
    )
    val_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    # 创建一个epoch，用于迭代数据样本
    val_epoch = smp_utils.train.ValidEpoch(global_net, loss=Config.loss, metrics=Config.metrics, device=DEVICE, verbose=True)

    # 在全量数据上验证全局模型的性能
    val_logs = {}
    print('---- Valid global net ----')
    val_logs = val_epoch.run(val_loader)

    # 保存验证数据
    result_path = os.path.join('result', Config.data_name, 'global')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(os.path.join(result_path, 'val_logs.json'), "w") as file:
        json.dump(val_logs, file)
    # 保存当前全局模型
    torch.save(global_net, os.path.join(result_path, 'global_net.pth'))
    print('Global net saved!')
    return val_logs


if __name__ == '__main__':
    local_w = defaultdict(dict)
    local_train_log = defaultdict(dict)
    local_val_log = defaultdict(dict)
    global_val_log = defaultdict(dict)

    global_net = smp.UnetPlusPlus(
        encoder_name=Config.encoder,
        encoder_weights=Config.encoder_weights,
        classes=len(Config.classes),
        activation=Config.activation,
    )

    # 总联邦学习训练轮次
    for e in range(Config.epoch):
        print('#**************** Epoch: {} ****************#'.format(e))
        global_net.train()
        # 每个client在本地训练一定轮次
        for i in range(1, Config.region_num + 1):
            print('----Client {} local train----'.format(i))
            local_w[e][i], local_train_log[e][i], local_val_log[e][i] = local_train(global_net, Config.get_data_dir(i), i)
        # 模型聚合
        avg_w = FedAvg(local_w[e])
        global_net.load_state_dict(avg_w)
        global_net.eval()
        global_val_log[e] = global_val(Config.get_data_dir(0), global_net)

        # 每一轮保存数据
        result_path = os.path.join('result', Config.data_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, 'local_train_logs.json'), "w") as file:
            json.dump(local_train_log, file)
        with open(os.path.join(result_path, 'local_val_logs.json'), "w") as file:
            json.dump(local_val_log, file)
        with open(os.path.join(result_path, 'global_val_logs.json'), "w") as file:
            json.dump(global_val_log, file)
