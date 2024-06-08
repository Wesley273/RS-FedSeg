# -*- coding: UTF-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import ssl
from collections import defaultdict

import torch
from segmentation_models_pytorch import utils as smp_utils
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader, random_split

from config import Config
from datasets import DataAug, Full, Region
from fed_algos import FedAvg

if torch.cuda.is_available():
    DEVICE = torch.device(Config.device)
else:
    DEVICE = torch.device('cpu')

ssl._create_default_https_context = ssl._create_unverified_context


def split_dataset(dataset):
    train_num = int(0.8 * len(dataset))
    val_num = int(0.1 * len(dataset))
    test_num = len(dataset) - train_num - val_num
    train_data, val_data, test_data = random_split(dataset,
                                                            lengths=[train_num, val_num, test_num],
                                                            generator=torch.Generator().manual_seed(42))
    train_data.augmentation = DataAug.augment_train()
    val_data.augmentation = DataAug.augment_val()
    test_data.augmentation = DataAug.augment_val()
    return train_data, val_data, test_data


def get_region_data(region_data_dir) -> Region:
    img_dir = os.path.join(region_data_dir, 'img')
    mask_dir = os.path.join(region_data_dir, 'mask')
    region_dataset = Region(
        img_dir,
        mask_dir,
        preprocessing=DataAug.preprocessing(Config.preprocessing_fn)
    )
    return split_dataset(region_dataset)


def get_full_data(region_data_dirs) -> Full:
    img_dirs = [os.path.join(data_dir, 'img') for data_dir in region_data_dirs]
    mask_dirs = [os.path.join(data_dir, 'mask') for data_dir in region_data_dirs]
    full_dataset = Full(
        img_dirs,
        mask_dirs,
        preprocessing=DataAug.preprocessing(Config.preprocessing_fn)
    )
    return split_dataset(full_dataset)


def local_train(local_net, train_dataset, val_dataset, client):
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
        # with autocast():
        train_logs[i] = train_epoch.run(train_loader)
        val_logs[i] = val_epoch.run(val_loader)

        # 保存训练记录
        result_dir = os.path.join(Config.get_result_dir(), 'client_{}'.format(client))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        with open(os.path.join(result_dir, 'train_logs.json'), "w") as file:
            json.dump(train_logs, file)
        with open(os.path.join(result_dir, 'val_logs.json'), "w") as file:
            json.dump(val_logs, file)
        # 保存当前轮次模型
        torch.save(local_net.state_dict(), os.path.join(result_dir, 'latest_net.pth'))
        print('Latest local net saved!')
        # 保存最好的模型
        if max_score < val_logs[i]['iou_score']:
            max_score = val_logs[i]['iou_score']
            best_local_net = local_net
            torch.save(best_local_net.state_dict(), os.path.join(result_dir, 'best_net.pth'))
            print('Best local net saved!')

    return best_local_net.state_dict(), train_logs, val_logs


def global_val(full_val_dataset, global_net):
    val_loader = DataLoader(full_val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    # 创建一个epoch，用于迭代数据样本
    val_epoch = smp_utils.train.ValidEpoch(global_net, loss=Config.loss, metrics=Config.metrics, device=DEVICE, verbose=True)

    # 在全量数据上验证全局模型的性能
    val_logs = {}
    print('---- Valid global net ----')
    val_logs = val_epoch.run(val_loader)

    # 保存验证数据
    result_dir = os.path.join(Config.get_result_dir(), 'global')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, 'val_logs.json'), "w") as file:
        json.dump(val_logs, file)
    # 保存当前全局模型
    torch.save(global_net.state_dict(), os.path.join(result_dir, 'global_net.pth'))
    print('Global net saved!')
    return val_logs


if __name__ == '__main__':
    local_w = defaultdict(dict)
    local_train_log = defaultdict(dict)
    local_val_log = defaultdict(dict)
    global_val_log = defaultdict(dict)

    global_net = Config.get_net()

    # 总联邦学习训练轮次
    for e in range(Config.epoch):
        print('#**************** Epoch: {} ****************#'.format(e))
        global_net.train()
        # 每个client在本地训练一定轮次
        for i in range(1, Config.region_num + 1):
            print('----Client {} local train----'.format(i))
            local_train_data, local_val_data, _ = get_region_data(Config.get_data_dir(i))
            local_w[e][i], local_train_log[e][i], local_val_log[e][i] = local_train(global_net, local_train_data, local_val_data, client=i)
        # 模型聚合
        avg_w = FedAvg(local_w[e])
        global_net.load_state_dict(avg_w)
        global_net.eval()
        _, full_val_data, _ = get_full_data([Config.get_data_dir(j) for j in range(1, Config.region_num + 1)])
        global_val_log[e] = global_val(full_val_data, global_net)

        # 每一轮保存数据
        result_dir = Config.get_result_dir()
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        with open(os.path.join(result_dir, 'local_train_logs.json'), "w") as file:
            json.dump(local_train_log, file)
        with open(os.path.join(result_dir, 'local_val_logs.json'), "w") as file:
            json.dump(local_val_log, file)
        with open(os.path.join(result_dir, 'global_val_logs.json'), "w") as file:
            json.dump(global_val_log, file)
