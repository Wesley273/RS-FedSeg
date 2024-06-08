# -*- coding: UTF-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import ssl

import torch
from collections import defaultdict
from segmentation_models_pytorch import utils as smp_utils
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader

from config import Config
from datasets import Full, DataAug
from torch.utils.data import random_split
from fed_algos import FedAvg
from train import local_train, global_val

if torch.cuda.is_available():
    DEVICE = torch.device(Config.device)
else:
    DEVICE = torch.device('cpu')

ssl._create_default_https_context = ssl._create_unverified_context


def get_iid_dataset():
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
    # 验证数据集
    val_dataset = Full(
        val_dirs,
        val_annot_dirs,
        augmentation=DataAug.augment_val(),
        preprocessing=DataAug.preprocessing(Config.preprocessing_fn)
    )
    torch.manual_seed(42)  # 设置随机种子，保证结果可重复
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    num_trains = [num_train // Config.region_num] * (Config.region_num - 1)
    num_trains.append(num_train // Config.region_num + num_train % Config.region_num)
    num_vals = [num_val // Config.region_num] * (Config.region_num - 1)
    num_vals.append(num_val // Config.region_num + num_val % Config.region_num)
    train_datasets = random_split(train_dataset, num_trains)
    val_datasets = random_split(val_dataset, num_vals)

    return val_dataset, train_datasets, val_datasets


if __name__ == '__main__':
    global_net = Config.get_net()
    val_dataset, train_datasets, val_datasets = get_iid_dataset()

    local_w = defaultdict(dict)
    local_train_log = defaultdict(dict)
    local_val_log = defaultdict(dict)
    global_val_log = defaultdict(dict)
    # 总联邦学习训练轮次
    for e in range(Config.epoch):
        print('#**************** Epoch: {} ****************#'.format(e))
        global_net.train()
        # 每个client在本地训练一定轮次
        for i in range(1, Config.region_num + 1):
            print('----Client {} local train----'.format(i))
            local_w[e][i], local_train_log[e][i], local_val_log[e][i] = local_train(global_net, train_datasets[i - 1], val_datasets[i - 1], client=i)
        # 模型聚合
        avg_w = FedAvg(local_w[e])
        global_net.load_state_dict(avg_w)
        global_net.eval()
        global_val_log[e] = global_val(val_dataset, global_net)

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
