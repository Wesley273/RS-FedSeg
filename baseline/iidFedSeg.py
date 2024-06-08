# -*- coding: UTF-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import ssl
from collections import defaultdict

import torch
from torch.cuda.amp import autocast as autocast
from torch.utils.data import random_split

from config import Config
from fed_algos import FedAvg
from train import get_full_data, global_val, local_train

if torch.cuda.is_available():
    DEVICE = torch.device(Config.device)
else:
    DEVICE = torch.device('cpu')

ssl._create_default_https_context = ssl._create_unverified_context


def get_iid_dataset():
    data_dirs = [Config.get_data_dir(j) for j in range(1, Config.region_num + 1)]
    full_train_data, full_val_data, _ = get_full_data(data_dirs)

    num_train = len(full_train_data)
    num_val = len(full_val_data)
    num_trains = [num_train // Config.region_num] * (Config.region_num - 1)
    num_trains.append(num_train // Config.region_num + num_train % Config.region_num)
    num_vals = [num_val // Config.region_num] * (Config.region_num - 1)
    num_vals.append(num_val // Config.region_num + num_val % Config.region_num)
    train_datasets = random_split(full_train_data, num_trains, generator=torch.Generator().manual_seed(42))
    val_datasets = random_split(full_val_data, num_vals, generator=torch.Generator().manual_seed(42))

    return full_val_data, train_datasets, val_datasets


if __name__ == '__main__':
    Config.data_dist = 'iid'

    global_net = Config.get_net()
    full_val_data, local_train_datas, local_val_datas = get_iid_dataset()

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
            local_w[e][i], local_train_log[e][i], local_val_log[e][i] = local_train(global_net, local_train_datas[i - 1], local_val_datas[i - 1], client=i)
        # 模型聚合
        avg_w = FedAvg(local_w[e])
        global_net.load_state_dict(avg_w)
        global_net.eval()
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
