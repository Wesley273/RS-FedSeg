# -*- coding: utf-8 -*-
import copy

import torch


def FedAvg(local_w):
    w_avg = copy.deepcopy(local_w[1])
    for k in w_avg.keys():
        for i in range(1, len(local_w)):
            w_avg[k] += local_w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(local_w))
    return w_avg
