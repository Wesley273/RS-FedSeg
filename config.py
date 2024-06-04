# -*- coding: UTF-8 -*-
class Config:
    # 网络训练参数
    device = 'cuda'

    # 数据保存位置
    data_name = 'BH_POOLS_SPLIT'
    client = 1
    data_dir = './data/{}/REGION_{}'.format(data_name, client)
