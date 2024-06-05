# -*- coding: UTF-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import ssl

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import utils as smp_utils
from torch.utils.data import DataLoader

from config import Config
from my_utils.data_augmentation import (augment_train, augment_validation,
                                        preprocessing)

if torch.cuda.is_available():
    DEVICE = torch.device(Config.device)
else:
    DEVICE = torch.device('cpu')

ssl._create_default_https_context = ssl._create_unverified_context

# $# 创建模型并训练
# ---------------------------------------------------------------
if __name__ == '__main__':

    # 数据集所在的目录
    DATA_DIR = Config.data_dir
    MyDataset = Config.dataset

    # 训练集
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    # 验证集
    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    ENCODER = Config.encoder
    ENCODER_WEIGHTS = Config.encoder_weights
    CLASSES = Config.classes
    ACTIVATION = Config.activation  # could be None for logits or 'softmax2d' for multiclass segmentation

    # 用预训练编码器建立分割模型

    # 使用unet++模型
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # 加载训练数据集
    train_dataset = MyDataset(
        x_train_dir,
        y_train_dir,
        augmentation=augment_train(),
        preprocessing=preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # 加载验证数据集
    valid_dataset = MyDataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=augment_validation(),
        preprocessing=preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # 需根据显卡的性能进行设置，batch_size为每次迭代中一次训练的图片数，num_workers为训练时的工作进程数，如果显卡不太行或者显存空间不够，将batch_size调低并将num_workers调为0
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

    loss = smp_utils.losses.DiceLoss()
    metrics = [smp_utils.metrics.IoU(threshold=0.5)]

    optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.lr)

    # 创建一个简单的循环，用于迭代数据样本
    train_epoch = smp_utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # 进行40轮次迭代的模型训练
    max_score = 0
    train_logs = {}
    val_logs = {}
    result_path = os.path.join('result', Config.data_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for i in range(0, 40):
        print('\nEpoch: {}'.format(i))
        train_logs[i] = train_epoch.run(train_loader)
        val_logs[i] = valid_epoch.run(valid_loader)

        with open(os.path.join(result_path, 'train_logs.json'), "w") as file:
            json.dump(train_logs, file)
        with open(os.path.join(result_path, 'val_logs.json'), "w") as file:
            json.dump(val_logs, file)

        # 保存当前轮次模型
        torch.save(model, os.path.join(result_path, 'latest_model.pth'))
        print('Latest Model saved!')
        # 保存最好的模型
        if max_score < val_logs[i]['iou_score']:
            max_score = val_logs[i]['iou_score']
            torch.save(model, os.path.join(result_path, 'best_model.pth'))
            print('Best Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = Config.lr_low
            print('Decrease decoder learning rate to 1e-5!')
