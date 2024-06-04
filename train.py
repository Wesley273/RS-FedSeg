import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssl

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import utils as smp_utils
from torch.utils.data import DataLoader

from config import Config
from datasets import BHPOOLDataset, BHWATERTANKDataset, CamVidDataset
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
    MyDataset = BHPOOLDataset

    # 训练集
    x_train_dir = os.path.join(DATA_DIR, 'train')
    y_train_dir = os.path.join(DATA_DIR, 'trainannot')

    # 验证集
    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = MyDataset.CLASSES
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

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
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    loss = smp_utils.losses.DiceLoss()
    metrics = [smp_utils.metrics.IoU(threshold=0.5)]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

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

    for i in range(0, 40):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # 每次迭代保存下训练最好的模型
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, 'weights/best_model.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
