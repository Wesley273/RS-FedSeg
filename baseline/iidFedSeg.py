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
from datasets import NonIIDFull
from my_utils.data_augmentation import (augment_train, augment_val,
                                        preprocessing)

if torch.cuda.is_available():
    DEVICE = torch.device(Config.device)
else:
    DEVICE = torch.device('cpu')

ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == '__main__':
    pass
