# -*- coding: UTF-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
from config import Config

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学公式字体为stix
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

file_name = "val_logs.json" if Config.data_dist == "un_fed" else "global_val_logs.json"

# 打开并读取 JSON 文件
with open(os.path.join(Config.get_result_dir(), file_name), 'r') as file:
    data = json.load(file)

# 提取指标和迭代次数
iterations = list(data.keys())
dice_score = [1 - data[i]["dice_loss"] for i in iterations]
iou_score = [data[i]["iou_score"] for i in iterations]

# 绘制曲线图
plt.figure(figsize=(10, 5))
plt.plot(iterations, dice_score, label='Dice指标', color='red', marker='o')
plt.plot(iterations, iou_score, label='IoU指标', color='blue', marker='o')

# 添加标题和标签
plt.title('Dice和IoU指标随训练轮数的变化')
plt.xlabel('训练轮数')
plt.ylabel('指标值')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(Config.get_result_dir(), "Dice与IoU指标变化.svg"), dpi=300)
