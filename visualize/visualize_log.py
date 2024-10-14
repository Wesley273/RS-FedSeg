# -*- coding: UTF-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
from config import Config

# 打开并读取 JSON 文件
with open(os.path.join(Config.get_result_dir(), "global_val_logs.json"), 'r') as file:
    data = json.load(file)

# 提取指标和迭代次数
iterations = list(data.keys())
dice_score = [1-data[i]["dice_loss"] for i in iterations]
iou_score = [data[i]["iou_score"] for i in iterations]

# 绘制曲线图
plt.figure(figsize=(10, 5))
plt.plot(iterations, dice_score, label='Dice Score', color='red', marker='o')
plt.plot(iterations, iou_score, label='IoU Score', color='blue', marker='o')

# 添加标题和标签
plt.title('Dice and IoU Score Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
