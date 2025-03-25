# -*- coding: UTF-8 -*-
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
from config import Config

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学公式字体为stix
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def load_data(data_name, data_dist):
    """加载 JSON 文件并提取数据"""
    file_name = "val_logs.json" if data_dist == "un_fed" else "global_val_logs.json"
    with open(os.path.join(Config.get_result_dir(data_dist=data_dist, data_name=data_name), file_name), 'r') as file:
        data = json.load(file)
    iterations = list(data.keys())
    dice_score = [1 - data[i]["dice_loss"] for i in iterations]
    iou_score = [data[i]["iou_score"] for i in iterations]
    return iterations, dice_score, iou_score


def plot_metrics(iterations, dice_score, iou_score):
    """绘制 Dice 和 IoU 指标曲线图并保存为图片"""
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
    plt.close()  # 关闭图像，避免内存泄漏


def plot_together(data_name, data_dist_list, iterations_list, dice_score_list, iou_score_list):
    """将多个数据分布的指标绘制在同一张图中"""
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green']
    for i, data_dist in enumerate(data_dist_list):
        plt.plot(iterations_list[i], dice_score_list[i], label=f'{data_dist} - Dice指标', color=colors[i], linestyle='-', marker='o')
        plt.plot(iterations_list[i], iou_score_list[i], label=f'{data_dist} - IoU指标', color=colors[i], linestyle='--', marker='x')
    # 添加标题和标签
    # plt.title(f'{data_name} 在不同数据分布下的 Dice 和 IoU 指标')
    plt.xlabel('训练轮数')
    plt.ylabel('指标值')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('result', f"{data_name}", f"{data_name}_所有数据分布_Dice与IoU指标变化.svg"), dpi=300)
    plt.close()  #


def plot_separate_metrics(data_name, data_dist_list, iterations_list, dice_score_list, iou_score_list):
    """将 Dice 和 IoU 指标分开绘制"""
    markers = ['o', 's', '^']  # 标记区分数据分布
    dist_dict = {'iid': 'IID', 'non_iid': 'Non-IID', 'un_fed': 'Non-Fed', }
    # 绘制 Dice 指标
    plt.figure(figsize=(8, 6))
    for i, data_dist in enumerate(data_dist_list):
        plt.plot(iterations_list[i], dice_score_list[i], label=f'{dist_dict[data_dist]}', marker=markers[i])
    # plt.title(f'{data_name} 在不同数据分布下的 Dice 指标')
    plt.xlabel('训练轮数')
    plt.ylabel('Dice 指标值')
    if data_name == 'IAIL':
        plt.ylim(0.6, 1)
    else:
        plt.ylim(0.3, 1)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join('result', f"{data_name}", f"{data_name}_所有数据分布_Dice指标变化.svg"), dpi=300)
    plt.close()  # 关闭图像，避免内存泄漏
    # 绘制 IoU 指标
    plt.figure(figsize=(8, 6))
    for i, data_dist in enumerate(data_dist_list):
        plt.plot(iterations_list[i], iou_score_list[i], label=f'{dist_dict[data_dist]}', marker=markers[i],)
    # plt.title(f'{data_name} 在不同数据分布下的 IoU 指标')
    plt.xlabel('训练轮数')
    plt.ylabel('IoU 指标值')
    if data_name == 'IAIL':
        plt.ylim(0.6, 1)
    else:
        plt.ylim(0.3, 1)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join('result', f"{data_name}", f"{data_name}_所有数据分布_IoU指标变化.svg"), dpi=300)
    plt.close()  # 关闭图像，避免内存泄漏


def save_metrics_to_csv(data_name, data_dist_list, iterations_list, dice_score_list, iou_score_list):
    """将不同数据分布下的 Dice 和 IoU 指标保存为 CSV 文件"""
    result_dir = os.path.join('result', data_name)
    os.makedirs(result_dir, exist_ok=True)  # 确保目录存在

    for i, data_dist in enumerate(data_dist_list):
        file_path = os.path.join(result_dir, f"{data_name}_{data_dist}_metrics.csv")
        df = pd.DataFrame({
            'Iterations': iterations_list[i],
            'Dice Score': dice_score_list[i],
            'IoU Score': iou_score_list[i]
        })
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"指标已保存至 {file_path}")


if __name__ == "__main__":

    data_names = ['BH_POOL', 'IAIL']
    data_dists = ['non_iid', 'iid', 'un_fed']

    """主函数"""
    for data_name in data_names:
        iterations_list = []
        dice_score_list = []
        iou_score_list = []
        for data_dist in data_dists:
            iterations, dice_score, iou_score = load_data(data_name, data_dist)
            iterations_list.append(iterations)
            dice_score_list.append(dice_score)
            iou_score_list.append(iou_score)

            # # 绘制曲线图并保存为图片
            # plot_metrics(iterations, dice_score, iou_score)

        # 绘制所有数据分布的曲线图并保存为图片
        # plot_together(data_name, data_dists, iterations_list, dice_score_list, iou_score_list)
        # 绘制分开的 Dice 和 IoU 指标图
        plot_separate_metrics(data_name, data_dists, iterations_list, dice_score_list, iou_score_list)
        # save_metrics_to_csv(data_name, data_dists, iterations_list, dice_score_list, iou_score_list)
