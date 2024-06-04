# -*- coding: UTF-8 -*-
import os
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#  图像宽不足裁剪宽度,填充至裁剪宽度


def fill_right(img, size_w):
    size = img.shape
    #  填充值为数据集均值
    img_fill_right = cv2.copyMakeBorder(img, 0, 0, 0, size_w - size[1], cv2.BORDER_CONSTANT, value=0)
    return img_fill_right

#  图像高不足裁剪高度,填充至裁剪高度


def fill_bottom(img, size_h):
    size = img.shape
    img_fill_bottom = cv2.copyMakeBorder(img, 0, size_h - size[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    return img_fill_bottom

#  图像宽高不足裁剪宽高度,填充至裁剪宽高度


def fill_right_bottom(img, size_w, size_h):
    size = img.shape
    img_fill_right_bottom = cv2.copyMakeBorder(img, 0, size_h - size[0], 0, size_w - size[1], cv2.BORDER_CONSTANT, value=0)
    return img_fill_right_bottom


def image_split(img_folder, out_img_folder, mask_folder, out_mask_folder, size_w=480, size_h=480, step=240):
    mask_list = os.listdir(mask_folder)
    for input_folder, out_folder, suffix in [(img_folder, out_img_folder, '.jpg'), (mask_folder, out_mask_folder, '.png')]:
        count = 0
        for img_name in mask_list:
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            number = 0
            #  去除.png后缀
            name = img_name[:-4]
            img = cv2.imread(os.path.join(input_folder, name + suffix))
            size = img.shape
            #  若图像宽高大于切割宽高
            if size[0] >= size_h and size[1] >= size_w:
                count = count + 1
                for h in range(0, size[0] - 1, step):
                    start_h = h
                    for w in range(0, size[1] - 1, step):
                        start_w = w
                        end_h = start_h + size_h
                        if end_h > size[0]:
                            start_h = size[0] - size_h
                            end_h = start_h + size_h
                        end_w = start_w + size_w
                        if end_w > size[1]:
                            start_w = size[1] - size_w
                        end_w = start_w + size_w
                        cropped = img[start_h: end_h, start_w: end_w]
                        #  用起始坐标来命名切割得到的图像，为的是方便后续标签数据抓取
                        name_img = name + '__1__' + str(start_h) + '___' + str(start_w)
                        cv2.imwrite('{}/{}.png'.format(out_folder, name_img), cropped)
                        number = number + 1
            #  若图像高大于切割高,但宽小于切割宽
            elif size[0] >= size_h and size[1] < size_w:
                print('图片{}需要在右面补齐'.format(name))
                count = count + 1
                img0 = fill_right(img, size_w)
                for h in range(0, size[0] - 1, step):
                    start_h = h
                    start_w = 0
                    end_h = start_h + size_h
                    if end_h > size[0]:
                        start_h = size[0] - size_h
                        end_h = start_h + size_h
                    end_w = start_w + size_w
                    cropped = img0[start_h: end_h, start_w: end_w]
                    name_img = name + '__1__' + str(start_h) + '___' + str(start_w)
                    cv2.imwrite('{}/{}.png'.format(out_folder, name_img), cropped)
                    number = number + 1
            #  若图像宽大于切割宽,但高小于切割高
            elif size[0] < size_h and size[1] >= size_w:
                count = count + 1
                print('图片{}需要在下面补齐'.format(name))
                img0 = fill_bottom(img, size_h)
                for w in range(0, size[1] - 1, step):
                    start_h = 0
                    start_w = w
                    end_w = start_w + size_w
                    if end_w > size[1]:
                        start_w = size[1] - size_w
                        end_w = start_w + size_w
                    end_h = start_h + size_h
                    cropped = img0[start_h: end_h, start_w: end_w]
                    name_img = name + '__1__' + str(start_h) + '___' + str(start_w)
                    cv2.imwrite('{}/{}.png'.format(out_folder, name_img), cropped)
                    number = number + 1
            #  若图像宽高小于切割宽高
            elif size[0] < size_h and size[1] < size_w:
                count = count + 1
                print('图片{}需要在下面和右面补齐'.format(name))
                img0 = fill_right_bottom(img, size_w, size_h)
                cropped = img0[0: size_h, 0: size_w]
                name_img = name + '__1__' + '0' + '___' + '0'
                cv2.imwrite('{}/{}.png'.format(out_folder, name_img), cropped)
                number = number + 1
            print('{}{}切割成{}张.'.format(name, suffix, number))
        print('共完成{}张图片'.format(count))


def delete_empty(img_folder, mask_folder):
    mask_list = os.listdir(mask_folder)
    for img_name in mask_list:
        mask = cv2.imread(os.path.join(mask_folder, img_name))
        if np.all(mask == 0):
            os.remove(os.path.join(mask_folder, img_name))
            os.remove(os.path.join(img_folder, img_name))
            print('{}为空，已被删除.'.format(img_name))


def get_train_val(split_folder, img_folder, mask_folder):
    train_folder = os.path.join(split_folder, 'train')
    trainannot_folder = os.path.join(split_folder, 'trainannot')
    val_folder = os.path.join(split_folder, 'val')
    valannot_folder = os.path.join(split_folder, 'valannot')
    for path in [train_folder, trainannot_folder, val_folder, valannot_folder]:
        if not os.path.exists(path):
            os.makedirs(path)

    mask_list = os.listdir(mask_folder)
    train_count = int(len(mask_list) * 0.9)
    for i, img_name in tqdm(enumerate(mask_list)):
        shutil.move(os.path.join(mask_folder, img_name), os.path.join(trainannot_folder, img_name))
        shutil.move(os.path.join(img_folder, img_name), os.path.join(train_folder, img_name))
        if i >= train_count:
            break
    mask_list = os.listdir(mask_folder)
    for i, img_name in tqdm(enumerate(mask_list)):
        shutil.move(os.path.join(mask_folder, img_name), os.path.join(valannot_folder, img_name))
        shutil.move(os.path.join(img_folder, img_name), os.path.join(val_folder, img_name))


if __name__ == "__main__":
    for i in [1, 2, 3, 4, 5, 6, 7, 8]:
        raw_folder = os.path.join('data', 'BH_WATERTANKS', 'REGION_{}'.format(i))
        split_folder = os.path.join('data', 'BH_WATERTANKS_SPLIT', 'REGION_{}'.format(i))
        #  图像数据集文件夹
        img_folder = os.path.join(raw_folder, 'IMAGES')
        #  切割得到的图像数据集存放文件夹
        out_img_folder = os.path.join(split_folder, 'IMAGES')
        #  mask数据集文件夹
        mask_folder = os.path.join(raw_folder, 'ANNOTATION')
        #  切割后数据集的标签文件存放文件夹
        out_mask_folder = os.path.join(split_folder, 'ANNOTATION')
        #  切割图像宽
        size_w = 480
        #  切割图像高
        size_h = 480
        #  切割步长,重叠度为(size_w - step)/size_w
        step = 240
        image_split(img_folder, out_img_folder, mask_folder, out_mask_folder, size_w, size_h, step)
        delete_empty(out_img_folder, out_mask_folder)
        get_train_val(split_folder, out_img_folder, out_mask_folder)
