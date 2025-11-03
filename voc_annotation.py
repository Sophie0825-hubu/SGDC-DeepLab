import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path      = '/root/autodl-tmp/Deeplabv3/VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    # 定义数据集类型
    datasets = ['train', 'val', 'test']

    # 创建或清空txt文件
    ftrain      = open(os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation/train.txt'), 'w')
    fval        = open(os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation/val.txt'), 'w')
    ftest       = open(os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation/test.txt'), 'w')

    all_seg_files = []

    # 检查并处理 'trian' 目录
    train_seg_path = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass/train')
    if os.path.exists(train_seg_path):
        for seg in os.listdir(train_seg_path):
            if seg.endswith(".png"):
                all_seg_files.append(os.path.join('VOC2007/SegmentationClass/train', seg))
    else:
        print(f"Warning: {train_seg_path} does not exist. Skipping train dataset.")

    # 检查并处理 'val' 目录
    val_seg_path = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass/val')
    if os.path.exists(val_seg_path):
        for seg in os.listdir(val_seg_path):
            if seg.endswith(".png"):
                all_seg_files.append(os.path.join('VOC2007/SegmentationClass/val', seg))
    else:
        print(f"Warning: {val_seg_path} does not exist. Skipping val dataset.")

    # 随机划分数据集
    random.shuffle(all_seg_files)
    num_total = len(all_seg_files)
    num_train = int(num_total * 0.8)
    num_val = int(num_total * 0.1)
    num_test = num_total - num_train - num_val

    train_files = all_seg_files[:num_train]
    val_files = all_seg_files[num_train:num_train + num_val]
    test_files = all_seg_files[num_train + num_val:]

    for name in train_files:
        ftrain.write(name.replace('VOC2007/SegmentationClass' + os.sep, '')[:-4] + '\n')
    for name in val_files:
        fval.write(name.replace('VOC2007/SegmentationClass' + os.sep, '')[:-4] + '\n')
    for name in test_files:
        ftest.write(name.replace('VOC2007/SegmentationClass' + os.sep, '')[:-4] + '\n')
    
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    classes_nums        = np.zeros([256], int)
    for seg_file_path in tqdm(all_seg_files):
        png_file_name   = os.path.join(VOCdevkit_path, seg_file_path)
        if not os.path.exists(png_file_name):
                raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。"%(png_file_name))
    
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。"%(name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。"%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")
    print("如果格式有误，参考:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")