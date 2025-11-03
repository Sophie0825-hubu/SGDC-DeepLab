import os
from PIL import Image
from tqdm import tqdm
from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 8
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes    = ["_ignore_", "background","building", "road", "water", "barren", "forest", " agriculture"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        # 确保创建所有必要的父目录
        os.makedirs(pred_dir, exist_ok=True)
            
        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            # 处理 image_id 中的路径分隔符
            image_id = image_id.replace('\\', '/')  # 统一使用正斜杠
            
            # 获取图片路径
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages", f"{image_id}.png")
            
            # 确保图片存在
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
                
            image = Image.open(image_path)
            image = deeplab.get_miou_png(image)
            
            # 获取保存路径
            save_path = os.path.join(pred_dir, f"{image_id}.png")
            
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存图片
            image.save(save_path)
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)