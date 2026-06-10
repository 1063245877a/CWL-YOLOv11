import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import YOLO


# nohup python train.py > trainweichengshuyolov11.log 2>&1 &
# tail -f trainweichengshuyolov11.log

if __name__ == '__main__':
    model = YOLO('/root/ultralytics-yolo11-main/ultralytics/cfg/models/11/yolo11.yaml')

    model.train(data='/root/PG-YOLO-Dataset-master/data.yaml',
               task='detect',
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=0,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD 优化器 默认为auto建议大家使用固定的.
                # resume=, # 续训的话这里填写True
                # amp=True,  # 如果出现训练损失为Nan可以关闭amp
                # device='0,1,2,3,4,5,6,7', # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
                # optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=False, # close amp | loss出现nan可以关闭amp
                fraction=1,
                project='runs/train',
                name='snu77-new',
                )
    model = YOLO('runs/train/snu77-new/weights/best.pt') # 选择训练好的权重路径
    model.val(data='/root/PG-YOLO-Dataset-master/data.yaml',
              split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
