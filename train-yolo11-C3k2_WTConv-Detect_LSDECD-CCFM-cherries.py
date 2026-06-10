import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡

warnings.filterwarnings('ignore')
from ultralytics import YOLO


# nohup python train-yolo11-C3k2_WTConv-Detect_LSDECD-CCFM-cherries.py > train-yolo11-C3k2_WTConv-Detect_LSDECD-CCFM-cherries-6.log 2>&1 &
# tail -f train-yolo11-C3k2_WTConv-Detect_LSDECD-CCFM-cherries-6.log

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/ultralytics-yolo11-main/ultralytics/cfg/models/11/yolo11-CCFM-Detect_LSDECD-C3k2_WTConv.yaml')
     # 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,
    # # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
    # model.load('yolov11n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
    # model.load('/root/ultralytics-yolo11-main/yolo11n.pt') # loading pretrain weights
    model.train(data='/root/autodl-tmp/cherries/data.yaml',
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
                name='CCFM-Detect_LSDECD-C3k2_WTConv-cherries-new-6',
                )
    model = YOLO('/root/autodl-tmp/ultralytics-yolo11-main/runs/train/CCFM-Detect_LSDECD-C3k2_WTConv-cherries-new-6/weights/best.pt') # 选择训练好的权重路径
    model.val(data='/root/autodl-tmp/cherries/data.yaml',
              split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='CCFM-Detect_LSDECD-C3k2_WTConv-cherries-6',
              )
    file_path = "runs/train/CCFM-Detect_LSDECD-C3k2_WTConv-cherries-3/weights/best.pt"
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    
    print(f"文件大小: {size_bytes} 字节 ({size_mb:.2f} MB)")
    
