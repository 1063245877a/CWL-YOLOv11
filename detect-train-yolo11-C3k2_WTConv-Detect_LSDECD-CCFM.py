import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/ultralytics-yolo11-main/runs/train/CCFM-Detect_LSDECD-C3k2_WTConv3/weights/best.pt') # select your model.pt path
    model.predict(source='/root/PG-YOLO-Dataset-master/test/images',
                  imgsz=640,
                  project='runs/detect',
                  name='CCFM-Detect_LSDECD-C3k2_WTConv',
                  save=True,
                  conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=1, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  save_crop=True, # save cropped images with results
                )