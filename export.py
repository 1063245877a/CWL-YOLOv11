import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO




if __name__ == '__main__':
    model = YOLO('/root/ultralytics-yolo11-main/runs/train/yolov11/weights/best.pt')
    model.export(format='onnx', simplify=True, opset=13)