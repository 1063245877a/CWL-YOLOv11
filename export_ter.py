from ultralytics import YOLO

# 1. 加载你的训练权重
model = YOLO('/root/ultralytics-yolo11-main/runs/train/CCFM-Detect_LSDECD-C3k2_WTConv3/weights/best.pt')

# 2. 导出为 TensorRT 格式
# half: 开启 FP16 半精度加速，边缘设备（Jetson）性能提升巨大，且精度损失极小
# device: 指定 GPU 索引
# simplify: 简化 ONNX 模型结构
path = model.export(format='engine', imgsz=640, half=True, device=0, simplify=True)

print(f"✅ 导出成功！TensorRT 模型路径在: {path}")