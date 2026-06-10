# Ultralytics YOLO11 Object Detection

This README describes how to install the project, use pretrained weights, run command-line training/validation/inference examples, and export models to ONNX.

---

## 1. Project Overview

This repository is based on **Ultralytics YOLO11** and supports object detection training, validation, inference, and model export.

It includes both the original YOLO11 configuration and an improved model configuration.

### Main Model Configurations

| Model | Config Path | Description |
|---|---|---|
| YOLO11 | `ultralytics/cfg/models/11/yolo11.yaml` | Original YOLO11 model configuration |
| Improved YOLO11 | `ultralytics/cfg/models/11/yolo11-CCFM-Detect_LSDECD-C3k2_WTConv.yaml` | Improved model with CCFM, Detect_LSDECD, C3k2, and WTConv modules |

### Common Scripts

| Script | Description |
|---|---|
| `train.py` | Basic YOLO11 training and validation example |
| `train-yolo11-C3k2_WTConv-Detect_LSDECD-CCFM.py` | Improved YOLO11 training and validation example |
| `detect-yolo.py` | Inference example |
| `export.py` | ONNX export example |
| `export_ter.py` | TensorRT export example |

---

## 2. Environment Requirements

Recommended environment:

- Python >= 3.8
- PyTorch >= 1.8
- torchvision >= 0.9
- CUDA GPU, recommended for training and TensorRT export
- Linux, Windows, and macOS are supported
- Linux with an NVIDIA GPU is recommended for model training

---

## 3. Installation

### 3.1 Enter the Project Directory

```bash
cd ultralytics-yolo11-main
```

### 3.2 Create a Virtual Environment

Using conda:

```bash
conda create -n yolo11 python=3.10 -y
conda activate yolo11
```

Using venv:

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate the environment on Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 3.3 Install PyTorch

Install PyTorch according to your CUDA version.

Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only environments:

```bash
pip install torch torchvision
```

### 3.4 Install This Project

Editable installation is recommended:

```bash
pip install -e .
```

To enable ONNX, OpenVINO, TensorFlow, and other export formats, install export dependencies:

```bash
pip install -e ".[export]"
```

You may also install common ONNX tools:

```bash
pip install onnx onnxruntime onnxslim
```

### 3.5 Verify Installation

```bash
yolo checks
```

Or:

```bash
python -c "from ultralytics import YOLO; print('YOLO import success')"
```

---

## 4. Pretrained Weights

The repository includes the following weight files:

| Model | Weight Path | ONNX Path | Description |
|---|---|---|---|
| YOLO11n | `runs/YOLOv11n/weight/best.pt` | `runs/YOLOv11n/weight/best.onnx` | YOLO11n detection model weights |
| Improved YOLO11 | `runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.pt` | `runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.onnx` | Improved model weights |
| YOLO11n official/local | `yolo11n.pt` | - | YOLO11n pretrained weights |
| YOLOv8n official/local | `yolov8n.pt` | - | YOLOv8n weights for comparison or testing |

After re-training, Ultralytics usually saves results to:

```text
runs/train/<name>/weights/
```

For example:

```text
runs/train/exp/weights/best.pt
runs/train/exp/weights/last.pt
```

The packaged weights in this repository are stored under:

```text
runs/<model_name>/weight/
```

---

## 5. Dataset Format

Prepare your dataset in the Ultralytics YOLO format and provide a `data.yaml` file.

Example:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: class_0
  1: class_1
```

YOLO label files should use the following txt format:

```text
class_id x_center y_center width height
```

All coordinates should be normalized to the range from 0 to 1.



---

## 6. Command-Line Examples

### 6.1 Train YOLO11n

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11.yaml \
  data=/path/to/data.yaml \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  device=0 \
  optimizer=SGD \
  project=runs/train \
  name=YOLOv11n
```

### 6.2 Train the Improved Model

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11-CCFM-Detect_LSDECD-C3k2_WTConv.yaml \
  data=/path/to/data.yaml \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  device=0 \
  optimizer=SGD \
  amp=False \
  close_mosaic=0 \
  project=runs/train \
  name=CCFM-Detect_LSDECD-C3k2_WTConv
```

To train from YOLO11n pretrained weights:

```bash
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11-CCFM-Detect_LSDECD-C3k2_WTConv.yaml \
  pretrained=yolo11n.pt \
  data=/path/to/data.yaml \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  device=0
```

### 6.3 Resume Training

```bash
yolo detect train \
  model=runs/train/CCFM-Detect_LSDECD-C3k2_WTConv/weights/last.pt \
  data=/path/to/data.yaml \
  resume=True
```

### 6.4 Validate or Test

```bash
yolo detect val \
  model=runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.pt \
  data=/path/to/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/val \
  name=CCFM-Detect_LSDECD-C3k2_WTConv
```

### 6.5 Predict Images or Videos

Using the improved model:

```bash
yolo detect predict \
  model=runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.pt \
  source=/path/to/images_or_video \
  imgsz=640 \
  conf=0.2 \
  save=True \
  save_crop=True \
  project=runs/detect \
  name=CCFM-Detect_LSDECD-C3k2_WTConv
```

Using YOLO11n weights:

```bash
yolo detect predict \
  model=runs/YOLOv11n/weight/best.pt \
  source=/path/to/images_or_video \
  imgsz=640 \
  conf=0.25 \
  save=True
```

### 6.6 Predict with an ONNX Model

```bash
yolo detect predict \
  model=runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.onnx \
  source=/path/to/images \
  imgsz=640 \
  conf=0.2 \
  save=True
```

---

## 7. Python API Examples

### 7.1 Training

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11-CCFM-Detect_LSDECD-C3k2_WTConv.yaml")

model.train(
    data="/path/to/data.yaml",
    task="detect",
    imgsz=640,
    epochs=100,
    batch=16,
    device="0",
    optimizer="SGD",
    amp=False,
    project="runs/train",
    name="CCFM-Detect_LSDECD-C3k2_WTConv",
)
```

### 7.2 Validation

```python
from ultralytics import YOLO

model = YOLO("runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.pt")

model.val(
    data="/path/to/data.yaml",
    split="test",
    imgsz=640,
    batch=16,
    device="0",
    project="runs/val",
    name="CCFM-Detect_LSDECD-C3k2_WTConv",
)
```

### 7.3 Inference

```python
from ultralytics import YOLO

model = YOLO("runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.pt")

model.predict(
    source="/path/to/images_or_video",
    imgsz=640,
    conf=0.2,
    save=True,
    save_crop=True,
    project="runs/detect",
    name="predict",
)
```

---

## 8. ONNX Export

### 8.1 Basic ONNX Export

Export by command line:

```bash
yolo export \
  model=runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.pt \
  format=onnx \
  imgsz=640 \
  opset=13 \
  simplify=True
```

Export by Python:

```python
from ultralytics import YOLO

model = YOLO("runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.pt")
model.export(format="onnx", imgsz=640, opset=13, simplify=True)
```

### 8.2 Export YOLO11n to ONNX

```bash
yolo export \
  model=runs/YOLOv11n/weight/best.pt \
  format=onnx \
  imgsz=640 \
  opset=13 \
  simplify=True
```

### 8.3 ONNX Export Options

| Option | Example | Description |
|---|---|---|
| `format` | `format=onnx` | Export format. Use `onnx` for ONNX export. |
| `model` | `model=best.pt` | Path to PyTorch `.pt` weights. |
| `imgsz` | `imgsz=640` or `imgsz=640,640` | Input image size. |
| `opset` | `opset=13` | ONNX opset version. If not specified, Ultralytics uses the default value. |
| `simplify` | `simplify=True` | Simplifies the ONNX graph. Recommended for deployment. |
| `dynamic` | `dynamic=True` | Enables dynamic input axes. Useful when input size or batch size may change. |
| `batch` | `batch=1` | Batch size used during export. |
| `half` | `half=True` | Exports an FP16 model when supported. Usually requires GPU. |
| `device` | `device=0` or `device=cpu` | Device used for export. |
| `nms` | `nms=True` | Adds NMS into the exported model if supported. |

### 8.4 Dynamic ONNX Export

If input resolution or batch size may vary during deployment, enable `dynamic=True`:

```bash
yolo export \
  model=runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.pt \
  format=onnx \
  imgsz=640 \
  opset=13 \
  simplify=True \
  dynamic=True
```

### 8.5 FP16 ONNX Export

If your deployment environment supports FP16, try:

```bash
yolo export \
  model=runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.pt \
  format=onnx \
  imgsz=640 \
  half=True \
  device=0 \
  simplify=True
```

### 8.6 Export ONNX with NMS

To include NMS in the exported model, try:

```bash
yolo export \
  model=runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.pt \
  format=onnx \
  imgsz=640 \
  opset=13 \
  simplify=True \
  nms=True
```

ONNX models with NMS may not be supported by all inference frameworks. If compatibility issues occur, export ONNX without NMS and run NMS in your post-processing code.

### 8.7 Verify ONNX with ONNX Runtime

```bash
yolo detect predict \
  model=runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.onnx \
  source=/path/to/images \
  imgsz=640 \
  conf=0.2 \
  save=True
```

---

## 9. TensorRT Export

For deployment on NVIDIA GPUs or Jetson devices, export a TensorRT engine:

```bash
yolo export \
  model=runs/CCFM-Detect_LSDECD-C3k2_WTConv/weight/best.pt \
  format=engine \
  imgsz=640 \
  half=True \
  device=0 \
  simplify=True
```

You may also refer to:

```bash
python export_ter.py
```

---

## 10. Useful Scripts

```bash
# Basic training and validation
python train.py

# Improved model training and validation
python train-yolo11-C3k2_WTConv-Detect_LSDECD-CCFM.py

# Inference
python detect-yolo.py

# ONNX export
python export.py

# TensorRT export
python export_ter.py
```

Some scripts may contain absolute paths such as `/root/...`. Please modify them according to your local dataset and project paths before running.

---

## 11. Troubleshooting

### 11.1 NaN Loss During Training

Try disabling AMP:

```bash
amp=False
```

### 11.2 Out of GPU Memory

Try reducing batch size or image size:

```bash
batch=8
imgsz=512
cache=False
```

You can also use a smaller model.

### 11.3 Dataset Not Found

Check whether the following fields in `data.yaml` are correct:

```yaml
path:
train:
val:
test:
```

### 11.4 ONNX Export Failure

Update ONNX-related dependencies:

```bash
pip install -U onnx onnxruntime onnxslim
```

You may also try a different opset version:

```bash
opset=12
```

Or:

```bash
opset=13
```

### 11.5 ONNX Runtime Inference Error

Check the following items:

- Whether the exported `.onnx` file exists
- Whether `onnxruntime` is installed
- Whether the input image size matches the exported model
- Whether NMS is included in the model or implemented in post-processing
- Whether the deployment framework supports the selected opset version

---

## 12. Recommended Workflow

A typical workflow is:

```bash
# 1. Install the project
pip install -e ".[export]"

# 2. Train the model
yolo detect train \
  model=ultralytics/cfg/models/11/yolo11-CCFM-Detect_LSDECD-C3k2_WTConv.yaml \
  data=/path/to/data.yaml \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  device=0

# 3. Validate the model
yolo detect val \
  model=runs/train/CCFM-Detect_LSDECD-C3k2_WTConv/weights/best.pt \
  data=/path/to/data.yaml \
  split=test

# 4. Export to ONNX
yolo export \
  model=runs/train/CCFM-Detect_LSDECD-C3k2_WTConv/weights/best.pt \
  format=onnx \
  imgsz=640 \
  opset=13 \
  simplify=True

# 5. Run ONNX inference
yolo detect predict \
  model=runs/train/CCFM-Detect_LSDECD-C3k2_WTConv/weights/best.onnx \
  source=/path/to/images \
  imgsz=640 \
  conf=0.2 \
  save=True
```

---

## 13. License

This project is based on Ultralytics YOLO. Please refer to `pyproject.toml`, `CITATION.cff`, and source file headers for license information.
