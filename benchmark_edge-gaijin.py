import time
import torch
import warnings
import numpy as np
from ultralytics import YOLO
import psutil
import os

warnings.filterwarnings('ignore')

def get_memory_usage():
    """获取当前进程占用的系统物理内存 (RAM)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_cpu_optimized():
    # 1. 环境模拟 (4线程模拟嵌入式 CPU)
    torch.set_num_threads(4) 
    print("环境模拟：已限制为 4 线程 CPU 运行")

    # 2. 模型加载
    print("开始加载模型...")
    start_load = time.time()
    model_path = '/root/ultralytics-yolo11-main/runs/train/CCFM-Detect_LSDECD-C3k2_WTConv3/weights/best.onnx'
    model = YOLO(model_path) 
    load_time = (time.time() - start_load) * 1000
    print(f"✅ 模型加载成功 | 加载时间: {load_time:.2f} ms")

    # 3. 硬件预热 (解决归一化警告)
    print("正在进行硬件预热 (10次)...")
    # 使用 torch.rand (0-1之间) 解决归一化警告
    dummy_input = torch.rand(1, 3, 640, 640)
    for _ in range(10):
        _ = model.predict(dummy_input, verbose=False, device='cpu')

    # 4. 正式性能测试
    test_images_path = '/root/PG-YOLO-Dataset-master/test/images'
    print(f"正在测试图片集: {test_images_path}")
    
    pre_times = []
    infer_times = []
    post_times = []
    peak_ram = 0
    count = 0
    
    # 核心修改：使用 stream=True 避免内存堆积，保证内存测试准确
    results = model.predict(source=test_images_path, imgsz=640, device='cpu', verbose=False, stream=True)
    
    for res in results:
        # 记录延迟
        pre_times.append(res.speed['preprocess'])
        infer_times.append(res.speed['inference'])
        post_times.append(res.speed['postprocess'])
        # 记录峰值内存
        peak_ram = max(peak_ram, get_memory_usage())
        count += 1
        if count % 10 == 0:
            print(f"已处理: {count} 张图片...")

    # 5. 指标统计
    avg_pre = np.mean(pre_times)
    avg_infer = np.mean(infer_times)
    avg_post = np.mean(post_times)
    total_latency = avg_pre + avg_infer + avg_post
    fps = 1000 / total_latency

    # 6. 输出最终报告
    print("\n" + "="*45)
    print("📋 强制 CPU 部署实验报告 (最终版)")
    print("-" * 45)
    print(f"1. 硬件平台: Intel Xeon Gold 6430 (模拟 4 核 CPU)")
    print(f"2. 测试样本数: {count} 张图片")
    print(f"3. 模型加载时间: {load_time:.2f} ms")
    print(f"4. 平均推理延迟 (Inference): {avg_infer:.2f} ms")
    print(f"5. 总端到端延迟 (Latency): {total_latency:.2f} ms")
    print(f"6. 平均帧率 (FPS): {fps:.2f}")
    print(f"7. 峰值内存占用 (Peak RAM): {peak_ram:.2f} MB")
    
    if peak_ram < 2048:
        print(f"✅ 结论: 运行内存稳健，完全适配 2GB RAM 边缘设备。")
    print("="*45)

if __name__ == '__main__':
    benchmark_cpu_optimized()