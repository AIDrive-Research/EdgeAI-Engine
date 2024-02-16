## 环境安装
1. Clone repo and install [requirements.txt](https://github.com/AIDrive-Research/EdgeAI-Engine/blob/main/train/pose-detection/requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).推荐使用Conda虚拟环境

2. 安装依赖
   ```bash
    git clone https://github.com/AIDrive-Research/EdgeAI-Engine.git
    cd EdgeAI-Engine/train/pose-detection
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

## 推理

```python
python detect.py --source ultralytics/assets --weights yolov8n-pose.pt
```

## 模型导出
1. ONNX_RKNN导出，这里我们支持RK有NPU能力的全系列，包括RK1808、RV1109、RV1126、RK3399PRO、RK3566、RK3568、RK3588、RK3588S、RV1106、RV1103

   ```bash
   yolo mode=export model=yolov8n-pose.pt format=onnx opset=12 simplify=True
   ```
