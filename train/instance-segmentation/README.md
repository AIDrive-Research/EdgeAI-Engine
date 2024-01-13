## 环境安装
1. Clone repo and install [requirements.txt](requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).推荐使用Conda虚拟环境

2. ```bash
    git clone https://github.com/AIDrive-Research/EdgeAI-Engine.git
    cd EdgeAI-Engine/train/instance-segmentation
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

## 数据准备
1. 制作数据集，目录结构如下：

   ```bash
    JPEGImages:
    	xxx.jpg
    labels:
    	xxx.txt
   ```

2. 将数据拆分训练集和验证集

   ```bash
    python tools/1_split_train_val.py --input-images 图片文件路径 --input-labels YOLO格式标注文件存储路径 --output YOLO数据集存储路径
   ```

   目录结构如下图:

   ```bash
    images:
    	train
    		xxx.jpg
    	val
    		xxx.jpg
    labels:
    	train
    		xxx.txt
    	val
    		xxx.txt	
   ```

3. 新建训练数据yaml文件，参考data/coco128.yaml编辑自己的yaml文件

   ```bash
    cd data
    cp coco128.yaml custom.yaml
   ```

   修改custom.yaml:
   - path: 2中YOLO数据集存储路径
   - train: images/train
   - val: image/val
   - names: 类别名称

## 训练

1. 单卡训练

   ```python
    python segment/train.py --model yolov5s-seg.pt --data data/custom.yaml --epochs 300 --img 640 --batch-size 128
   ```

2. 推荐使用多卡训练

   ```python
    python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --model yolov5s-seg.pt --data data/custom.yaml --epochs 300 --img 640 --device 0,1,2,3
   ```

## 模型导出
1. ONNX_RKNN导出，这里我们支持RK有NPU能力的全系列，包括RK1808、RV1109、RV1126、RK3399PRO、RK3566、RK3568、RK3588、RK3588S、RV1106、RV1103

   ```python
    python export.py --weights xxx.py --include onnx --simplify --opset 12
   ```
