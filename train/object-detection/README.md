## 环境安装
1. Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).推荐使用Conda虚拟环境

2. ```bash
   git clone  https://github.com/AIDrive-Research/EdgeAI-Pipeline.git
   cd EdgeAI-Pipeline
   pip install -r requirements.txt 
   ```
## 数据准备
1. 把标注好的数据放置在自定义目录，目录结构如下图:

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

2. 新建训练数据yaml文件,参考coco128.yaml编辑自己的yaml文件

    ```
    cd data
    touch custom.yaml
    ```

## 训练

1. 单卡训练

   ```python
   python train.py --data data/custom.yaml --epochs 300 --weights '' --cfg yolov5s.yaml  --batch-size 128
   ```

   如果使用预训练

   ```
   python train.py --data data/custom.yaml --epochs 300 --weights yolov5s.pt   --batch-size 128
   ```

2. 推荐使用多卡训练

   ```python
   python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data data/custom.yaml --weights yolov5s.pt --device 0,1
   ```
## 模型导出
1. ONNX_RKNN导出，这里我们支持RK有NPU能力的全系列，包括RK1808、RV1109、RV1126、RK3399PRO、RK3566、RK3568、RK3588、RK3588S、RV1106、RV1103

   ```python
   python export_rk.py --weights xxx.py --include onnx --simplify --opset 12 --rknpu rk3588
   ```
