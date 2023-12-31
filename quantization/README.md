###### 环境安装
1.我们提供了python3.6、python3.8、python3.10的量化环境whl文件，根据自己的运行环境进行安装

```bash
cd rknn-toolkit2
pip3 install xxx.whl
```

###### 模型量化
1. 选取适当的图片进行量化模型，精度校准，并把图片路径保存至xxx.txt

1. 模型量化

    ```python
    python convert.py onnx_model_path platform i8/fp output_rknn_path
    ```
    
    其中onnx_model_path：训练后导出的onnx模型文件位置
    
    platform：[rk3562,rk3566,rk3568,rk3588]
    
    i8/fp：i8代表使用图片量化；fp代表不量化
    
    output_rknn_path：量化后模型的保存路径
