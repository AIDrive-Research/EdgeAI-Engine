# EdgeAI-Pipeline
🔥🔥🔥机器视觉边缘计算的成熟应用，适配RK瑞芯微/Ascend昇腾系列芯片，提供模型训练、模型量化源代码🔥🔥🔥
本工程历经大量的工业级应用：基于RK3568和RK3588的Arm-Linux芯片架构，先后在钢铁、金矿、石油、危化品、安防、电力能源等领域落地过大量应用，设备软硬件可靠性高、通用性强。

## 技术应用栈
利用本工程，作者团队基于边缘计算盒子设备解决了如下实际工业应用，兼备通用性和专业性，下表是作者团队已完成的商业化落地，各位开发者朋友也可利用本代码工程，完成相关应用：
### 应用栈
#### 通用业务-人员管理
| 应用名称 | 基础模型 | 后处理 |准确率 | 召回率 |
|----------|----------|----------|----------|----------|
| 安全帽佩戴检测 | 目标检测 | 检测head的反逻辑 | 99%+ | 92% |
| 反光衣穿戴检测 | 目标检测 | 检测普通穿戴的反逻辑 | 96%+ | 90% |
| 人脸识别 | ArcFace |  | 99%+ | 99%+ |
| 跌倒检测 | 目标检测 |  | 98% | 95% |
| 口罩佩戴识别 | 目标检测 |  | 99% | 95% | 
| 人员计数 | 目标检测 | bbox撞线 | 99%+ | 99%+ |
| 离岗检测 | 目标检测 | 检测反逻辑 | 99%+ | 99%+ |
| 人员聚集 | 目标检测 | 统计加和 | 99%+ | 99%+ |
| 区域入侵 | 目标检测 | 目标处于像素区域内 | 99%+ | 99%+ |
| 徘徊检测 | 目标跟踪&检测 |  | 95% | 92% |
| 睡岗检测 | 目标检测 | bbox静止超限 | 98% | 95% |
| 使用手机检测 | 目标检测 | 二次检测 | 91% | 99%+ |
| 抽烟检测 | 目标检测 | 二次检测 | 92% | 99%+ |
| 穿戴工服检测 | ArcFace |  | 95% | 90% |


#### 通用业务-险情防控
| 应用名称 | 基础模型 | 后处理 |准确率 | 召回率 |
|----------|----------|----------|----------|----------|
| 明火识别 | 目标检测 | 目标静态判断 | 95% | 92% |
| 明烟识别 | 目标检测 | 目标静态判断 | 93% | 90% |
| 静电夹检测 | 目标检测 |  | 96% | 98% |
| 电瓶车违停 | 目标检测 |  | 99%+ | 95% |
| 灭火器离位检测 | 目标检测 |  | 99%+ | 99%+ |



#### 通用业务-车辆管理
| 应用名称 | 基础模型 | 后处理 |准确率 | 召回率 |
|----------|----------|----------|----------|----------|
| 车辆计数 | 像素均值（传统方法） | |99%+ | 99%+ |
| 车辆违停 | 目标跟踪&检测 | |99%+ | 99%+ |
| 车型识别 | 目标检测 |  | 97% | 98% |

#### 专用业务-钢铁领域
#### 专用业务-油田领域
| 应用名称 | 基础模型 | 后处理 |准确率 | 召回率 |
|----------|----------|----------|----------|----------|
| 漏油液滞检测 | 目标检测 |  | 89%+ | 97%% |
| 滴液检测 | 目标检测 |  | coming soon | coming soon |

#### 专用业务-金矿领域
| 应用名称 | 基础模型 | 后处理 |准确率 | 召回率 |
|----------|----------|----------|----------|----------|
| 设备停机检测 | 帧差 |  |99%+ | 99%+ |


#### 专用业务-其它
| 应用名称 | 基础模型 | 后处理 |准确率 | 召回率 |
|----------|----------|----------|----------|----------|
| 黑屏检测 | 像素均值（传统方法） | |99%+ | 99%+ |
| 轨道异物检测 | 图像分类（resnet50） | |99%+ | 99%+ |
| 皮带偏离检测 | 实例分割 |  | 99%+ | 99%+ |
| 垃圾检测 | 目标检测 |  | 90% | 88% |
| 货架拿物品动作识别| 目标检测&姿态检测 |  | 96% | 95% |
| 移动侦测 | 帧差 |  | 99% | 99% |
| 大货车计数 | 目标检测 | bbox撞线 | 99% | 99% |



### 技术栈
本工程包含如下技术栈

## 文件说明
本工程包含两部分，分别是：模型 训练源代码 和 模型量化 源代码，前者基于pytorch平台，后者分别基于RK和Ascend（coming soon）平台。
### Train-模型训练源码

### Quantization-模型量化源码

## 授权
本项目完全开源，可商用。  
作者团队希望为开源世界助力，让更多人拥有AI的开发应用能力，而不仅仅是在x86服务端做实验室玩具制造。
## 作者寄语

## 加入我们
