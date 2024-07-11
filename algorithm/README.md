## 自定义算法
参考 postprocessor 文件夹中的内容实现自定义算法后处理，重点介绍两部分内容：utils和`__init__.py`。

### postprocessor/utils
- cv_utils
    - `color_utils.py`：提供了一些用于生成随机 RGB 颜色、反转图像颜色通道（RGB/BGR 和 RGBA/BGRA）以及将图像转换为灰度图的实用工具函数。
    - `crop_utils.py`：提供了一些用于裁剪矩形和多边形区域的实用工具函数。
    - `geo_utils.py`：提供了一系列用于几何计算的实用函数，包括点、线段、多边形和矩形的相交判断、点在区域内判断、以及矩形的交并比计算。
- `image_utils.py`：提供了在 bytes、base64 字符串、OpenCV 图像数组（ndarray）、以及 PIL 图像（Image）之间进行转换的实用工具函数。
- `json_utils.py`：提供了一些用于处理 JSON 数据的实用工具函数，包括加载、解析、序列化和保存 JSON 数据。
- `time_utils.py`：提供了一些用于时间和日期处理的实用工具函数，包括时间格式转换、计算耗时、获取星期几和一天中的秒数等功能。
- `unique_id_utils.py`：提供了一些用于生成唯一ID的实用工具函数。


### `postprocessor/__init__.py`
文件定义了一个名为 Postprocessor 的后处理基类，用于对检测结果进行过滤、处理和可视化。该类主要包括初始化方法、多个辅助方法以及一个主后处理方法 postprocess。

- 类属性  

    rq_source: 初始化 Redis 队列，用于获取数据。

- 初始化方法  
    `__init__(self, source_id, alg_name)`: 初始化方法，设置源 ID 和算法名称，并初始化各种属性，如颜色、标记、参数和结果等。

- 静态方法  

    `_is_in_plan(timestamp, plan=None)`: 判断给定时间戳是否在计划时间内。  
    `_get_label(label_conf, label)`: 根据标签配置获取标签的字符串表示。  
    `_get_point(rectangle, strategy)`: 根据策略获取矩形框的某个点。  
    `_merge_cross_line_counting_result(result1, result2)`: 合并跨线计数结果。  
    `_set_ext(obj, *args, **kwargs)`: 设置扩展字段。  
    `_get_ext(obj, key, pop=False)`: 获取扩展字段的值。  

- 私有方法  
    `_gen_random_color(self, seed=1)`: 生成随机颜色。  
    `_get_color(self, label_conf, label)`: 获取标签对应的颜色。  
    `_is_rectangle_in_polygon(self, rectangle, polygon, strategy)`: 判断矩形框是否在多边形内。  
    `_cross_line_counting(self, rectangle1, rectangle2, line, direction, strategy='center')`: 跨线计数。  
    `_gen_rectangle(self, xyxy, color, label, conf_, **kwargs)`: 生成矩形框。  
    `_gen_polygons(self, data=None)`: 生成多边形。  
    `_gen_lines(self, data=None)`: 生成线段。  
    `_scale(self, xyxy)`: 坐标缩放。  
    `_filter_by_conf(model_conf, conf_)`: 过滤掉置信度低于阈值的目标。  
    `_filter_by_label(self, model_conf, label)`: 过滤掉不在标签列表中的目标。  
    `_filter_by_roi(self, xyxy, strategy='center')`: 过滤掉中心点不在多边形内的目标。  
    `_process(self, result, filter_result)`: 处理过滤后的结果，生成最终结果。  
    `_filter(self, model_name, model_data)`: 过滤不符合条件的目标。  

- 公有方法  
`postprocess(self, args, draw_image)`: 后处理方法。根据提供的参数和图像数据进行后处理，过滤、处理目标并生成最终结果。