### 目录
- 模型量化

- 附录

### 模型量化
[tensorrt模型量化](https://blog.csdn.net/qq_39333636/article/details/135955581)
- 将yolo的txt转为onnx 

pt转onnx格式可以用yolov5里的export.py代码

比如：python export.py --weights yolov5s.pt --include onnx --device 0

- 将onnx转为trt

[根据教程使用tensorrt转换工具](https://blog.csdn.net/qq_39333636/article/details/135955581)

- 推理过程

将转换出来的best.trt改后缀为best.engine 运用detect.py文件进行重新推理训练，记得改root的pt改为engine的路径
---
### 附录
前几日操作使用工具

[markdown操作教程](https://markdown.com.cn/basic-syntax/)

[标注工具](https://www.makesense.ai/)
