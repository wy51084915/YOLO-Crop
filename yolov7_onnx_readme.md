# 模型名称
helmet-smoke-phone-dynamic-batch.onnx
# 模型功能
头盔手机香烟检测
# 模型量化级别
int8
# 前处理
（注：尽可能将前处理包含在onnx模型内部，如果已包含则不需写此部分）
1. 色彩维度顺序: R-G-B
2. 归一化到 [0, 1]
# 输入信息:
## 形状
### 形状大小
b x 3 x 640 x 640
### 维度含义
batch x RGB-channel x height x width 
## 数据类型
float32
## 是否支持动态batch
是

## 输入内容
标准化 RGB 图像

# 输出信息
## 形状
### 形状大小
b x 25200 x 7
### 维度含义
batch x 检测框数量 x 检测框信息 
#### 检测框数量解读
输出有三个大小的特征图（80*80,40*40,20*20），每个点有三个anchor，所以25200=3*（80*80,40*40,20*20），即锚框的数量
#### 检测框信息解读
1个batchid;

4个坐标值：x0,y0,x1,y1;  

1个类别id：cls_id  ==》0为头盔，1为未佩戴头盔，2为手机，3为香烟；

1个置信度：score 

### 数据类型
float32
## 阈值
### 目标类别得分阈值
0.6
# 后处理
包含后处理

# 模型性能 （如有）
## 数据集指标

## 测试脚本

```python
import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
 
cuda = False
w = "./weights/helmet-smoke-phone-dynamic-batch.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)
 
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
 
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
 
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
 
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2
 
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)
 
names = ['helmet', 'no_helmet', 'cell phone', 'smoke' ]
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
 
# url = 'https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/image1.jpg'
# file = requests.get(url)
 
img_path = './inference/images/test.png'
# img = cv2.imdecode(np.frombuffer(file.content, np.uint8), 1)
img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
image = img.copy()
image, ratio, dwdh = letterbox(image, auto=False)
image = image.transpose((2, 0, 1))
image = np.expand_dims(image, 0)
image = np.ascontiguousarray(image)
 
im = image.astype(np.float32)
im /= 255
 
 
outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]
 
inp = {inname[0]:im}
t1 = time.time()
outputs = session.run(outname, inp)[0]
print('inference time :%.4f'%(time.time()-t1))
print(outputs)
 
ori_images = [img.copy()]
 
for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
    image = ori_images[int(batch_id)]
    box = np.array([x0,y0,x1,y1])
    box -= np.array(dwdh*2)
    box /= ratio
    box = box.round().astype(np.int32).tolist()
    cls_id = int(cls_id)
    score = round(float(score),3)
    name = names[cls_id]
    color = colors[name]
    name += ' '+str(score)
    cv2.rectangle(image,box[:2],box[2:],color,2)
    cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
 
cv2.imshow('dddd',ori_images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
 
# Image.fromarray(ori_images[0])
```



## 实际场景效果

