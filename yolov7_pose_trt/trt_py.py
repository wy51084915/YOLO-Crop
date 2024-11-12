#coding=utf-8
"""
导出onnx后。
1 生成engine
    trtexec --onnx=./yolov7.onnx --saveEngine=./yolov7_fp16.engine --fp16 --workspace=200
    D:\TensorRT-8.4.1.5.Windows10.x86_64.cuda-10.2.cudnn8.4\TensorRT-8.4.1.5\lib\trtexec.exe --onnx=./yolov7.onnx --saveEngine=./yolov7_fp32.engine --workspace=1000
    D:\TensorRT-8.4.1.5.Windows10.x86_64.cuda-10.2.cudnn8.4\TensorRT-8.4.1.5\lib\trtexec.exe --onnx=./yolov7.onnx --saveEngine=./yolov7_fp16.engine --fp16 --workspace=1000
2 使用该脚本infer
"""
import cv2
import numpy as np
from collections import OrderedDict,namedtuple
import time

import os
import tensorrt as trt
import torch


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y # yes, can be calc on gpu



def non_max_suppression_kpt(prediction, iou_thres=0.45):
    prediction = np.array(prediction)

    dets = xywh2xyxy(prediction[:, :4])
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = prediction[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= iou_thres)[0]
        index = index[idx + 1]  # because index start from 1
    return prediction[keep]


class non_max_suppression_kpt_gpu():
    def __init__(self):
        super().__init__()
        self.zero = torch.zeros(1).cuda()
    
    def nms(self, prediction, iou_thres=0.45):
        # prediction = np.array(prediction)

        dets = xywh2xyxy(prediction[:, :4])
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = prediction[:, 4]
        keep = []
        index = scores.argsort(descending=True)
        # print(index.shape)
        while index.shape[0] > 0:
            i = index[0]  # every time the first is the biggst, and add it directly
            keep.append(i)

            x11 = torch.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
            y11 = torch.maximum(y1[i], y1[index[1:]])
            x22 = torch.minimum(x2[i], x2[index[1:]])
            y22 = torch.minimum(y2[i], y2[index[1:]])

            w = torch.maximum(self.zero, x22 - x11 + 1)  # the weights of overlap
            h = torch.maximum(self.zero, y22 - y11 + 1)  # the height of overlap

            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

            idx = torch.where(ious <= iou_thres)[0]
            index = index[idx + 1]  # because index start from 1
        return prediction[keep,:]


class TRT_engine():
    def __init__(self, weight) -> None:
        self.imgsz = [640,640]
        self.weight = weight
        self.device = torch.device('cuda:0')
        self.init_engine()

        self.nms = non_max_suppression_kpt_gpu()

    def init_engine(self):
        # Infer TensorRT Engine
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.weight, 'rb') as self.f, trt.Runtime(self.logger) as self.runtime:
            self.model = self.runtime.deserialize_cuda_engine(self.f.read())
        self.bindings = OrderedDict()
        self.fp16 = False
        # print(f"num binding = {self.model.num_bindings}")
        for index in range(self.model.num_bindings):
            self.name = self.model.get_binding_name(index)
            # print(f"name = {self.name}")
            self.dtype = trt.nptype(self.model.get_binding_dtype(index))
            self.shape = tuple(self.model.get_binding_shape(index))
            self.data = torch.from_numpy(np.empty(self.shape, dtype=np.dtype(self.dtype))).to(self.device)
            self.bindings[self.name] = self.Binding(self.name, self.dtype, self.shape, self.data, int(self.data.data_ptr()))
            if self.model.binding_is_input(index) and self.dtype == np.float16:
                self.fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()

    def letterbox(self,im,color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        self.r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            self.r = min(self.r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * self.r)), int(round(shape[0] * self.r))
        self.dw, self.dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            self.dw, self.dh = np.mod(self.dw, stride), np.mod(self.dh, stride)  # wh padding
        self.dw /= 2  # divide padding into 2 sides
        self.dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        self.img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return self.img,self.r,self.dw,self.dh

    def preprocess(self,image):
        self.img,self.r,self.dw,self.dh = self.letterbox(image)
        self.img = self.img[...,::-1]  # BGR转RGB
        self.img = self.img.transpose((2, 0, 1))
        self.img = np.expand_dims(self.img,0)
        self.img = np.ascontiguousarray(self.img)
        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.float()
        self.img /= 255.0
        return self.img

    def predict(self,img,threshold):
        t00 = time.time()
        img = self.preprocess(img)
        t01 = time.time()
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        # nums = self.bindings['num_detections'].data[0].tolist()
        # boxes = self.bindings['detection_boxes'].data[0].tolist()
        # scores =self.bindings['detection_scores'].data[0].tolist()
        # classes = self.bindings['detection_labels'].data[0].tolist()
        # #num = int(nums[0])
        # num = nums
        
        outputs = self.bindings['output'].data[0]#.tolist()
        t02 = time.time()
        # print(outputs.shape)
        # print(f"outputs size = {len(outputs)}")
        scores = outputs[:, 4]
        # print("scores.shpe", scores.shape)
        idx = torch.where(scores>=threshold)[0]
        # print("idx.shpe", idx.shape)
        new_bboxes = outputs[idx, :]
        # for i in range(len(outputs)):
        #     if (outputs[i][4] < threshold):
        #         continue
        #     new_bboxes.append(outputs[i]) # cx cy w h
        # for i in range(num):
            # if(scores[i] < threshold):
                # continue
            # xmin = (boxes[i][0] - self.dw)/self.r
            # ymin = (boxes[i][1] - self.dh)/self.r
            # xmax = (boxes[i][2] - self.dw)/self.r
            # ymax = (boxes[i][3] - self.dh)/self.r
            # new_bboxes.append([classes[i],scores[i],xmin,ymin,xmax,ymax])
        # print(f"after = {new_bboxes.shape[0]}")
        # new_bboxes = torch.stack(new_bboxes, 0)
        # print(new_bboxes.shape)
        t021 = time.time()
        output_box = self.nms.nms(new_bboxes, 0.65)
        t03 = time.time()
        print("preprocess", (t01-t00)*1000)
        print("infer", (t02-t01)*1000)
        print("postprocess", (t03-t02)*1000)
        print(" - nms", (t03-t021)*1000)
        
        return output_box, img

    

def visualize(img,bbox_array):
    for temp in bbox_array:
        xmin = int(temp[0] - temp[2]/2)
        ymin = int(temp[1] - temp[3]/2)
        xmax = int(temp[0] + temp[2]/2)
        ymax = int(temp[1] + temp[3]/2)
        #clas = int(temp[0])
        score = temp[4]
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax), (105, 237, 249), 2)
        #img = cv2.putText(img, "class:"+str(clas)+" "+str(round(score,2)), (xmin,int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 1)

    return img

    
trt_engine = TRT_engine("./yolov7w6_640.engine")
img = cv2.imread('2.jpg')
img = cv2.resize(img, (1920, 1080))
print(img.shape)
# i = 0
# while(i < 10):
#     results,_ = trt_engine.predict(img,threshold=0.5)
#     i+=1

i=0
sumtime = 0
iter = 10
while(i<iter):
    tic1 = time.perf_counter()
    results, img_ = trt_engine.predict(img, threshold=0.5)
    print("results:", results)
    toc1 = time.perf_counter()
    print(f"(Total) one img infer time = {(toc1-tic1)*1000} ms")
    print('-'*30)
    sumtime += (toc1-tic1)
    i+=1

print("="*30)
print(f"(Total)Avg infer time = {(sumtime/iter)*1000} ms")
img = img_.squeeze(0)
nimg = img.permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

nimg = visualize(nimg,results)

print(nimg.shape)
cv2.imwrite("out1.jpg",nimg)
