#coding=utf-8
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    

import cv2
import numpy as np
from collections import OrderedDict,namedtuple
import time
import os
import tensorrt as trt
import torch
import pycuda.driver as cuda

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
        # cuda.init()
        # self.cfx = cuda.Device(0).make_context(0)
        # self.cfx.push()
        
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

    '''
        results: list[result_1,result_2,...]
        result_n:   tensor.size(57)
                    result[0]: x coordinate of the center of bbox
                    result[1]: y coordinate of the center of bbox
                    result[2]: the width of bbox
                    result[3]: the height of bbox 
                    result[4]: bbox score
                    result[5]: no idea
                    result[6:23](17):  x coordinates of key points
                    result[23:40](17): y coordinates of key points
                    result[40:57](17): the conf score of each key point

    '''
    def predict(self,img,threshold):
        # self.cfx.push()
        #t00 = time.time()
        img = self.preprocess(img)
        #t01 = time.time()
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        # nums = self.bindings['num_detections'].data[0].tolist()
        # boxes = self.bindings['detection_boxes'].data[0].tolist()
        # scores =self.bindings['detection_scores'].data[0].tolist()
        # classes = self.bindings['detection_labels'].data[0].tolist()
        # #num = int(nums[0])
        # num = nums
        
        outputs = self.bindings['ouputs'].data[0]#.tolist()
        #t02 = time.time()
        scores = outputs[:, 4]
        idx = torch.where(scores>=threshold)[0]
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
        #t021 = time.time()
        output_box = self.nms.nms(new_bboxes, 0.65)
        # t03 = time.time()
        # print("preprocess", (t01-t00)*1000)
        # print("infer", (t02-t01)*1000)
        # print("postprocess", (t03-t02)*1000)
        # print(" - nms", (t03-t021)*1000)
        # self.cfx.pop()
        return output_box, img

    # 标准reformat函数
    def reformat_result(self,result):
        person_results = []
        dict_results = []
        for res in result:
            res = res.to('cpu')
            xmin = torch.floor(res[0] - res[2]/2)
            ymin = torch.floor(res[1] - res[3]/2)
            xmax = torch.floor(res[0] + res[2]/2)
            ymax = torch.floor(res[1] + res[3]/2)
            score = res[4]
            person_res = torch.tensor((xmin,ymin,xmax,ymax,score))

            key_points_x = torch.tensor(res[6:23]).unsqueeze(-1)
            key_points_y = torch.tensor(res[23:40]).unsqueeze(-1)
            key_points = torch.cat((key_points_x,key_points_y),dim=1) # 17x2
            kp_scores = torch.tensor(res[40:57]).unsqueeze(-1)

            person_results.append(person_res)
            dict_results.append(
                {
                'bbox': torch.tensor((xmin,ymin,xmax,ymax)),
                'bbox_score': score,
                'keypoints': key_points,
                'kp_score': kp_scores,
                })

        person_results = torch.stack(person_results) if len(person_results) else None # person_number x 5
        return person_results,dict_results

    # 经特殊处理reformat函数
    def reformat_result_action(self,result):
        person_results = []
        dict_results = []
        for res in result:
            res = res.to('cpu')
            xmin = torch.floor(res[0] - res[2]/2)
            ymin = torch.floor(res[1] - res[3]/2)
            xmax = torch.floor(res[0] + res[2]/2)
            ymax = torch.floor(res[1] + res[3]/2)
            score = res[4]
            person_res = torch.tensor((xmin,ymin,xmax,ymax,score))
            key_points_x = torch.cat((res[6:7],res[11:23]),dim=0).unsqueeze(-1)  # 删除掉[1,2,3,4]左眼、右眼、左耳、右耳共4个点
            key_points_y = torch.cat((res[23:24],res[28:40]),dim=0).unsqueeze(-1) # 删除掉[1,2,3,4]左眼、右眼、左耳、右耳共4个点
            key_points = torch.cat((key_points_x,key_points_y),dim=1)       # 13x2
            kp_scores = torch.cat((res[40:41],res[45:57]),dim=0).unsqueeze(-1)    # 删除掉[1,2,3,4]左眼、右眼、左耳、右耳共4个点             

            person_results.append(person_res)
            dict_results.append(
                {
                'bbox': torch.tensor((xmin,ymin,xmax,ymax)),
                'bbox_score': score,
                'keypoints': key_points,
                'kp_score': kp_scores,
                })
        
        person_results = torch.stack(person_results) if len(person_results) else None # person_number x 5
        return person_results,dict_results
    

def visualize(img,bbox_array):
    # draw box and key poing for each person
    for temp in bbox_array:
        xmin = int(temp[0] - temp[2]/2)
        ymin = int(temp[1] - temp[3]/2)
        xmax = int(temp[0] + temp[2]/2)
        ymax = int(temp[1] + temp[3]/2)
        #clas = int(temp[0])
        score = temp[4]
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax), (105, 237, 249), 2)
        #img = cv2.putText(img, "class:"+str(clas)+" "+str(round(score,2)), (xmin,int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 1)

        all_points = temp[6:]
        kpts = []
        iii = 0
        for i in range(17):
            x = all_points[i]
            y = all_points[17+i]
            conf = all_points[17*2+i]
            kpts.append([x, y, conf])
            if conf < 0.5:
                continue
            iii+=1
            cv2.circle(img, (int(x), int(y)), 5, (int(255), int(0), int(0)), -1)
            cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        #print(kpts)

    return img

# if __name__ == '__main__':
#     trt_engine = TRT_engine("../../Models/YOLOV7_POSE_TRT/pose_640_fp16.engine")
#     img = cv2.imread('1.png')
#     img = cv2.resize(img, (640, 640))
#     print(img.shape)
#     # i = 0
#     # while(i < 10):
#     #     results,_ = trt_engine.predict(img,threshold=0.5)
#     #     i+=1

#     i=0
#     sumtime = 0
#     iter = 10
#     while(i<iter):
#         tic1 = time.perf_counter()
#         results, img_ = trt_engine.predict(img, threshold=0.5)
#         results_reformat = trt_engine.reformat_result(results)
#         toc1 = time.perf_counter()
#         print(f"(Total) one img infer time = {(toc1-tic1)*1000} ms")
#         print('-'*30)
#         sumtime += (toc1-tic1)
#         i+=1

#     print("="*30)
#     print(f"(Total)Avg infer time = {(sumtime/iter)*1000} ms")
#     img = img_.squeeze(0)
#     nimg = img.permute(1, 2, 0) * 255
#     nimg = nimg.cpu().numpy().astype(np.uint8)
#     nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

#     nimg = visualize(nimg,results)

#     print(nimg.shape)
#     nimg = cv2.resize(nimg,(0,0),fx=2.0,fy=2.0)
#     cv2.imshow("img",nimg)
#     cv2.waitKey(0)
