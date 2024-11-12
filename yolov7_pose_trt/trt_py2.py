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
# os.environ['path'] += ";E:\TensorRT-8.4.1.5.Windows10.x86_64.cuda-10.2.cudnn8.4\TensorRT-8.4.1.5\lib"
import tensorrt as trt
import torch

delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
scoreThreds = 0.3
matchThreds = 5
areaThres = 0  # 40 * 40.5
alpha = 0.1


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
            # print("new_unpad",new_unpad)
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
        
        outputs = self.bindings['ouputs'].data[0]#.tolist()
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
            # cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

            # print(i, ":", x, ' ', y)
        # print(kpts)

    return img


def resizeResults(results, img_shape, res_shape, top_left, right_down):
    #img_shape: 1080*1920
    #res_shape:640*640
    k1 = img_shape[0]/res_shape[0]
    k2 = img_shape[1]/res_shape[1]
    k = max(k1, k2)
    k3 = img_shape[0]/img_shape[1]
    margin = (res_shape[0]-res_shape[1]*k3)/2
    # print(k1," ",k2,"",margin)
    result_list = results.tolist()
    final_result = []
    for res in result_list:
        res[0] = res[0]*k
        res[1] = (res[1]-margin)*k
        res[2] = res[2]*k
        res[3] = res[3]*k
        for i in range(17):
            res[i+6] = res[i+6]*k
            res[i+23] = (res[i+23]-margin)*k
        # print((res[21]+res[22])/2, ' ',(res[38]+res[39])/2)
        if ((res[21]+res[22])/2 > top_left[0]) & ((res[21]+res[22])/2 < right_down[0]) \
            & ((res[38]+res[39])/2 > top_left[1]):
            final_result.append(res)
        else:
            continue
    results = torch.tensor(final_result).to('cuda:0')
    return results



def getRects(keypoint_results,
            imshape = [640,640],
            visual_thresh=0.3  # 可见性
            ):
    # 获取人体关键点和方框坐标
    skeletons = [kp['keypoints'].tolist() for kp in keypoint_results]
    scores = [kp['kp_score'].tolist() for kp in keypoint_results]
    person_bbox = [kp['bbox'].tolist() for kp in keypoint_results]

    # print("skeletons:",skeletons)
    # print("scores:",scores)
    # print("person_bbox:",person_bbox)
    # 统计检测出姿态的人数
    num_person = len(skeletons)
    print("num_person:", num_person)
    # 需要用到的骨架点坐标要先进行定义，若没有检测到返回为空
    nose = []
    right_ear = []
    left_ear = []
    right_hand = []
    left_hand = []
    right_shoulder = []
    left_shoulder = []
    right_ankle = []
    left_ankle = []

    bodyRects = []

    right_hand_center = []
    left_hand_center = []
    if num_person>0 :
        for j in range(num_person):
            # 计算人体方框高、宽
            xmin, ymin, xmax, ymax = person_bbox[j]
            pwidth, pheight = xmax-xmin, ymax-ymin
            # print("pwidth",pwidth)
            # print("pheight",pheight)
            # print("imshape[0]",imshape[0])
            # print("imshape[1]",imshape[1])
            if pwidth*pheight < 0.01 * imshape[0]*imshape[1]:#@wh
                continue
            # 计算右手、左手坐标   
            right_hand_vis = scores[j][10][0] > visual_thresh
            left_hand_vis = scores[j][9][0] > visual_thresh

            # 计算右肘、左肘坐标
            right_elbow_vis = scores[j][8][0] > visual_thresh # 8
            left_elbow_vis = scores[j][7][0] > visual_thresh # 7
            

            # 计算双耳、鼻子的坐标
            right_ear_vis = scores[j][4][0] > visual_thresh
            left_ear_vis = scores[j][4][0] > visual_thresh
            nose_vis = scores[j][0][0] > visual_thresh

            # 计算双肩的坐标
            right_shoulder_vis = scores[j][6][0] > visual_thresh
            left_shoulder_vis = scores[j][5][0] > visual_thresh

            # 计算脚踝的坐标
            right_ankle_vis = scores[j][16][0] > visual_thresh
            left_ankle_vis = scores[j][15][0] > visual_thresh

            # 护目镜、口罩部分
            if right_ear_vis and left_ear_vis and nose_vis:
                right_ear = skeletons[j][4][:2]
                left_ear = skeletons[j][3][:2]
                nose = skeletons[j][0][:2]

            # 右手
            if right_hand_vis and right_elbow_vis:#x2-x0 = λ(x1-x0) #4/3
                right_hand = skeletons[j][10][:2]  # 10  list 不支持一次性读取一列
                right_elbow = skeletons[j][8][:2]
                right_hand_center = [int((23/15)*right_hand[0]-(8/15)*right_elbow[0]), int((23/15)*right_hand[1]-(8/15)*right_elbow[1])]#求手中心坐标
            
            # 左手
            if left_hand_vis and left_elbow_vis:
                left_hand = skeletons[j][9][:2]   # 9
                left_elbow = skeletons[j][7][:2]
                left_hand_center = [int((23/15)*left_hand[0]-(8/15)*left_elbow[0]), int((23/15)*left_hand[1]-(8/15)*left_elbow[1])]#求手中心坐标

            # 衣服、鞋子部分
            if right_shoulder_vis and left_shoulder_vis and right_ankle_vis and left_ankle_vis:
                right_shoulder = skeletons[j][6][:2]
                left_shoulder = skeletons[j][5][:2]
                right_ankle = skeletons[j][16][:2]
                left_ankle = skeletons[j][15][:2]


            # 输入到list
            bodyRects.append(right_ear)# 右耳
            bodyRects.append(left_ear)# 左耳
            bodyRects.append(nose)# 鼻子
            bodyRects.append(right_hand)# 右手
            bodyRects.append(left_hand)# 左手
            bodyRects.append(right_shoulder)# 右肩
            bodyRects.append(left_shoulder)# 左肩
            bodyRects.append(right_ankle)# 右脚踝
            bodyRects.append(left_ankle)# 左脚踝

    return bodyRects,right_hand_center,left_hand_center



def getPoses(results):
    tt0 = time.time()
    result_cpu = results.cpu()
    final_result = []
    final_bbox = []
    tt1 = time.time()
    print("gpu to cpu:", (tt1-tt0)*1000)
    for res in result_cpu:
        tt2 = time.time()
        temp = res[0:4].tolist()
        bbox=[]
        bbox.append(temp[0] - temp[2]/2)
        bbox.append(temp[1] - temp[3]/2)
        bbox.append(temp[0] + temp[2]/2)
        bbox.append(temp[1] + temp[3]/2)
        final_bbox.append(bbox)
        bbox=torch.tensor(bbox)
        tt3 = time.time()
        print("bbox:", (tt3-tt2)*1000)
        bbox_score = res[4]
        result_list = res.tolist()
        key_list = []
        for i in range(17):
            key_list.append(result_list[i+6])
            key_list.append(result_list[i+23])
        key_points = torch.tensor(key_list).view(17,2)
        tt4 = time.time()
        print("keypoints:", (tt4-tt3)*1000)
        kp_score = res[40:].view(17,1)
        
        tt5=time.time()
        print("kpscore:", (tt5-tt4)*1000)
        final_result.append({
                'bbox': bbox,
                'bbox_score': bbox_score,
                'keypoints': key_points,
                'kp_score': kp_score
            })
        print("append time:", (time.time()-tt5)*1000)
    return final_result, final_bbox

trt_engine = TRT_engine("../weights/pose_640_fp16.engine")
img = cv2.imread('test1.png')
img = cv2.resize(img, (1920, 1080))
print(img.shape)
# i = 0
# while(i < 10):
#     results,_ = trt_engine.predict(img,threshold=0.5)
#     i+=1
img_shape = [1080, 1920]
res_shape = [640, 640]
topleftX = 600
topleftY = 600
rightdownX = 1000
rightdownY = 900


i=0
sumtime = 0
iter = 10
while(i<iter):
    tic1 = time.perf_counter()
    results, img_ = trt_engine.predict(img, threshold=0.5)
    toc1 = time.perf_counter()
    print(f"(Total) one img infer time = {(toc1-tic1)*1000} ms")
    print('-'*30)
    sumtime += (toc1-tic1)
    i+=1

print("="*30)
t0=time.time()
results = resizeResults(results, img.shape[:2], res_shape, [topleftX, topleftY], [rightdownX, rightdownY])
# print("results:", results)
t1=time.time()
print(f"Get results time = {(t1-t0)*1000} ms")
poses , bbox = getPoses(results)
# print("poses:", poses)
# print("bbox:", bbox)
t2=time.time()
print(f"Get poses time = {(t2-t1)*1000} ms")
body_rects, right_hand_center, left_hand_center = getRects(keypoint_results=poses, imshape=img.shape[:2], visual_thresh=0.3)
print("body_rects",body_rects)
print("right_hand_center",right_hand_center)
print("left_hand_center",left_hand_center)
t3=time.time()
print(f"Get body rects time = {(t3-t2)*1000} ms")
print(f"(Total)Avg infer time = {(sumtime/iter)*1000} ms")

# img = img_.squeeze(0)
# nimg = img.permute(1, 2, 0) * 255
# nimg = nimg.cpu().numpy().astype(np.uint8)
# nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

nimg = visualize(img,results)

nimg = cv2.resize(nimg, (960, 540))
cv2.imshow("img",nimg)
cv2.waitKey(0)
