from yolov7_pose_trt.trt_pose import TRT_engine
import cv2

from yolov9Loader import Yolov9Detector


# get one frame by opencv
def get_one_frame_from_localcam(cam):
    ret, frame = cam.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def get_person_crops(img, boxes, ignore_threshold=0.01, expand_ratio=0.15, cut_legs=False):  # 计算人体骨干
    crops, new_rects = get_person_from_rect(img, boxes, ignore_threshold=ignore_threshold, expand_ratio=expand_ratio, cut_legs=cut_legs)  # 调用get_person_from_rect进行人体骨干提取
    return crops, new_rects  #pt1, pt2

def expand_crop(images, rect, expand_ratio=0.15, cut_legs=False):  # 扩展人体框
    imgh, imgw = images.shape[:2] # HWC  获得图像size
    xmin, ymin, xmax, ymax = [int(x) for x in rect.tolist()]  # 从人体图像坐标中得出上下左右各个边界大小
    # if label != 0:   # crop person only @yjy
    #     return None, None
    org_rect = [xmin, ymin, xmax, ymax]  # 得到初始框大小
    h_half = (ymax - ymin) * (1 + expand_ratio) / 2.  # 上下方向扩展后的中点
    w_half = (xmax - xmin) * (1 + expand_ratio) / 2.  # 水平方向终点
    # if h_half > w_half * 4 / 3:
    #     w_half = h_half * 0.75
    center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]  #  图像中心点
    ymin = max(0, int(center[0] - h_half * 1.1))  # 增加向上扩张
    # ymax = min(imgh - 1, int(center[0] + h_half))
    if cut_legs:
        ymax = min(imgh - 1, int(center[0])) # 减小向下扩张 1/8
    else:
        ymax = min(imgh - 1, int(center[0] + h_half*2/3)) # 减小向下扩张 1/2
    xmin = max(0, int(center[1] - w_half))  # 水平边界最小值
    xmax = min(imgw - 1, int(center[1] + w_half))  # 水平边界最大值
    return images[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax] #, org_rect  返回扩张后的人体框

def get_person_from_rect(image, results, ignore_threshold=0.05, expand_ratio=0.15, cut_legs=False):  # 从框中提取人体
    # crop the person result from image
    # det_results = results
    # mask = det_results[:, 1] > det_threshold  # 黑屏时没有物体，下标1可能报错 @yjy
    # valid_rects = det_results[mask]

    imgh, imgw = image.shape[:2]  # 480*640*3 @yjy  得到图像的size
    rect_images = []  # 框内图像
    new_rects = []  # 新框
    # org_rects = []
    for rect in results:  # 遍历yolo检测得到的人体框
        rect_image, new_rect = expand_crop(image, rect, expand_ratio=expand_ratio, cut_legs=cut_legs)  # crop label=person @yjy  对检测到的人员框进行扩展
        if rect_image is None or rect_image.size == 0:  # 图像为空则跳过
            continue
        # cv2.imshow("Crops", rect_image[:, :, ::-1]) # RGB HWC
        # if cv2.waitKey(1) & 0xFF==ord('q'):
        #     break
        pheight, pwidth = new_rect[3]-new_rect[1], new_rect[2]-new_rect[0]   # xmin,ymin,xmax,ymax, pheight=ymax-ymin @yjy 计算框的长宽
        if pwidth*pheight < ignore_threshold*imgh*imgw:  # ignore small person @yjy  太小的人直接忽略
            continue
        rect_images.append(rect_image)  # 将框内图像保存入列表
        new_rects.append(new_rect)  # 新框的大小保存到列表
        # org_rects.append(org_rect)
    return rect_images, new_rects #, org_rects

frame_count = 0
input_size = (640,640)
show_detect_classes = [3] 
detect_classes = [3] 
localcam = cv2.VideoCapture('./test1.mp4')
result = cv2.VideoWriter('./test1_infer.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 640))
# midRes = cv2.VideoWriter('./midRes3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 640))

yolov7_trt_engine = TRT_engine("./yolov7_pose_trt/yolov7_pose_640.engine")
yolov7_helmet_phone_smoke = Yolov9Detector(weights='./weights/best400e.pt',  # 模型参数
                                                        imgsz=[640],   # 像素数量
                                                        conf_thres=0.5,  # 置信度
                                                        device="cuda", # device  # GPU
                                                        classes=detect_classes,
                                                        hide_conf=False, # 隐藏置信度
                                                        half=False,     #不使用FP16
                                                        phone_smoke_filter=False)   # 是否使用姿态来过滤手机使用和吸烟

while localcam.isOpened():
    frame_count += 1
    inps = []
    ret, frame = localcam.read()
    if frame is not None:
        if frame.shape != (640, 640, 3):  # 如果图像大小不对
            frame = cv2.resize(frame, input_size)  # 对图像进行resize
        pred, frame_ = yolov7_trt_engine.predict(frame, threshold=0.65)
        person_res,pose_result = yolov7_trt_engine.reformat_result_action(pred)
        detected = person_res.to('cpu') if person_res is not None else None
        if detected is not None:
            inps, new_rects = get_person_crops(frame, detected[:, 0:4],   # 对得到的bbox进行裁剪
                                                            ignore_threshold=0.0, 
                                                            expand_ratio=0.05, 
                                                            cut_legs=False)  # RGB HWC
            # for i, inp in enumerate(inps):
            #     cv2.imwrite(f'./middle/smoke2/{frame_count}_{i}.jpg', inp)
            for rect in new_rects:
                frame = cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            # print('num_person:',len(inps))
            # print('rects:',new_rects)
            # # for i in inps:
            #     cv2.imshow('single', i)
            #     cv2.waitKey(0)
    else:
        print('frame is None')
    if(len(inps)):  # 如果有裁剪后的图像
        dataset = yolov7_helmet_phone_smoke.dataOnPerson(imgs=inps,
                                                        ori_frame=frame, 
                                                        new_rects=new_rects)  # RGB
        # print('dataset:',dataset)
        frame, det_res = yolov7_helmet_phone_smoke.detectOnPerson(dataset=dataset, 
                                                        frame=frame, 
                                                        poses=pose_result, 
                                                        show_detect_classes=show_detect_classes)

        # print('yes')
        #save result video
        result.write(frame)

        if frame is not None:
            cv2.imshow('test_0', frame)
            key = cv2.waitKey(1)
cv2.destroyAllWindows() 
