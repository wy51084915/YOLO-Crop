import torch
import onnx 
from models.experimental import attempt_load
import argparse
import onnx_graphsurgeon as gs 
from onnx import shape_inference
import torch.nn as nn
from utils.general import non_max_suppression
class YOLOv9AddNMS(nn.Module):
    def __init__(self, model, max_det=100, iou_thres=0.45, conf_thres=0.25, max_wh=None, device=None, n_classes=80):
        super().__init__()
        assert isinstance(max_wh,(int)) or max_wh is None
        self.device = device if device else torch.device("cpu")
        self.model = model.to(device)
        self.model.eval()
        self.max_det = max_det
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.max_wh = max_wh # if max_wh != 0 : non-agnostic else : agnostic
        self.n_classes=n_classes

    def forward(self, input):
        """ 
            Split output [n_batch, n_bboxes, 85] to 3 output: bboxes, scores, classes
        """ 
        # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # x, y, w, h -> x1, y1, x2, y2
        output = self.model(input)
        # print('Output: ', len(output))
        for x in output:
            if type(x).__name__ == 'tuple':
                print([y.shape for y in x])
            else:
                print('type x ', type(x))
        ## yolov9-c.pt and yolov9-e.pt return list output[0] is prediction of aux branch, output[1] is prediction of main branch.
        output = output[1] # https://github.com/WongKinYiu/yolov9/issues/130#issuecomment-1974792028
        output = non_max_suppression(output, self.conf_thres, self.iou_thres, self.n_classes, False, max_det=self.max_det)
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='weights path')
    parser.add_argument('--output', type=str, default='weights/yolov9.onnx', help='output ONNX model path')
    parser.add_argument('--max_size', type=int, default=640, help='max size of input image')
    parser.add_argument('--data', type=str, default="./data/Newthings.yaml", help='max size of input image')
    opt = parser.parse_args()

    model_weights = opt.weights 
    output_model_path = opt.output
    data = opt.data
    max_size = opt.max_size
    device = torch.device('cpu')

    # load model 
    model = attempt_load(model_weights, device=device, inplace=True, fuse=True)
    model.eval()
    img = torch.zeros(1, 3, max_size, max_size).to(device)
    
    for k, m in model.named_modules():
        m.export = True

    for _ in range(2):
        y = model(img)  # dry runs
    print('[INFO] Convert from Torch to ONNX')
    model = YOLOv9AddNMS(model)
    model.to(device).eval()

    torch.onnx.export(model,               # model being run
                      img,                         # model input (or a tuple for multiple inputs)
                      output_model_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    print('[INFO] Finished Convert!')