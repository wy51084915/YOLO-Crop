"""
An example that uses TensorRT's Python api to make inferences.
"""
import threading
import time
import cv2
from pytest import importorskip
import torch
import numpy as np
# import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

def cuda_init():
    cuda.init()

class YoLov5TRT_Person(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path,trt_ctx):
        # Create a Context on this device,
        self.ctx = trt_ctx #read exist context
        self.ctx.push()# activate context
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.length_of_outputs = len(host_outputs) #@zmh
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
    
#@zmh
    def infer_webcam(self, image_raw):
        # Make self the active context, pushing it on top of the context stack.
        #self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        
        ##batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        ##np.copyto(batch_input_image[0], image_raw)
        ##batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        ##np.copyto(host_inputs[0], batch_input_image.ravel())
        np.copyto(host_inputs[0], np.ascontiguousarray(image_raw).ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        for i in range(self.length_of_outputs):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        #self.ctx.pop()
        # Here we use the last output
        ##output = host_outputs[4] #0:1632000 1:408000 2:102000 3:25500 4:2167500
        #print('output size',host_outputs[4].shape)
        pred = torch.tensor(host_outputs[4]).view(1,25500,85)#.view(1,25500,5+4) exp31 best.pt
        # print('output size',np.array(output).shape)
        # print('pred size:',pred.shape)
        # print('host_outputs type:',type(host_outputs))
        print('Person detector pure infer time->{:.3f}ms'.format((end-start) * 1000))
        
        return pred

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
# categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
#         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
#         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
#         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
#         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#         "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
#         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
#         "hair drier", "toothbrush"]

class AlphaposeTRT_Pose(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path,trt_ctx):
        # Create a Context on this device
        self.ctx = trt_ctx
        self.ctx.push()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.length_of_outputs = len(host_outputs) #@zmh
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        print("engine.max_batch_size:",engine.max_batch_size)
        self.batch_size = engine.max_batch_size
    
#@zmh
    def infer_webcam(self, image_raw):
        # Make self the active context, pushing it on top of the context stack.
        #self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        
        ##batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        ##np.copyto(batch_input_image[0], image_raw)
        ##batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        ##np.copyto(host_inputs[0], batch_input_image.ravel())
        np.copyto(host_inputs[0], np.ascontiguousarray(image_raw).ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        for i in range(self.length_of_outputs):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        #self.ctx.pop()
        # Here we use the last output
        ##output = host_outputs[4] #0:1632000 1:408000 2:102000 3:25500 4:2167500
        #print('output size',host_outputs[4].shape)
        pred = torch.tensor(host_outputs[0]).view(1,17,48,40)#.view(1,25500,5+4) exp31 best.pt
        # print('output size',np.array(output).shape)
        # print('pred size:',pred.shape)
        # print('host_outputs type:',type(host_outputs))
        
        print('Alphapose pure infer time->{:.3f}ms'.format((end-start) * 1000))
        
        return pred

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()



class YoLov5TRT_Things(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path, trt_ctx):
        # Create a Context on this device,
        #self.ctx = cuda.Device(0).make_context()
        self.ctx = trt_ctx
        self.ctx.push()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.length_of_outputs = len(host_outputs) #@zmh
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
    
#@zmh
    def infer_webcam(self, image_raw):
        # Make self the active context, pushing it on top of the context stack.
        #self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        
        ##batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        ##np.copyto(batch_input_image[0], image_raw)
        ##batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        ##np.copyto(host_inputs[0], batch_input_image.ravel())
        np.copyto(host_inputs[0], np.ascontiguousarray(image_raw).ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        for i in range(self.length_of_outputs):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        #self.ctx.pop()
        # Here we use the last output
        ##output = host_outputs[4] #0:1632000 1:408000 2:102000 3:25500 4:2167500
        #print('output size',host_outputs[4].shape)
        #pred = torch.tensor(host_outputs[4]).view(1,25500,9)#.view(1,25500,5+4) Things_m_trt_320
        pred = torch.tensor(host_outputs[0]).view(1,6300,9)# Things_s_trt_320
        # print('output size',np.array(output).shape)
        # print('pred size:',pred.shape)
        # print('host_outputs type:',type(host_outputs))
        print('Other things infer time->{:.3f}ms'.format((end-start) * 1000))
        
        return pred

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()