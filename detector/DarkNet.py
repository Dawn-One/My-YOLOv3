"""
-*- coding: utf-8 -*-
@Author      :   Zhengyi Li 
@Time        :   2022/04/12 14:29:11
@Description :   DarkNet is the name of the underlying architecture of YOLO. This file will 
                 contain the code that creates the YOLO network. 
"""

from __future__ import division
from .utils import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EmptyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors) -> None:
        super().__init__()
        self.anchors = anchors

class Darknet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = parse_cfg('./cfg/yolov3.cfg')
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1: ]
        output = {}     # # We cache the outputs for the route layer

        # If write is 0, it means the collector hasn't been initialized. 
        # If it is 1, it means that the collector has been initialized 
        # and we can just concatenate our detection maps to it.
        write = 0
        for i, module in enumerate(modules):
            module_type = module['type']

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]

                if (layers[0] > 0):
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = output[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = output[i + layers[0]]
                    map2 = output[i + layers[1]]

                    x = torch.cat((map1, map2), 1)      # the format `B X C X H X W. The depth corresponding the the channel dimension
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = output[i-1] + output[i+from_]
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                numClass = int(module['classes'])

                x = x.data
                x = predict_transfrom(x, inp_dim=inp_dim, num_classes=numClass, anchors=anchors, CUDA=CUDA)
                
                if not write:
                    detection = x
                    write = 1
                else:
                    detection = torch.cat((detection, x), 1)
            output[i] = x
        return detection

    def load_weights(self):
        fp = open('./yolov3.weights', 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weight = np.fromfile(fp, dtype=np.float32)

        ptr = 0     # keep track of where we are in the weights array
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']

            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]
            
                if batch_normalize:
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()     # ???

                    bn_biases = torch.from_numpy(weight[ptr: ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weight[ptr: ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weight[ptr: ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weight[ptr: ptr+num_bn_biases])
                    ptr += num_bn_biases

                    # cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weight[ptr: ptr+num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weight[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)





def parse_cfg(cfgfile: str) -> list:
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]    # 去除空白行
    lines = [x for x in lines if x[0] != '#']   # 去除注释
    lines = [x.lstrip().rstrip() for x in lines]  # 去除两边的空格
    
    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1: -1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks: list) -> nn.ModuleList:
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3    # track the previous filter's number
    output_filters = []

    for index, x in enumerate(blocks[1: ]):
        modules = nn.Sequential()

        #check the type of block
        #create a new module for the block
        #append to module_list

        if x['type'] == 'convolutional':
            activation = x['activation']

            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # add convolutional layer
            conv = nn.Conv2d(in_channels=prev_filters, out_channels=filters, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)
            modules.add_module('conv{0}'.format(index), conv)

            # add B-N layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                modules.add_module('batch_norm{0}'.format(index), bn)
            
            # add activation
            if activation == 'leaky':
                acti = nn.LeakyReLU(0.1, inplace=True)
                modules.add_module('leaky{0}'.format(index), acti)

        #If it's an upsampling layer
        #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            modules.add_module("upsample_{}".format(index), upsample)

        elif (x['type'] == 'route'):
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            modules.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        #shortcut corresponds to skip connection
        elif x['type'] == "shortcut":
            shortcut = EmptyLayer()
            modules.add_module("shortcut_{}".format(index), shortcut)

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(anchor) for anchor in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            modules.add_module("Detection_{0}".format(index), detection)
        
        module_list.append(modules)
        prev_filters = filters
        output_filters.append(filters)
    
    return net_info, module_list

def get_test():
    img = cv2.imread('./dog-cycle-car.png')
    img = cv2.resize(img, (416, 416))
    img_ =  img[:, :, : : -1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis, :, :, :] / 255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


# model = Darknet()
# model.load_weights()




