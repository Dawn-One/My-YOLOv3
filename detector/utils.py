"""
-*- coding: utf-8 -*-
@Author      :   Zhengyi Li 
@Time        :   2022/04/12 14:30:31
@Description :   utils.py will contain the code for various helper functions. 
"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transfrom(prediction: torch.Tensor, inp_dim, anchors, num_classes: int, CUDA=True):
    """
    predict_transform function takes an detection feature map and turns it into a 2-D tensor, 
    where each row of the tensor corresponds to attributes of a bounding box

    @parameters:
    @prediction: our output feature map
    @inp_dim: input image dimension
    @anchors: num_classes
    @[optional] CUDA flag

    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)      # ???
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors]

    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset
