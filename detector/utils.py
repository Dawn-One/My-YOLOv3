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
    @inp_dim: input image's height
    @anchors: num_classes
    @[optional] CUDA flag

    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    # grid_size = prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors]

    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])    # b_x
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])    # b_y
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])    # score

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)

    if CUDA:
        x_y_offset = x_y_offset.cuda()

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    prediction[:, :, 5: 5+num_classes] = torch.sigmoid(prediction[:, :, 5: 5+num_classes])

    prediction[:, :, :4] *= stride

    return prediction

def write_results(prediction: torch.Tensor, confidence, num_classes, nms_conf=0.4):
    """
    we must subject our output to objectness score thresholding and Non-maximal suppression, 
    to obtain what I will call in the rest of this post as the true detections.
    
    parameters:
    @prediction: contains information about B x 10647 bounding boxes.
    @confidence: objectness score threshold
    @num_class: 
    @num_conf: the NMS IoU threshold

    """

    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction *= conf_mask

    # transform the (center x, center y, height, width) attributes of our boxes, 
    # to (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y).
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = 0

    for ind in range(batch_size):
        image_pred = prediction[ind]

        max_conf, max_conf_score = torch.max(image_pred[:, 5: 5+num_classes], 1)    # each bounding box only predict a target
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)     # merge tensors
        image_pred = torch.cat(seq, 1)

        non_zero_ind = torch.nonzero(image_pred[:, 4])

        # The try-except block is there to handle situations where we get no detections. 
        # In that case, we use continue to skip the rest of the loop body for this image.
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue
        img_classes = unique(image_pred_[:, -1])    # class that the image include

        for cls in img_classes:     # go through each classes
            # get the detections with one particular class, drop the other classes
            cls_mask = image_pred_ * ((image_pred_[:,-1] == cls).float().unsqueeze(1))
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections

            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = (batch_ind, image_pred_class)

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    try:
        return output
    except:
        return 0

def unique(tensor: torch.Tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new_empty(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)

    return tensor_res
        
def bbox_iou(box1, box2):
    """
    return the IOU between the two boxes
    
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def letterbox_image(img, inp_dim):
    '''
    resize image with unchanged aspect ratio using padding,
    keeping the aspect ratio consistent, and padding the left out areas with the color (128,128,128)

    '''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim

    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2: (h-new_h)//2 + new_h, (w-new_w)//2: (w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas
