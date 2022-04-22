"""
-- coding: utf-8 --
@Author      :   Zhengyi Li 
@Time        :   2022/04/22 06:58:07
@Description :   
"""

from __future__ import division
import torch
import torch.nn as nn
import cv2
import time
from torch.autograd import Variable
import numpy as np
import argparse
import os
import os.path as osp
import pickle as pkl
import pandas as pd
import random
from detector.utils import *
from detector.DarkNet import Darknet

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")

    parser.add_argument("--images", dest='images', help = 
                        "Image / Directory containing images to perform detection upon", 
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()

def load_classes():
    fp = open('./data/coco.names', 'r')
    names = fp.read().split('\n')[: -1]

    return names

def write_(x, result, colors):
    c1 = tuple(x[1: 3].int())   # x and y
    c2 = tuple(x[3: 5].int())   # bx and by
    img = result[int(x[0])]     # define the image
    cls = int(x[-1])            # define the classes
    label = "{0}".format(classes[cls])
    color = random.choice(colors)      
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

    return img

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
num_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80    # For COCO dataset
classes = load_classes()

print('loading the network...')
model = Darknet()
model.load_weights()
print('network has been loaded successfully...')

model.net_info['height'] = args.reso
inp_dim = int(model.net_info['height'])

assert inp_dim > 32

if CUDA:
    model = model.cuda()

model.eval()

videofile = './movie.avi'
cap = cv2.VideoCapture(videofile)

assert cap.isOpened(), 'Cannot capture source'
frames = 0
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf=num_thesh)

        if type(output) == int:
            frame += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1: 5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])
        
        fp = open('./pallete', 'rb')
        colors = pkl.load(fp)

        list(map(lambda x: write_(x, frame, colors), output))

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)

        if key & 0xff == ord('q'):
            break

        frame += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break