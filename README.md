# What would a picture experience in YOLO v3 ?
>The repository is not original. It is from repository [ayooshkathuria/YOLO_v3_tutorial_from_scratch](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch). His tutorial is very wonderful. [Click here](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/) to go!
## Folder Architecture:
- detector.py: main function
- detector
    - DarkNet.py: Baseline
    - utils.py: usful tool function
- cfg
    - yolov3.cfg: DarkNet's architecture and YOLO v3's net information
- yolov3.weights: YOLO v3's weights [Download Here](https://pjreddie.com/media/files/yolov3.weights)
- video.py: object detect in video

## A Picture's Experience
### load the pictures
- The picture or its folder and their infomation are feed to `detector.py` by `arg_parse`,By `arg_parse`, We can input the following parameters:
    -  `--image`: Image dictory/path
    - `--det`: Image / dictory to store detections to
    - `--bs`: Batch size
    - `--confidence`: Object confidence to filter predictions
    - `--nms_thresh`: NMS threshold
    - `--cfg`: Config file path
    - `--weights`: weights file path
    - `--reso`: Input resolution of the network
- Pictures are read by `cv2.imread` and stored a `list`
- Pictures are trimmed to a fixed size and channels
    - BGR->RGB || 416 * 416 * 3 -> 3 * 416 * 416
- Pictures are concatenated into some batches by `torch.cat`.
- We run each batch and get prediction. The `Tensor` from model is transform to `batch_size * num of bounding box * num of attrubute`. We need to perform non-maximum suppression on the prediction. The non-maximum suppression processing is as follows.
    - We drop all bounding boxes that confidence is less than a threshold.
    - Attribute 0 to 4 of prediction(`batch_size * num of bounding box * num of attrubute`) are converted to four corner coordinates of bounding boxes.
    - We go through each picture in a batch, and select the highest score and the class it belongs to as the class the bounding box detects.
    - For classes of the image combines, sort bounding box confidence in descending order.
    - Calculate `ious` each other, if `iou` is smaller than a threshold, we drop the smaller bounding box.
    - Concatenate `batch_ind` and `Tensor`. Output it!
- Mapping the bounding box coordinates to original pictures.
- Draw the bounding box and class in pictures.

## Load Model
### Parse Configure File
- function `parse_cfg` return blocks of YOLO v3
### Create Modules by Blocks
- return `net_info` (a dictory) and `module_list`(a `nn.ModuleList`)
### Load Weights
- Go through the `module_list` and copy the weights to each modules
- A variable `ptr` is used to track the weights file
- Pay attention to the order in `yolov3.weights`, convolutional layers' weight of each block is in the end. The order is `bn-bias`, `bn-weight`, `bn-running-mean`, `bn-running-var`(if batch normalization is required), `conv-bias`(if batch normalization is not required) and `conv-weights`


