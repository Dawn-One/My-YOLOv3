# What would a picture experience in YOLO v3 ?
## Folder Architecture:
- detector.py: main function
- detector
    - DarkNet.py: Baseline
    - utils.py: usful tool function
- cfg
    - yolov3.cfg: DarkNet's architecture and YOLO v3's net information
- yolov3.weights: YOLO v3's weights [Download Here](https://pjreddie.com/media/files/yolov3.weights)

## A Picture's Experience
### The picture is feed to `detector.py` by `arg_parse`
By `arg_parse`, We can input the following parameters:
- `--image`: Image dictory/path
- `--det`: Image / dictory to store detections to
- `--bs`: Batch size
- `--confidence`: Object confidence to filter predictions
- `--nms_thresh`: NMS threshold
- `--cfg`: Config file path
- `--weights`: weights file path
- `--reso`: Input resolution of the network

### BGR->RGB || 416 * 416 * 3 -> 3 * 416 * 416

## Load Model
### Parse Configure File
- function `parse_cfg` return blocks of YOLO v3
### Create Modules by Blocks
- return `net_info` (a dictory) and `module_list`(a `nn.ModuleList`)
### Load Weights
- Go through the `module_list` and copy the weights to each modules
- A variable `ptr` is used to track the weights file
- Pay attention to the order in `yolov3.weights`, convolutional layers' weight of each block is in the end. The order is `bn-bias`, `bn-weight`, `bn-running-mean`, `bn-running-var`(if batch normalization is required), `conv-bias`(if batch normalization is not required) and `conv-weights`


