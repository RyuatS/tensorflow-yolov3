## part 1. Introduction

Implementation of YOLO v3 object detector for objects365 dataset in Tensorflow (TF-Slim). This repository  is forked from [YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3).
In this project we cover several segments as follows:<br>
- [x] [YOLO v3 architecture](https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/yolov3.py)
- [x] Weights converter (util for exporting loaded COCO weights as TF checkpoint)
- [x] Basic working demo
- [x] Non max suppression on the both `GPU` and `CPU` is supported
- [x] Training pipeline
- [x] Compute COCO mAP


## part 2. Train on objects365 dataset.
1. Clone this file
```bashrc
$ git clone https://github.com/RyuatS/tensorflow-yolov3.git
```
2.  You are supposed  to install some dependencies before getting out hands with these codes.
```bashrc
$ cd tensorflow-yolov3
$ pip install -r ./docs/requirements.txt
```
3. Download objects365 dataset and put in `./data/objects365`.

4. Exporting loaded COCO weights as TF checkpoint(`yolov3.ckpt`) and frozen graph (`yolov3_gpu_nms.pb`) . If you don't have [yolov3.weights](https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3.weights). Download and put it in the dir `./checkpoint`

5. Convert dataset format to this repository's format. Then you will get `.txt`, `.tfrecords` and `.tfrecords` in the dir `./data/objects365`. This files is used by train scripts.
```bashrc
$sh make_tfrecord.sh
```

6. Train. If you want to change the hyperparameters, you can do so by changing the arguments in the `train.sh` file.
```bashrc
$sh train.sh
```

7. Evaluate and Test. Once training is complete, checkpoints will be created on dir `./checkpoints`. Convert to `.pb` file which will be converted with the following code:
```bashrc
$python convert_weight.py -cf <checkpoint_path> -nc 365 -ap ./data/objects365/o365_anchors.txt --freeze
$python test.py
$python evaluate.py
```
