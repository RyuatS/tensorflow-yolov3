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

6. Train. 以下のファイルを実行すれば訓練が開始される。もし、ハイパーパラメータを変更したい場合は、ファイル内の引数を変えることによって変更できる。
```bashrc
$sh train.sh
```

7. Evaluate and Test. 訓練が終わると、dir`./checkpoints`にチェックポイントが作られる。これを、以下のコードで変換する`.pb`ファイルに変換する。
```bashrc
$python convert_weight.py -cf <checkpoint_path> -nc 1 -ap ./data/objects365/o365_anchors.txt --freeze
$python quick_test.py
$python evaluate.py
```

```bashrc
$ python convert_weight.py --convert --freeze
```
4. Then you will get some `.pb` files in the dir `./checkpoint`,  and run the demo script
```bashrc
$ python nms_demo.py
$ python video_demo.py # if use camera, set video_path = 0
```
![image](./docs/images/611_result.jpg)
## part 3. Train on your own dataset
Three files are required as follows:

- `dataset.txt`:

```
xxx/xxx.jpg 18.19 6.32 424.13 421.83 20 323.86 2.65 640.0 421.94 20
xxx/xxx.jpg 55.38 132.63 519.84 380.4 16
# image_path x_min y_min x_max y_max class_id  x_min y_min ... class_id
```
- `anchors.txt`

```
0.10,0.13, 0.16,0.30, 0.33,0.23, 0.40,0.61, 0.62,0.45, 0.69,0.59, 0.76,0.60,  0.86,0.68,  0.91,0.76
```

- `class.names`

```
person
bicycle
car
...
toothbrush
```
