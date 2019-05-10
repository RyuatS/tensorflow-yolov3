# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================
"""
non-maximum-suppressionをCPUで実行した時とGPUで実行した時の違いを見るデモコード.

USAGE)

"""
import time
import numpy as np
from PIL import Image
from core import utils
from random import randrange
import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('names_file',
                            './data/objects365/objects365.names',
                            'name file which is written about class name')

tf.app.flags.DEFINE_string('image_dir',
                           './data/objects365/val',
                           'demo image')


def main(unused):
    IMAGE_H, IMAGE_W = 416, 416
    EPOCHS = 5
    classes = utils.read_coco_names(FLAGS.names_file)
    num_classes = len(classes)
    image_path_list = os.listdir(FLAGS.image_dir)
    random_index = randrange(len(image_path_list))
    image_path = os.path.join(FLAGS.image_dir, image_path_list[random_index])
    img = Image.open(image_path)
    img_resized = np.array(img.resize(size=(IMAGE_H,  IMAGE_W)), dtype=np.float32)
    img_resized = img_resized / 255
    cpu_nms_graph, gpu_nms_graph = tf.Graph(), tf.Graph()

    # nms on GPU
    input_tensor, output_tensors = utils.read_pb_return_tensors(gpu_nms_graph, './checkpoint/yolov3_gpu_nms.pb',
                                                ['Placeholder:0', 'concat_10:0', 'concat_11:0', 'concat_12:0'])
    with tf.Session(graph=gpu_nms_graph) as sess:
        for i in range(EPOCHS):
            start = time.time()
            boxes, scores, labels = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
            print('=> nms on gpu the number of boxes= %d time %.2f ms' %(len(boxes), 1000*(time.time()-start)))
        image = utils.draw_boxes(img, boxes, scores, labels, classes, [IMAGE_H, IMAGE_W], show=True)

    # nms on CPU
    input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, './checkpoint/yolov3_cpu_nms.pb',
                                            ['Placeholder:0', 'concat_9:0', 'mul_6:0'])
    with tf.Session(graph=cpu_nms_graph) as sess:
        for i in range(EPOCHS):
            start = time.time()
            boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
            boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.5, iou_thresh=0.5)
            print('=> nms on cpu the number of boxes= %d time=%.2f ms ' %(len(boxes), 1000*(time.time() - start)))

        image = utils.draw_boxes(img, boxes, scores, labels, classes, [IMAGE_H, IMAGE_W], show=True)


if __name__ == '__main__':
    tf.app.run()
