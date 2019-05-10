# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================
"""
学習したモデルの結果を見るためのスクリプト

"""
import numpy as np
import os
import random
from PIL import Image
from core import utils
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('names_file',
                           './data/objects365/objects365.names',
                           'クラス名が記載されているファイル')

tf.app.flags.DEFINE_string('image_dir',
                           './data/objects365/val',
                           'image directory')

tf.app.flags.DEFINE_string('pb_file',
                           './checkpoint/yolov3_cpu_nms.pb',
                           'pb file created by \'convert_tfrecord.py\'')


def main(argv):
    IMAGE_H, IMAGE_W = 416, 416
    classes = utils.read_coco_names(FLAGS.names_file)
    num_classes = len(classes)
    files = os.listdir(FLAGS.image_dir)
    random_index = random.randrange(len(files))
    image_path = os.path.join(FLAGS.image_dir, files[random_index])
    img = Image.open(image_path)
    img_resized = np.array(img.resize(size=(IMAGE_W, IMAGE_H)), dtype=np.float32)
    img_resized = img_resized / 255.
    cpu_nms_graph = tf.Graph()

    input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, FLAGS.pb_file,
                                    ['Placeholder:0', 'concat_9:0', 'mul_6:0'])

    with tf.Session(graph=cpu_nms_graph) as sess:
        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.3, iou_thresh=0.5)
        image = utils.draw_boxes(img, boxes, scores, labels, classes, [IMAGE_H, IMAGE_W], show=True)


if __name__ == '__main__':
    tf.app.run()
