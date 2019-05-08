# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================

import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from PIL import Image
from core import utils
from core.dataset import Parser, dataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecords_file',
                           './data/objects365/tfrecord/*.tfrecords',
                           'tfrecords file_path')

tf.app.flags.DEFINE_string('anchors_file',
                           './data/objects365/o365_anchors.txt',
                           'anchors file path')

tf.app.flags.DEFINE_string('names_file',
                           './data/objects365/object365.names',
                           'names file')

def main(unused):
    sess = tf.Session()

    IMAGE_H, IMAGE_W = 416, 416
    BATCH_SIZE = 1
    SHUFFLE_SIZE = 2

    train_tfrecord = FLAGS.tfrecords_file
    anchors = utils.get_anchors(FLAGS.anchors_file, IMAGE_H, IMAGE_W)
    classes = utils.read_coco_names(FLAGS.names_file)
    num_classes = len(classes)

    parser = Parser(IMAGE_H, IMAGE_W, anchors, num_classes, debug=True)
    trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)

    is_training = tf.placeholder(tf.bool)
    example = trainset.get_next()

    for l in range(1):
        image, boxes = sess.run(example)
        image, boxes = image[0], boxes[0]

        n_box = len(boxes)
        for i in range(n_box):
            image = cv2.rectangle(image, (int(float(boxes[i][0])),
                                          int(float(boxes[i][1]))),
                                         (int(float(boxes[i][2])),
                                          int(float(boxes[i][3]))), (255, 0, 0), 1)
            label = classes[boxes[i][4]-1]
            image = cv2.putText(image, label, (int(float(boxes[i][0])), int(float(boxes[i][1]))),
                                cv2.FONT_HERSHEY_SIMPLEX,  .6, (0, 255, 0), 2, 2)

        image = Image.fromarray(np.uint8(image))
        image.show()

if __name__ == '__main__':
    tf.app.run()
