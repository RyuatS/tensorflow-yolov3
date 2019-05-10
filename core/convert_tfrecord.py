#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_tfrecord.py
#   Author      : YunYang1994
#   Created date: 2018-12-18 12:34:23
#   Description :
#
#================================================================

import sys
import argparse
import numpy as np
import os
import tensorflow as tf

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_txt", default='./data/train_data/quick_train_data/quick_train_data.txt')
    parser.add_argument("--tfrecord_path_prefix", default='./data/train_data/quick_train_data/tfrecords/quick_train_data')
    flags = parser.parse_args()

    if os.path.exists(flags.tfrecord_path_prefix + '.tfrecords'):
        print('{} already exists'.format(flags.tfrecord_path_prefix + '.tfrecords'))
        exit()


    dataset = {}
    f = open(flags.dataset_txt, 'r')
    num_lines = len(f.readlines())
    f.close()
    with open(flags.dataset_txt,'r') as f:
        for i, line in enumerate(f.readlines(), start=1):
            sys.stdout.write('\rreading image and bbox from {}. done {}/{}'.format(flags.dataset_txt,
                                                                                    i,
                                                                                    num_lines))
            sys.stdout.flush()
            example = line.split(' ')
            image_path = example[0]
            boxes_num = len(example[1:]) // 5
            boxes = np.zeros([boxes_num, 5], dtype=np.float32)
            for i in range(boxes_num):
                boxes[i] = example[1+i*5:6+i*5]
            dataset[image_path] = boxes

    image_paths = list(dataset.keys())
    images_num = len(image_paths)
    print("\n>> Processing %d images" %images_num)

    if not os.path.exists(os.path.dirname(flags.tfrecord_path_prefix)):
        os.makedirs(flags.tfrecord_path_prefix)

    tfrecord_file = flags.tfrecord_path_prefix+".tfrecords"
    with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
        for i in range(images_num):
            sys.stdout.write('\r>>createing tfrecord. done {}/{}'.format(i+1, images_num))
            sys.stdout.flush()
            image = tf.gfile.FastGFile(image_paths[i], 'rb').read()
            boxes = dataset[image_paths[i]]
            boxes = boxes.tostring()

            example = tf.train.Example(features = tf.train.Features(
                feature={
                    'image' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                    'boxes' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [boxes])),
                }
            ))

            record_writer.write(example.SerializeToString())
        print("\n>> Saving %d images in %s" %(images_num, tfrecord_file))


if __name__ == "__main__":main(sys.argv[1:])
