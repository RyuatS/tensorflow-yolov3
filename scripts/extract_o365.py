# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-05-07T06:35:26.353Z
# Description:
#
# ===============================================

import os
import sys
import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

def main(argv):

    o365_dir = os.path.join('.', 'data', 'objects365')
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default=o365_dir + '/objects365_Tiny_val.json')
    parser.add_argument('--image_path', default=o365_dir + '/val')
    parser.add_argument('--dataset_info_path', default= o365_dir + '/tiny_val.txt')
    flags = parser.parse_args()

    dataset = defaultdict(list)
    with open(os.path.realpath(flags.dataset_info_path), 'w') as f:
        labels = json.load(open(flags.json_path, encoding='utf-8'))
        annotations = labels['annotations']

        image_folder_path = os.path.realpath(flags.image_path)

        for annotation in annotations:
            image_id = annotation['image_id']

            if 'val' in flags.json_path:
                single_image_path = os.path.join(image_folder_path, 'obj365_val_%012d.jpg' %image_id)
            elif 'train' in flags.json_path:
                single_image_path = os.path.join(image_folder_path, 'obj365_train_%012d.jpg' %image_id)

            category_id = annotation['category_id']

            x_min, y_min, width, height = annotation['bbox']
            x_max = x_min+width
            y_max = y_min+height
            box = [x_min, y_min, x_max, y_max]
            dataset[single_image_path].append([category_id, box])

        for single_image_path in dataset.keys():
            write_content = [single_image_path]
            for category_id, box in dataset[single_image_path]:
                x_min, y_min, x_max, y_max = box
                write_content.append(str(x_min))
                write_content.append(str(y_min))
                write_content.append(str(x_max))
                write_content.append(str(y_max))
                write_content.append(str(category_id))
            write_content = " ".join(write_content)
            print(write_content)
            f.write(write_content+'\n')

if __name__ == '__main__': main(sys.argv[1:])
