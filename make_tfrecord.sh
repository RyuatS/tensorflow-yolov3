
# ===============================================
# Author: RyutaShitomi
# date: 2019-05-10T05:51:44.208Z
# Description:
#
# ===============================================

# まず、coco形式アノテーションのtrain.jsonと画像が保存しているパスを与えて、訓練用のフォーマット(dataset_info_path)に変換する
python scripts/extract_o365.py --json_path=data/objects365/objects365_train.json \
                               --image_path=data/objects365/train \
                               --dataset_info_path=data/objects365/train.txt

#valに対しても同様の処理を行う
python scripts/extract_o365.py --json_path=data/objects365/objects365_val.json \
                               --image_path=data/objects365/val  \
                               --dataset_info_path=data/objects365/val.txt

# convert train data to tfrecord.
python core/convert_tfrecord.py --dataset_txt=data/objects365/train.txt \
                                --tfrecord_path_prefix=data/objects365/tfrecord/train

# convert val data to tfrecord.
python core/convert_tfrecord.py --dataset_txt=data/objects365/val.txt \
                                --tfrecord_path_prefix=data/objects365/tfrecord/val


# get anchors size using k-means algorithm,
# python kmeans.py --dataset_txt=./data/objects365/train.txt \
#                  --anchors_txt=./data/objects365/o365_anchors.txt \
#                  --cluster_num=9
