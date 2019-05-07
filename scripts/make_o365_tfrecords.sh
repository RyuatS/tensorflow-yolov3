
# ===============================================
# Author: RyutaShitomi
# date: 2019-05-07T08:38:02.724Z
# Description:
#
# ===============================================

python scripts/extract_o365.py --json_path ./data/objects365/objects365_train.json \
                               --image_path ./data/objects365/train \
                               --dataset_info_path ./data/objects365/train.txt
python scripts/extract_o365.py --json_path ./data/objects365/objects365_val.json \
                               --image_path ./data/objects365/val \
                               --dataset_info_path ./data/objects365/val.txt

python core/convert_tfrecord.py --dataset_txt ./data/objects365/train.txt \
                                --tfrecord_path_prefix ./data/objects365/tfrecord/train
python core/convert_tfrecord.py --dataset_txt ./data/objects365/val.txt \
                                --tfrecord_path_prefix ./data/objects365/tfrecord/test
