
# ===============================================
# Author: RyutaShitomi
# date: 2019-05-07T08:50:16.417Z
# Description:
#
# ===============================================

python train.py --names_file='./data/objects365/objects365.names' \
                --anchor_file='./data/objects365/o365_anchors.txt' \
                --train_tfrecord_path='./data/objects365/tfrecord/train.tfrecords' \
                --val_tfrecord_path='./data/objects365/tfrecord/val.tfrecords' \
                --batch_size=16 \
                --steps=2500 \
                --lr=0.001 \
                --decay_rate=0.9 \
                --decay_steps=100 \
                --shuffle_size=200 \
                --eval_internal=100 \
                --save_internal=500
