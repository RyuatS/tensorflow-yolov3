# coding=utf-8

# ===============================================
# Author: RyutaShitomi
# date: 2019-04-18T13:07:20.248Z
# Description:
#
# ===============================================

from core import utils, yolov3
from core.dataset import dataset, Parser
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# set file
tf.app.flags.DEFINE_string('names_file',
                           './data/objects365/object365.names',
                           'class label names file')

tf.app.flags.DEFINE_string('anchor_file',
                           './data/objects365/o365_anchors.txt',
                           'file contains anchor size.')

tf.app.flags.DEFINE_string('train_tfrecord_path',
                           './data/objects365/tfrecord/train.tfrecords',
                           'train tfrecord path')

tf.app.flags.DEFINE_string('val_tfrecord_path',
                           './data/objects365/tfrecord/val.tfrecords',
                           'validation tfrecord path')

# set hyper parameter
tf.app.flags.DEFINE_integer('batch_size',
                            32,
                            'batch size')

tf.app.flags.DEFINE_integer('steps',
                            2500,
                            'a number of steps')

tf.app.flags.DEFINE_float('lr',
                            0.001,
                            'learning rate. if Nan, set 0.0005, 0.0001')

tf.app.flags.DEFINE_integer('decay_steps',
                            100,
                            'decay steps')

tf.app.flags.DEFINE_float('decay_rate',
                            0.9,
                            'decay rate')

tf.app.flags.DEFINE_integer('shuffle_size',
                            200,
                            'shuffle size for dataset.')

tf.app.flags.DEFINE_integer('eval_internal',
                            100,
                            'evaluate interval')

tf.app.flags.DEFINE_integer('save_internal',
                            500,
                            'save interval.')



def main(argv):
    sess = tf.Session()
    IMAGE_H, IMAGE_W = 416, 416
    BATCH_SIZE       = FLAGS.batch_size
    STEPS            = FLAGS.steps
    LR               = FLAGS.lr
    DECAY_STEPS      = FLAGS.decay_steps
    DECAY_RATE       = FLAGS.decay_rate
    SHUFFLE_SIZE     = FLAGS.shuffle_size
    CLASSES          = utils.read_coco_names(FLAGS.names_file)
    ANCHORS          = utils.get_anchors(FLAGS.anchor_file, IMAGE_H, IMAGE_W)
    NUM_CLASSES      = len(CLASSES)
    EVAL_INTERNAL    = FLAGS.eval_internal
    SAVE_INTERNAL    = FLAGS.save_internal

    train_tfrecord = FLAGS.train_tfrecord_path
    val_tfrecord = FLAGS.val_tfrecord_path

    parser = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
    trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)
    valset = dataset(parser, val_tfrecord, BATCH_SIZE, shuffle=None)

    is_training = tf.placeholder(tf.bool)
    example = tf.cond(is_training, lambda: trainset.get_next(), lambda: valset.get_next())

    images, *y_true = example
    model = yolov3.yolov3(NUM_CLASSES, ANCHORS)

    with tf.variable_scope('yolov3'):
        pred_feature_map = model.forward(images, is_training=is_training)
        loss             = model.compute_loss(pred_feature_map, y_true)
        y_pred           = model.predict(pred_feature_map)

    tf.summary.scalar('loss/coord_loss', loss[1])
    tf.summary.scalar('loss/sizes_loss', loss[2])
    tf.summary.scalar('loss/confs_loss', loss[3])
    tf.summary.scalar('loss/class_loss', loss[4])

    global_step  = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    write_op     = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter('./data/train')
    writer_val   = tf.summary.FileWriter('./data/val')

    saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=['yolov3/darknet-53']))
    update_vars      = tf.contrib.framework.get_variables_to_restore(include=['yolov3/yolo-v3'])
    learning_rate    = tf.train.exponential_decay(LR, global_step, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # set dependencies for BN ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss[0], var_list=update_vars, global_step=global_step)

    if(tf.__version__.startswith('0.') and int(tf.__varsion__.split('.')[1])<12):
        sess.run([tf.initialize_all_variables(), tf.initialize_local_variables()])
    else:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver_to_restore.restore(sess, './checkpoint/yolov3.ckpt')
    saver = tf.train.Saver(max_to_keep=2)

    for step in range(STEPS):
        run_items = sess.run([train_op, write_op, y_pred, y_true] + loss, feed_dict={is_training: True})

        if (step+1) % EVAL_INTERNAL == 0:
            train_rec_value, train_prec_value = utils.evaluate(run_items[2], run_items[3])

        writer_train.add_summary(run_items[1], global_step=step)
        writer_train.flush() # Flushes the event file to disk
        if (step+1) % SAVE_INTERNAL==0:
            saver.save(sess, save_path='./checkpoint/yolov3.ckpt', global_step=step+1)

        print('=> STEP %10d [TRAIN]:\tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f'
            %(step+1, run_items[5], run_items[6], run_items[7], run_items[8]))

        run_items = sess.run([write_op, y_pred, y_true] + loss, feed_dict={is_training:False})
        if (step+1) % EVAL_INTERNAL == 0:
            print("\n=======================> evaluation result <================================\n")
            print("=> STEP %10d [TRAIN]:\trecall:%7.4f \tprecision:%7.4f" %(step+1, train_rec_value, train_prec_value))
            print("=> STEP %10d [VALID]:\trecall:%7.4f \tprecision:%7.4f" %(step+1, test_rec_value,  test_prec_value))
            print("\n=======================> evaluation result <================================\n")

        writer_val.add_summary(run_items[0], global_step=step)
        writer_val.flush() # Flushes the event file to disk


if __name__ == '__main__':
    tf.app.run()
