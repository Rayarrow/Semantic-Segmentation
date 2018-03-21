import glob
import math
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from config import *


def commission_training_task(model, dump_home, d_train, d_val, learning_rate, momentum, nr_iter, batch_size,
                             report_interval=10, val_iter_interval=1000, iter_ckpt_interval=500):
    """
    :param dump_home: the place to write summary.
    :param val_size: specify the number of validation samples.
    :param max_nr_iter: specify the max number of iteration of each epoch. By default, the number of iterations is ceil(#{nr_training_data}/#{batch_size}). If `max_nr_iter` is less than the default number of iterations, an epoch will end in advance.
    :param val_interval: specify the number of epochs to
    :param epoch_checkpoint: specify the number of epochs to save the model.
    :return:
    """

    with tf.variable_scope('mask'):
        logits_masked = tf.boolean_mask(model.logits, model.y_mask_input, name='logit_mask')
        labels_masked = tf.boolean_mask(model.y_input, model.y_mask_input, name='label_mask')
        pred_masked = tf.boolean_mask(model.y_pred, model.y_mask_input, name='pred_mask')

    # define metrics
    with tf.variable_scope('metrics'):
        train_accuracy = tf.metrics.accuracy(labels_masked, pred_masked, name='train_accuracy')
        val_accuracy = tf.metrics.accuracy(labels_masked, pred_masked, name='val_accuracy')

        train_meaniou = tf.metrics.mean_iou(labels_masked, pred_masked, model.num_classes, name='train_meaniou')
        val_meaniou = tf.metrics.mean_iou(labels_masked, pred_masked, model.num_classes, name='val_meaniou')

    # inspect checkpoint home and summary home in case they do not exist.
    ckpt_home = join(dump_home, 'checkpoint')
    summary_home = join(dump_home, 'summary')

    if not os.path.exists(ckpt_home):
        os.makedirs(ckpt_home)
        logger.info('{} does not exists and created.'.format(ckpt_home))

    if not os.path.exists(summary_home):
        os.makedirs(summary_home)
        logger.info('{} does not exists and created.'.format(summary_home))

    # loss
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels_masked, logits_masked)

    print('regularized weights:')
    print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = cross_entropy + tf.losses.get_regularization_loss()
    global_step = tf.Variable(1, False, name='global_step')
    trainer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(total_loss, global_step)

    # summary
    loss_summary = tf.summary.scalar('loss', total_loss)
    train_accuracy_summary = tf.summary.scalar('train_accuracy', train_accuracy[0], collections='TRAIN_SCORE')
    val_accuracy_summary = tf.summary.scalar('val_accuracy', val_accuracy[0], collections='VAL_SCORE')
    train_meaniou_summary = tf.summary.scalar('train_meaniou', train_meaniou[0], collections='TRAIN_SCORE')
    val_meaniou_summary = tf.summary.scalar('val_meaniou', val_meaniou[0], collections='VAL_SCORE')
    train_summary = tf.summary.merge([train_accuracy_summary, train_meaniou_summary])
    val_summary = tf.summary.merge([val_accuracy_summary, val_meaniou_summary])
    lr_summary = tf.summary.scalar('learning_rate', learning_rate)

    train_saver = tf.train.Saver()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(join(summary_home), sess.graph)
        exit(-1)
        sess.run(tf.global_variables_initializer())

        # restore the model and the progress of the training process.
        if os.path.exists(ckpt_home) and os.listdir(ckpt_home):
            model_ckpts = glob.glob(join(ckpt_home, 'model_*.ckpt.index'))
            last_iter = max(
                [int(re.search(r'model_(\d+)', each_ckpt_name).group(1)) for each_ckpt_name in model_ckpts])
            model_name = 'model_{}.ckpt'.format(last_iter)
            train_saver.restore(sess, join(ckpt_home, model_name))
            logger.info('{} exists and loaded successfully.'.format(model_name))
            last_iter += 1

        else:
            last_iter = 1

        train_it = d_train.shuffle(3000).repeat().batch(batch_size).make_one_shot_iterator().get_next()

        cur_loss = np.inf
        sess.run(tf.local_variables_initializer())
        for iteration in range(last_iter, nr_iter + last_iter):
            # X_train_next, y_train_next, y_train_mask_next, _ = sess.run(train_it)
            # Get next batch of data.
            next_batch = sess.run(train_it)
            fd = {model.X_input: next_batch[0], model.y_input: next_batch[1]}
            if model.ignore:
                fd[model.y_mask_input] = next_batch[2]
            cur_loss, _, _, _ = sess.run([total_loss, trainer, train_accuracy[1], train_meaniou[1]], feed_dict=fd)

            # train accuracy, mean IOU and summary for each epoch.
            if iteration % report_interval == 0:
                cur_loss_summary, cur_train_acc, cur_train_miou, cur_train_summary, cur_lr_summary = sess.run(
                    [loss_summary, train_accuracy[0], train_meaniou[0], train_summary, lr_summary], feed_dict=fd)
                logger.info(
                    'training {}/{}, accuracy: {}, mean IOU {}, loss: {}'.format(iteration, nr_iter, cur_train_acc,
                                                                                 cur_train_miou, cur_loss))
                summary_writer.add_summary(cur_train_summary, iteration)
                summary_writer.add_summary(cur_loss_summary, iteration)
                summary_writer.add_summary(cur_lr_summary, iteration)
                logger.info('train summary {} written.'.format(iteration))

                sess.run(tf.local_variables_initializer())

            if iteration % val_iter_interval == 0:
                logger.info('Reach iteration {} and start validating...'.format(iteration))
                val_batch_it = d_val.repeat(1).batch(batch_size).make_one_shot_iterator().get_next()

                while True:
                    try:
                        next_batch = sess.run(val_batch_it)
                        fd = {model.X_input: next_batch[0], model.y_input: next_batch[1]}
                        if model.ignore:
                            fd[model.y_mask_input] = next_batch[2]
                    except OutOfRangeError:
                        logger.info('Finish validation')
                        break

                    _, _ = sess.run([val_accuracy[1], val_meaniou[1]], feed_dict=fd)

                # val accuracy, mean IOU and summary for each epoch.
                cur_val_acc, cur_val_miou, cur_val_summary = sess.run([val_accuracy[0], val_meaniou[0], val_summary])
                logger.info('iteration {} val accuracy: {}, mean IOU {}'.format(iteration, cur_val_acc, cur_val_miou))
                summary_writer.add_summary(cur_val_summary, iteration)

                logger.info('val summary {} written.'.format(iteration))

            if iteration % iter_ckpt_interval == 0:
                train_saver.save(sess, join(ckpt_home, 'model_{}.ckpt'.format(iteration)))
                logger.info('model_{} saved.'.format(iteration))


class shuffler():
    def __init__(self, len):
        self.random_idx = list(range(len)) if len is not None else None

    def shuffle(self, *args):
        np.random.shuffle(self.random_idx)
        res = []
        for each_arg in args:
            res.append(each_arg[self.random_idx])

        return res

    def next_batch(self, batch_size, iter, *args):
        res = []
        for each_arg in args:
            if batch_size != 1:
                res.append(each_arg[batch_size * iter: batch_size * (iter + 1)])
            else:
                res.append(each_arg[batch_size * iter: batch_size * (iter + 1)][0][None])

        return res


def commission_predict(model, model_epoch, dump_home, X, y, mask, batch_size=1):
    mean_iou = tf.metrics.mean_iou(model.y_input, model.y_pred, model.output_class, model.y_mask_input,
                                   name='pred_mean_iou')
    accuracy = tf.metrics.accuracy(model.y_input, model.y_pred, model.y_mask_input, name='pred_accuracy')
    if dump_home:
        ckpt_home = join(dump_home, 'checkpoint')

    train_saver = tf.train.Saver()
    s = shuffler(None)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if dump_home and os.path.exists(ckpt_home) and os.listdir(ckpt_home):
            model_ckpts = glob.glob(join(ckpt_home, 'model_*.ckpt.index'))
            if model_epoch:
                last_epoch = model_epoch
            else:
                last_epoch = max(
                    [int(re.search(r'model_(\d+)', each_ckpt_name).group(1)) for each_ckpt_name in model_ckpts])
            model_name = 'model_{}.ckpt'.format(last_epoch)
            train_saver.restore(sess, join(ckpt_home, model_name))
            logger.info('{} exists and loaded successfully.'.format(model_name))

        res = []
        nr_iter = math.ceil(len(X) / batch_size)
        for iter in range(nr_iter):
            X_n, y_n, mask_n = s.next_batch(batch_size, iter, X, y, mask)
            logger.info('predicting {} / {}'.format(iter + 1, nr_iter))
            y_pred, _, _ = sess.run([model.y_pred, mean_iou[1], accuracy[1]],
                                    feed_dict={model.X_input: X_n, model.y_input: y_n, model.y_mask_input: mask_n})
            res.extend(y_pred)
        accuracy, mean_iou = sess.run([mean_iou[0], accuracy[0]])
        logger.info('prediction accuracy: {}\nprediction mean IoU: {}'.format(accuracy, mean_iou))
    return res, accuracy, mean_iou
