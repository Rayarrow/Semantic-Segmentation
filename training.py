import glob
import math
import os
import re
from os.path import join

import numpy as np
import tensorflow as tf

from config import *


def commission_training_task(model, dump_home, X_train, y_train, y_train_mask, X_val, y_val, y_val_mask, learning_rate,
                             momentum, nr_epoch, batch_size, val_size, max_nr_iter=999999, val_interval=10,
                             epoch_checkpoint=100):
    """
    :param dump_home: the place to write summary.
    :param val_size: specify the number of validation samples.
    :param max_nr_iter: specify the max number of iteration of each epoch. By default, the number of iterations is ceil(#{nr_training_data}/#{batch_size}). If `max_nr_iter` is less than the default number of iterations, an epoch will end in advance.
    :param val_interval: specify the number of epochs to
    :param epoch_checkpoint: specify the number of epochs to save the model.
    :return:
    """
    ckpt_home = join(dump_home, 'checkpoint')
    summary_home = join(dump_home, 'summary')

    # define metrics
    with tf.variable_scope('metrics'):
        train_accuracy = tf.metrics.accuracy(model.y_input, model.y_pred, model.y_mask_input, name='train_accuracy')
        val_accuracy = tf.metrics.accuracy(model.y_input, model.y_pred, model.y_mask_input, name='val_accuracy')

        train_meaniou = tf.metrics.mean_iou(model.y_input, model.y_pred, model.num_classes, model.y_mask_input,
                                            name='train_meaniou')
        val_meaniou = tf.metrics.mean_iou(model.y_input, model.y_pred, model.num_classes, model.y_mask_input,
                                          name='val_meaniou')

    # works the same way as the ops in tf.metrics.
    loss_container = tf.placeholder(tf.float32, shape=[None], name='loss_container')
    overall_loss = tf.reduce_mean(loss_container, name='loss_adder')

    val_size = min(val_size, len(X_val))
    if not os.path.exists(ckpt_home):
        os.makedirs(ckpt_home)
        logger.info('{} does not exists and created.'.format(ckpt_home))

    if not os.path.exists(summary_home):
        os.makedirs(summary_home)
        logger.info('{} does not exists and created.'.format(summary_home))

    # Deal with learning rate decay policy.
    if isinstance(learning_rate, float):
        learning_rate = [learning_rate, learning_rate]
        decay_interval = [0, nr_epoch]
    else:
        decay_interval = learning_rate[1]
        learning_rate = learning_rate[0]
    learning_rate, decay_interval = iter(learning_rate), iter(decay_interval)
    lr = next(learning_rate)
    di = next(decay_interval)

    learning_rate_holder = tf.placeholder(tf.float32, name='learning_rate')
    total_loss = model.cross_entropy + tf.losses.get_regularization_loss()
    trainer = tf.train.MomentumOptimizer(learning_rate_holder, momentum).minimize(total_loss)

    # summary
    loss_summary = tf.summary.scalar('loss', overall_loss)
    train_accuracy_summary = tf.summary.scalar('train_accuracy', train_accuracy[0], collections='TRAIN_SCORE')
    val_accuracy_summary = tf.summary.scalar('val_accuracy', val_accuracy[0], collections='VAL_SCORE')
    train_meaniou_summary = tf.summary.scalar('train_meaniou', train_meaniou[0], collections='TRAIN_SCORE')
    val_meaniou_summary = tf.summary.scalar('val_meaniou', val_meaniou[0], collections='VAL_SCORE')
    train_summary = tf.summary.merge([train_accuracy_summary, train_meaniou_summary])
    val_summary = tf.summary.merge([val_accuracy_summary, val_meaniou_summary])
    lr_summary = tf.summary.scalar('learning_rate', learning_rate_holder)

    train_saver = tf.train.Saver()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(join(summary_home), sess.graph)
        sess.run(tf.global_variables_initializer())

        # restore the model and the progress of the training process.
        if os.path.exists(ckpt_home) and os.listdir(ckpt_home):
            model_ckpts = glob.glob(join(ckpt_home, 'model_*.ckpt.index'))
            last_epoch = max(
                [int(re.search(r'model_(\d+)', each_ckpt_name).group(1)) for each_ckpt_name in model_ckpts])
            model_name = 'model_{}.ckpt'.format(last_epoch)
            train_saver.restore(sess, join(ckpt_home, model_name))
            logger.info('{} exists and loaded successfully.'.format(model_name))
            last_epoch += 1

        else:
            last_epoch = 1

        train_shuffler = shuffler(len(X_train))
        val_shuffler = shuffler(len(X_val))
        for epoch in range(last_epoch, nr_epoch + 1):
            if epoch == di:
                lr = next(learning_rate)
                di = next(decay_interval)

            sess.run(tf.local_variables_initializer())
            loss_each_iter = []
            X_train, y_train, y_train_mask = train_shuffler.shuffle(X_train, y_train, y_train_mask)
            for i in range(min(max_nr_iter, math.ceil(len(X_train) / batch_size))):
                X_train_next, y_train_next, y_train_mask_next = train_shuffler.next_batch(batch_size, i, X_train,
                                                                                          y_train, y_train_mask)

                _, cur_loss, _, _ = sess.run([trainer, total_loss, train_accuracy[1], train_meaniou[1]],
                                             feed_dict={model.X_input: X_train_next, model.y_input: y_train_next,
                                                        model.y_mask_input: y_train_mask_next,
                                                        learning_rate_holder: lr})
                loss_each_iter.append(cur_loss)

            # train accuracy, mean IOU and summary for each epoch.
            cur_loss, cur_loss_summary, cur_train_acc, cur_train_miou, cur_train_summary, cur_lr_summary = sess.run(
                [overall_loss, loss_summary, train_accuracy[0], train_meaniou[0], train_summary, lr_summary],
                feed_dict={loss_container: loss_each_iter, learning_rate_holder: lr})
            # sess.run(
            #     [train_accuracy[0], train_meaniou[0]])

            logger.info('training {}/{}, accuracy: {}, mean IOU {}, loss: {}'.format(epoch, nr_epoch, cur_train_acc,
                                                                                     cur_train_miou, cur_loss))
            summary_writer.add_summary(cur_train_summary, epoch)
            summary_writer.add_summary(cur_loss_summary, epoch)
            summary_writer.add_summary(cur_lr_summary, epoch)
            logger.info('train summary {} written.'.format(epoch))

            if epoch % val_interval == 0:
                val_shuffler.shuffle(X_val, y_val, y_val_mask)
                for i in range(math.ceil(val_size / batch_size)):
                    X_val_next, y_val_next, y_val_mask_next = val_shuffler.next_batch(batch_size, i, X_val, y_val,
                                                                                      y_val_mask)
                    _, _ = sess.run([val_accuracy[1], val_meaniou[1]],
                                    feed_dict={model.X_input: X_val_next, model.y_input: y_val_next,
                                               model.y_mask_input: y_val_mask_next})

                # val accuracy, mean IOU and summary for each epoch.
                cur_val_acc, cur_val_miou, cur_val_summary = sess.run([val_accuracy[0], val_meaniou[0], val_summary])
                logger.info('epoch {} val accuracy: {}, mean IOU {}'.format(epoch, cur_val_acc, cur_val_miou))
                summary_writer.add_summary(cur_val_summary, epoch)

                logger.info('val summary {} written.'.format(epoch))

            if epoch % epoch_checkpoint == 0:
                train_saver.save(sess, join(ckpt_home, 'model_{}.ckpt'.format(epoch)))
                logger.info('model_{} saved.'.format(epoch))

            # if epoch % 10 == 0:
            #     val_idx = np.random.randint(0, len(X_val), val_size)
            #
            #     cur_accuracy = sess.run(model.accuracy,
            #                             feed_dict={model.X_input: X_val[val_idx], model.y_input: y_val[val_idx],
            #                                        model.y_mask_input: y_val_mask[val_idx]})
            #
            #     logger.info('epoch {} val accuracy: {}'.format(epoch, cur_accuracy))


class shuffler():
    def __init__(self, len):
        self.random_idx = list(range(len))

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


def commission_predict(model, model_epoch, dump_home, X, batch_size=1):
    # mean_iou = tf.metrics.mean_iou(model.y_input, model.y_pred, model.output_class, model.y_mask_input, name='mean_iou')
    # accuracy =
    if dump_home:
        ckpt_home = join(dump_home, 'checkpoint')

    train_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
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
        if batch_size != 1:
            for iter in range(nr_iter):
                logger.info('predicting {} / {}'.format(iter + 1, nr_iter))
                res.extend(
                    sess.run(model.y_pred, feed_dict={model.X_input: X[iter * batch_size: (iter + 1) * batch_size]}))
        else:
            for iter in range(nr_iter):
                logger.info('predicting {} / {}'.format(iter + 1, nr_iter))
                res.extend(sess.run(model.y_pred, feed_dict={model.X_input: X[iter:iter + 1][0][None]}))
    return res
