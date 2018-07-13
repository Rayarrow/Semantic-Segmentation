from models_front_end import *


def segmentation_model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    init_model_path = params['init_model_path']
    front_end = params['front_end'](features, params['image_height'], params['image_width'], get_FCN=1,
                                    is_training=is_training)
    model = params['model'](front_end, params['num_classes'], is_training=is_training)

    predictions = {
        'y_pred': model.y_pred,
        'y_prob': model.y_prob,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    if init_model_path:
        variables_to_restore = tf.train.list_variables(init_model_path)
        variables_to_restore.remove(tf.train.get_global_step())
        tf.train.init_from_checkpoint(init_model_path, {v.name.split(':')[0]: v for v in variables_to_restore})

    # tf.global_variables_initializer().run(session=tf.Session())

    valid_mask = tf.not_equal(labels, params['ignore_label'])
    valid_mask.set_shape([None, None, None])
    valid_logits = tf.boolean_mask(model.logits, valid_mask)
    valid_label = tf.boolean_mask(labels, valid_mask)
    valid_pred = tf.boolean_mask(model.y_pred, valid_mask)

    cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(valid_label, valid_logits)
    if params['weight_decay'] is not None:
        trainable_vars = [v for v in tf.trainable_variables()]
        loss = cross_entropy_loss + tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars]) * params['weight_decay']
    else:
        loss = cross_entropy_loss

    tf.identity(loss, name='loss')

    metrics = {
        'accuracy': tf.metrics.accuracy(valid_label, valid_pred),
        'miou': tf.metrics.mean_iou(valid_label, valid_pred, params['num_classes']),
    }

    miou = compute_mean_iou(metrics['miou'][1], params['num_classes'])

    tf.identity(metrics['accuracy'][1], name='acc')
    tf.identity(miou, name='miou')
    tf.identity(cross_entropy_loss, name='cross_entropy_loss')

    # print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    # exit(1)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    tf.summary.scalar('acc', metrics['accuracy'][1])
    tf.summary.scalar('miou', miou)

    # gt_decoded_labels = tf.py_func(decode_labels,
    #                                [labels, params['palette'], params['batch_size'], params['num_classes']], tf.uint8)

    # if mode == tf.estimator.ModeKeys.TRAIN:
    #     images = tf.cast(mean_addition(features), tf.uint8)
    #     tf.summary.image('images', tf.concat(axis=2, values=[images, gt_decoded_labels, pred_decoded_labels]),
    #                      max_outputs=12)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        if params['lr_decay'] == 'poly':
            lr = tf.train.polynomial_decay(params['learning_rate'], global_step, params['decay_step'],
                                           end_learning_rate=params['end_learning_rate'], power=params['power'],
                                           name='learning_rate')
        elif params['lr_decay'] == 'stable':
            lr = params['learning_rate']

        else:
            raise Exception('invalid lr decay policy.')

        tf.summary.scalar('lr', lr)

        # [print(v) for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        # exit(1)

        # with tf.variable_scope('', reuse=True):
        #     moving_means = [
        #         tf.get_variable('aspp/conv_1x1/BatchNorm/moving_mean'),
        #         tf.get_variable('aspp/conv_3x3_1/BatchNorm/moving_mean'),
        #         tf.get_variable('aspp/conv_3x3_2/BatchNorm/moving_mean'),
        #         tf.get_variable('aspp/conv_3x3_3/BatchNorm/moving_mean'),
        #         tf.get_variable('aspp/image_level_features/conv_1x1/BatchNorm/moving_mean'),
        #         tf.get_variable('aspp/conv_1x1_concat/BatchNorm/moving_mean')
        #     ]
        #
        #     moving_variances = [
        #         tf.get_variable('aspp/conv_1x1/BatchNorm/moving_variance'),
        #         tf.get_variable('aspp/conv_3x3_1/BatchNorm/moving_variance'),
        #         tf.get_variable('aspp/conv_3x3_2/BatchNorm/moving_variance'),
        #         tf.get_variable('aspp/conv_3x3_3/BatchNorm/moving_variance'),
        #         tf.get_variable('aspp/image_level_features/conv_1x1/BatchNorm/moving_variance'),
        #         tf.get_variable('aspp/conv_1x1_concat/BatchNorm/moving_variance')
        #     ]
        #
        # reduce_moving_mean = [tf.reduce_mean(each_mm) for each_mm in moving_means]
        # reduce_moving_variance = [tf.reduce_mean(each_mv) for each_mv in moving_variances]
        #
        # tf.identity(reduce_moving_mean, name='debug_moving_mean')
        # tf.identity(reduce_moving_variance, name='debug_moving_variance')
        #
        # tf.Print(reduce_moving_mean[0], reduce_moving_mean)
        # tf.Print(reduce_moving_variance[0], reduce_moving_variance)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(lr, params['momentum'])
            train_op = optimizer.minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def compute_mean_iou(total_cm, num_classes, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(num_classes):
        tf.identity(iou[i], name='train_iou_class{}'.format(i))
        tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(tf.greater(num_valid_entries, 0), tf.reduce_sum(iou, name=name) / num_valid_entries, 0)
    return result
