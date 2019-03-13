from model_encoders import *


def segmentation_model_fn(features, labels, mode, params):
    print(2)
    is_training = mode == tf.estimator.ModeKeys.TRAIN and not params['frozen']
    init_model_path = params['init_model_path']
    logger.info(f'!!! is_training: {is_training}')

    if params['structure_mode'] == 'seg' or params['structure_mode'] == 'sup':
        if params['structure_mode'] == 'sup':
            feature = tf.concat([features[0], features[1]], axis=-1)
        else:
            feature = features

        encoder = params['encoder'](feature, params['image_height'], params['image_width'], get_FCN=1,
                                    is_training=is_training)
        model = params['decoder'](encoder, params['num_classes'], is_training=is_training)
        logits = model.logits

    elif params['structure_mode'] == 'siamese':
        with tf.variable_scope('') as scope:
            encoder1 = params['encoder'](features[0], params['image_height'], params['image_width'], get_FCN=1,
                                         is_training=is_training)
            model_1 = params['decoder'](encoder1, params['num_classes'], is_training=is_training)
            scope.reuse_variables()
            encoder2 = params['encoder'](features[1], params['image_height'], params['image_width'], get_FCN=1,
                                         is_training=is_training)
            model_2 = params['decoder'](encoder2, params['num_classes'], is_training=is_training)

            logits = tf.abs(model_1.logits - model_2.logits)

    else:
        raise Exception('invalid structure_mode.')

    y_pred = tf.argmax(logits, axis=-1, output_type=tf.int32, name='y_pred')
    y_prob = tf.nn.softmax(logits)

    predictions = {
        'y_pred': y_pred,
        'y_prob': y_prob,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    valid_mask = tf.not_equal(labels, params['ignore_label'])
    valid_mask.set_shape([None, None, None])
    valid_logits = tf.boolean_mask(logits, valid_mask)
    valid_label = tf.boolean_mask(labels, valid_mask)
    valid_pred = tf.boolean_mask(y_pred, valid_mask)

    cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(valid_label, valid_logits)

    if init_model_path:
        logger.info(f'Manually restoring from pre-trained network{init_model_path}...')
        variables_in_ckpt = set(e[0] for e in tf.train.list_variables(init_model_path))
        if 'global_step' in variables_in_ckpt:
            variables_in_ckpt.remove('global_step')

        variables_to_restore = {}
        for each_variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if each_variable.name in variables_in_ckpt:
                variables_to_restore[each_variable.name] = each_variable

        tf.train.init_from_checkpoint(init_model_path, variables_to_restore)
        logger.info(f'{len(variables_to_restore)} variables restored successfully.')

    # tf.global_variables_initializer().run(session=tf.Session())

    if params['weight_decay'] is not None:
        trainable_vars = [v for v in tf.trainable_variables()]
        loss = cross_entropy_loss + tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars]) * params['weight_decay']
    else:
        loss = cross_entropy_loss

    metrics = {
        'accuracy': tf.metrics.accuracy(valid_label, valid_pred),
        'mean_iou': tf.metrics.mean_iou(valid_label, valid_pred, params['num_classes']),
    }

    miou = compute_mean_iou(metrics['mean_iou'][1], params['num_classes'])
    f1 = compute_mean_iou(metrics['mean_iou'][1], params['num_classes'], name='f1')

    tf.identity(loss, name='loss')
    tf.identity(cross_entropy_loss, name='cross_entropy_loss')
    tf.identity(metrics['accuracy'][1], name='acc')
    tf.identity(miou, name='miou')
    tf.identity(f1, name='f1')

    # print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    # exit(1)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    tf.summary.scalar('acc', metrics['accuracy'][1])
    tf.summary.scalar('miou', miou)
    tf.summary.scalar('f1', f1)

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
    if name == 'f1':
        denominator += cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(num_classes):
        tf.identity(iou[i], name=f'train_{name}_class{i}')
        tf.summary.scalar(f'train_{name}_class{i}', iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(tf.greater(num_valid_entries, 0), tf.reduce_sum(iou, name=name) / num_valid_entries, 0)
    if name == 'f1':
        result *= 2
    return result
