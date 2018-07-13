import tensorflow as tf
import os

ckpt_dir = '/Volumes/Transcend/summary/SRes101@16_Deeplabv3#trainaug#50#8#0.007#1e-06#66150#513#debug_print'
to_dir = '/Volumes/Transcend/summary/rename/model.ckpt-66150'

if not os.path.exists(to_dir):
    os.makedirs(to_dir)

rename_mapping = {
    'reduce/biases': 'upsampling_logits/conv_1x1/biases',
    'reduce/biases/Momentum': 'upsampling_logits/conv_1x1/biases/Momentum',
    'reduce/weights': 'upsampling_logits/conv_1x1/weights',
    'reduce/weights/Momentum': 'upsampling_logits/conv_1x1/weights/Momentum',
}


def change_ckpt_var_name_and_save(ckpt_dir, to_dir, rename_mapping):
    for var_name, shape in tf.train.list_variables(ckpt_dir):
        print(f'creating variable {var_name}')
        var = tf.train.load_variable(ckpt_dir, var_name)
        if var_name in rename_mapping:
            var_name = rename_mapping[var_name]

        tf.Variable(var, name=var_name)

    print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, to_dir)


change_ckpt_var_name_and_save(ckpt_dir, to_dir, rename_mapping)
