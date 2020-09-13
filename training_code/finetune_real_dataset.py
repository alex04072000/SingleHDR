import logging

logging.basicConfig(level=logging.INFO)
import argparse
import tensorflow as tf
from util import get_tensor_shape, apply_rf
import os
import glob
from random import shuffle
from dequantization_net import Dequantization_net
from linearization_net import Linearization_net
import hallucination_net
from refinement_net import Refinement_net

FLAGS = tf.app.flags.FLAGS
epsilon = 0.001


# ---

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--it_num', type=int, default=100000)  # 500k
parser.add_argument('--logdir_path', type=str, required=True)
parser.add_argument('--tfrecords_path', type=str, required=True)
parser.add_argument('--deq_ckpt', type=str, required=True)
parser.add_argument('--lin_ckpt', type=str, required=True)
parser.add_argument('--hal_ckpt', type=str, required=True)
ARGS = parser.parse_args()


def load_data(filename_queue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'ref_HDR': tf.FixedLenFeature([], tf.string),
            'ref_LDR': tf.FixedLenFeature([], tf.string),
        })

    ref_HDR = tf.decode_raw(img_features['ref_HDR'], tf.float32)
    ref_LDR = tf.decode_raw(img_features['ref_LDR'], tf.float32)
    ref_HDR = tf.reshape(ref_HDR, [256, 256, 3])
    ref_LDR = tf.reshape(ref_LDR, [256, 256, 3])

    ref_HDR = ref_HDR / (1e-6 + tf.reduce_mean(ref_HDR)) * 0.5
    ref_LDR = ref_LDR / 255.0

    distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)

    # flip horizontally
    ref_HDR = tf.cond(tf.less(distortions[0], 0.5), lambda: tf.image.flip_left_right(ref_HDR), lambda: ref_HDR)
    ref_LDR = tf.cond(tf.less(distortions[0], 0.5), lambda: tf.image.flip_left_right(ref_LDR), lambda: ref_LDR)

    # rotate
    k = tf.cast(distortions[1] * 4 + 0.5, tf.int32)
    ref_HDR = tf.image.rot90(ref_HDR, k)
    ref_LDR = tf.image.rot90(ref_LDR, k)

    # TODO: channel swapping?

    ref_HDR_batch, ref_LDR_batch = tf.train.shuffle_batch(
        [ref_HDR, ref_LDR],
        batch_size=8,
        num_threads=8,
        capacity=256,
        min_after_dequeue=64)

    return ref_HDR_batch, ref_LDR_batch

tfrecord_list = glob.glob(os.path.join(ARGS.tfrecords_path, '*.tfrecords'), recursive=True)
print(len(tfrecord_list))
assert (tfrecord_list)
shuffle(tfrecord_list)
print('\n\n====================\ntfrecords list:')
[print(f) for f in tfrecord_list]
print('====================\n\n')

with tf.device('/cpu:0'):
    filename_queue = tf.train.string_input_producer(tfrecord_list)
    ref_HDR_batch, ref_LDR_batch = load_data(filename_queue)




# ---


# --- graph

_clip = lambda x: tf.clip_by_value(x, 0, 1)


def fix_quantize(
        img,  # [b, h, w, c]
        is_training,
):
    b, h, w, c, = get_tensor_shape(img)

    const_bit = tf.constant(8.0, tf.float32, [1, 1, 1, 1])

    bit = const_bit
    s = (2 ** bit) - 1

    img = _clip(img)
    img = tf.round(s * img) / s
    img = _clip(img)

    return img


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def build_graph(
        ldr,  # [b, h, w, c]
        hdr,  # [b, h, w, c]
        is_training,
):
    b, h, w, c, = get_tensor_shape(ldr)
    b, h, w, c, = get_tensor_shape(hdr)

    # Dequantization-Net
    with tf.variable_scope("Dequantization_Net"):
        dequantization_model = Dequantization_net(is_train=True)
        C_pred = _clip(dequantization_model.inference(ldr))

    # Linearization-Net
    lin_net = Linearization_net()
    pred_invcrf = lin_net.get_output(C_pred, True)
    B_pred = apply_rf(C_pred, pred_invcrf)

    # Hallucination-Net
    thr = 0.12
    alpha = tf.reduce_max(B_pred, reduction_indices=[3])
    alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
    alpha = tf.reshape(alpha, [-1, tf.shape(B_pred)[1], tf.shape(B_pred)[2], 1])
    alpha = tf.tile(alpha, [1, 1, 1, 3])
    with tf.variable_scope("Hallucination_Net"):
        net_test, vgg16_conv_layers_test = hallucination_net.model(B_pred, ARGS.batch_size, True)
        y_predict_test = net_test.outputs
        y_predict_test = tf.nn.relu(y_predict_test)
        A_pred = (B_pred) + alpha * y_predict_test

    t_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
    print('all layers:')
    for var in t_vars: print(var.name)

    # Refinement-Net
    with tf.variable_scope("Refinement_Net"):
        refinement_model = Refinement_net(is_train=is_training)
        refinement_output = tf.nn.relu(refinement_model.inference(tf.concat([A_pred, B_pred, C_pred], -1)))

    _log = lambda x: tf.log(x + 1.0 / 255.0)

    refinement_output = refinement_output / (1e-6 + tf.reduce_mean(refinement_output, axis=[1, 2, 3], keepdims=True)) * 0.5
    refinement_output_gamma = tf.log(1.0 + 10.0 * refinement_output) / tf.log(1.0 + 10.0)
    hdr_gamma = tf.log(1.0 + 10.0 * hdr) / tf.log(1.0 + 10.0)
    loss = tf.reduce_mean(tf.abs(refinement_output_gamma - hdr_gamma))


    """compare with DLH output (without refinement net)"""
    A_pred_output = A_pred / (1e-6 + tf.reduce_mean(A_pred, axis=[1, 2, 3], keepdims=True)) * 0.5
    A_pred_output_gamma = tf.log(1.0 + 10.0 * A_pred_output) / tf.log(1.0 + 10.0)
    A_pred_loss = tf.reduce_mean(tf.abs(A_pred_output_gamma - hdr_gamma))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)

    tf.summary.scalar('loss', tf.reduce_mean(loss))
    tf.summary.scalar('A_pred_loss', tf.reduce_mean(A_pred_loss))
    tf.summary.image('hdr', hdr)
    tf.summary.image('ldr', ldr)
    tf.summary.image('C_pred', C_pred)
    tf.summary.image('B_pred', B_pred)
    tf.summary.image('A_pred', A_pred)
    tf.summary.image('refinement_output', refinement_output)
    tf.summary.histogram('hdr_histo', hdr)
    tf.summary.histogram('refinement_output_histo', refinement_output)

    return train_op, loss


b, h, w, c = ARGS.batch_size, 512, 512, 3

is_training = tf.placeholder(tf.bool)

train_op, loss = build_graph(ref_LDR_batch, ref_HDR_batch, is_training)
saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)



# ---

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord=coord)
sess.run(tf.global_variables_initializer())

total_parameters = 0
for variable in tf.trainable_variables():
    if 'Refinement_Net' in variable.name:
    # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
print(total_parameters)

restorer1 = tf.train.Saver(
    var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Dequantization_Net' in var.name])
restorer1.restore(sess, ARGS.deq_ckpt)

restorer2 = tf.train.Saver(
    var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'crf_feature_net' in var.name or 'ae_invcrf_' in var.name])
restorer2.restore(sess, ARGS.lin_ckpt)

restorer3 = tf.train.Saver(
    var_list=[var for var in tf.get_collection(tf.GraphKeys.VARIABLES) if 'Hallucination_Net' in var.name])
restorer3.restore(sess, ARGS.hal_ckpt)



# hallucination_net.load_vgg_weights(vgg16_conv_layers, 'vgg16_places365_weights.npy', sess)

summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(
    os.path.join(ARGS.logdir_path, 'summary'),
    sess.graph,
)



for it in range(ARGS.it_num):
    print(it)
    if it == 0 or it % 10000 == 9999:
        print('start save')
        checkpoint_path = os.path.join(ARGS.logdir_path, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=it)
        # tl.files.save_npz(net.all_params, name=os.path.join(ARGS.logdir_path, 'model'+str(it)+'.npz'), sess=sess)
        print('finish save')
    _, summary_val = sess.run([train_op, summary], {is_training: True})
    if it == 0 or it % 10000 == 9999:
        summary_writer.add_summary(summary_val, it)
        logging.info('test')


coord.request_stop()
coord.join(threads)
