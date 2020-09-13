import logging

logging.basicConfig(level=logging.INFO)
import argparse
import os
import tensorflow as tf
from util import get_tensor_shape, apply_rf, get_l2_loss_with_mask
from dataset import get_train_dataset, RandDatasetReader
from linearization_net import Linearization_net

FLAGS = tf.app.flags.FLAGS
epsilon = 0.001

# ---

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--it_num', type=int, default=500000)  # 500k
parser.add_argument('--logdir_path', type=str, required=True)
parser.add_argument('--hdr_prefix', type=str, required=True)
ARGS = parser.parse_args()

# ---





# --- graph

_clip = lambda x: tf.clip_by_value(x, 0, 1)


def rand_quantize(
        img,  # [b, h, w, c]
        is_training,
):
    b, h, w, c, = get_tensor_shape(img)

    const_bit = tf.constant(8.0, tf.float32, [1, 1, 1, 1])

    bit = tf.cond(is_training, lambda: const_bit, lambda: const_bit)
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
        hdr,  # [b, h, w, c]
        crf,  # [b, k]
        invcrf,
        t,  # [b]
        is_training,
):

    b, h, w, c, = get_tensor_shape(hdr)
    b, k, = get_tensor_shape(crf)
    b, = get_tensor_shape(t)

    _hdr_t = hdr * tf.reshape(t, [b, 1, 1, 1])

    # Augment Poisson and Gaussian noise
    sigma_s = 0.08 / 6 * tf.random_uniform([tf.shape(_hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0,
                                                     dtype=tf.float32, seed=1)
    sigma_c = 0.005 * tf.random_uniform([tf.shape(_hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0, dtype=tf.float32, seed=1)
    noise_s_map = sigma_s * _hdr_t
    noise_s = tf.random_normal(shape=tf.shape(_hdr_t), seed=1) * noise_s_map
    temp_x = _hdr_t + noise_s
    noise_c = sigma_c * tf.random_normal(shape=tf.shape(_hdr_t), seed=1)
    temp_x = temp_x + noise_c
    _hdr_t = tf.nn.relu(temp_x)

    # Dynamic range clipping
    clipped_hdr_t = _clip(_hdr_t)

    # Camera response function
    ldr = apply_rf(clipped_hdr_t, crf)

    # Quantization and JPEG compression
    quantized_hdr = tf.round(ldr * 255.0)
    quantized_hdr_8bit = tf.cast(quantized_hdr, tf.uint8)
    jpeg_img_list = []
    for i in range(ARGS.batch_size):
        II = quantized_hdr_8bit[i]
        II = tf.image.adjust_jpeg_quality(II, int(round(float(i)/float(ARGS.batch_size-1)*10.0+90.0)))
        jpeg_img_list.append(II)
    jpeg_img = tf.stack(jpeg_img_list, 0)
    jpeg_img_float = tf.cast(jpeg_img, tf.float32) / 255.0
    jpeg_img_float.set_shape([None, 256, 256, 3])


    # loss mask to exclude over-/under-exposed regions
    gray = tf.image.rgb_to_grayscale(jpeg_img)
    over_exposed = tf.cast(tf.greater_equal(gray, 249), tf.float32)
    over_exposed = tf.reduce_sum(over_exposed, axis=[1, 2], keepdims=True)
    over_exposed = tf.greater(over_exposed, 256.0 * 256.0 * 0.5)
    under_exposed = tf.cast(tf.less_equal(gray, 6), tf.float32)
    under_exposed = tf.reduce_sum(under_exposed, axis=[1, 2], keepdims=True)
    under_exposed = tf.greater(under_exposed, 256.0 * 256.0 * 0.5)
    extreme_cases = tf.logical_or(over_exposed, under_exposed)
    loss_mask = tf.cast(tf.logical_not(extreme_cases), tf.float32)

    lin_net = Linearization_net()
    pred_invcrf = lin_net.get_output(ldr, is_training)
    pred_lin_ldr = apply_rf(ldr, pred_invcrf)
    crf_loss = tf.reduce_mean(tf.square(pred_invcrf - invcrf), axis=1, keepdims=True)
    loss = get_l2_loss_with_mask(pred_lin_ldr, clipped_hdr_t)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(tf.reduce_mean((loss+0.1*crf_loss)*loss_mask))

    mse = tf.reduce_mean((pred_lin_ldr - clipped_hdr_t) ** 2)
    psnr = 20.0 * log10(1.0) - 10.0 * log10(mse)
    mse = tf.reduce_mean((ldr - clipped_hdr_t) ** 2)
    psnr_no_q = 20.0 * log10(1.0) - 10.0 * log10(mse)

    tf.summary.scalar('loss', tf.reduce_mean(loss))
    tf.summary.scalar('crf_loss', tf.reduce_mean(crf_loss))
    tf.summary.image('pred_lin_ldr', pred_lin_ldr)
    tf.summary.image('ldr', ldr)
    tf.summary.image('clipped_hdr_t', clipped_hdr_t)
    tf.summary.scalar('loss mask 0', tf.squeeze(loss_mask[0]))
    tf.summary.scalar('loss mask 1', tf.squeeze(loss_mask[1]))
    tf.summary.scalar('loss mask 2', tf.squeeze(loss_mask[2]))

    return loss, train_op, psnr, psnr_no_q


b, h, w, c = ARGS.batch_size, 256, 256, 3

hdr = tf.placeholder(tf.float32, [None, None, None, c])
crf = tf.placeholder(tf.float32, [None, None])
invcrf = tf.placeholder(tf.float32, [None, None])
t = tf.placeholder(tf.float32, [None])
is_training = tf.placeholder(tf.bool)

loss, train_op, psnr, psnr_no_q = build_graph(hdr, crf, invcrf, t, is_training)
saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

# ---

total_parameters = 0
for variable in tf.trainable_variables():
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
print('total params: ')
print(total_parameters)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
lin_net.crf_feature_net.overwrite_init(sess)

summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(
    os.path.join(ARGS.logdir_path, 'summary'),
    sess.graph,
)
dataset_reader = RandDatasetReader(get_train_dataset(ARGS.hdr_prefix), b)

for it in range(ARGS.it_num):
    print(it)
    if it == 0 or it % 10000 == 9999:
        print('start save')
        checkpoint_path = os.path.join(ARGS.logdir_path, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=it)
        print('finish save')
    hdr_val, crf_val, invcrf_val, t_val = dataset_reader.read_batch_data()
    _, summary_val = sess.run([train_op, summary], {
        hdr: hdr_val,
        crf: crf_val,
        invcrf: invcrf_val,
        t: t_val,
        is_training: True,
    })
    if it == 0 or it % 10000 == 9999:
        summary_writer.add_summary(summary_val, it)
        logging.info('test')

