import logging
import inspect
import time

logging.basicConfig(level=logging.INFO)
import argparse
import os
import tensorflow as tf
import hallucination_net
from util import get_tensor_shape, apply_rf, get_l2_loss
from dataset import get_train_dataset, RandDatasetReader
import numpy as np

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

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

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

def get_final(network, x_in):
    sb, sy, sx, sf = x_in.get_shape().as_list()
    y_predict = network.outputs

    # Highlight mask
    thr = 0.05
    alpha = tf.reduce_max(x_in, reduction_indices=[3])
    alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
    alpha = tf.reshape(alpha, [-1, sy, sx, 1])
    alpha = tf.tile(alpha, [1, 1, 1, 3])

    # Linearied input and prediction
    x_lin = tf.pow(x_in, 2.0)
    y_predict = tf.exp(y_predict) - 1.0 / 255.0

    # Alpha blending
    y_final = (1 - alpha) * x_lin + alpha * y_predict

    return y_final


def build_graph(
        hdr,  # [b, h, w, c]
        crf,  # [b, k]
        t,  # [b]
        is_training,
):
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

    # loss mask
    ldr = apply_rf(clipped_hdr_t, crf)
    quantized_hdr = tf.round(ldr * 255.0)
    quantized_hdr_8bit = tf.cast(quantized_hdr, tf.uint8)
    jpeg_img_list = []
    for i in range(ARGS.batch_size):
        II = quantized_hdr_8bit[i]
        II = tf.image.adjust_jpeg_quality(II, int(round(float(i) / float(ARGS.batch_size - 1) * 10.0 + 90.0)))
        jpeg_img_list.append(II)
    jpeg_img = tf.stack(jpeg_img_list, 0)
    jpeg_img_float = tf.cast(jpeg_img, tf.float32) / 255.0
    jpeg_img_float.set_shape([None, 256, 256, 3])
    gray = tf.image.rgb_to_grayscale(jpeg_img)
    over_exposed = tf.cast(tf.greater_equal(gray, 249), tf.float32)
    over_exposed = tf.reduce_sum(over_exposed, axis=[1, 2], keepdims=True)
    over_exposed = tf.greater(over_exposed, 256.0 * 256.0 * 0.5)
    under_exposed = tf.cast(tf.less_equal(gray, 6), tf.float32)
    under_exposed = tf.reduce_sum(under_exposed, axis=[1, 2], keepdims=True)
    under_exposed = tf.greater(under_exposed, 256.0 * 256.0 * 0.5)
    extreme_cases = tf.logical_or(over_exposed, under_exposed)
    loss_mask = tf.cast(tf.logical_not(extreme_cases), tf.float32)

    # Highlight mask
    thr = 0.12
    alpha = tf.reduce_max(clipped_hdr_t, reduction_indices=[3])
    alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
    alpha = tf.reshape(alpha, [-1, tf.shape(clipped_hdr_t)[1], tf.shape(clipped_hdr_t)[2], 1])
    alpha = tf.tile(alpha, [1, 1, 1, 3])

    with tf.variable_scope("Hallucination_Net"):
        net, vgg16_conv_layers = hallucination_net.model(clipped_hdr_t, ARGS.batch_size, True)
        y_predict = tf.nn.relu(net.outputs)
        y_final = (clipped_hdr_t) + alpha * y_predict # residual

    with tf.variable_scope("Hallucination_Net", reuse=True):
        net_test, vgg16_conv_layers_test = hallucination_net.model(clipped_hdr_t, ARGS.batch_size, False)
        y_predict_test = tf.nn.relu(net_test.outputs)
        y_final_test = (clipped_hdr_t) + alpha * y_predict_test # residual


    _log = lambda x: tf.log(x + 1.0/255.0)

    vgg = Vgg16('vgg16.npy')
    vgg.build(tf.log(1.0+10.0*y_final)/tf.log(1.0+10.0))
    vgg2 = Vgg16('vgg16.npy')
    vgg2.build(tf.log(1.0+10.0*_hdr_t)/tf.log(1.0+10.0))
    perceptual_loss = tf.reduce_mean(tf.abs((vgg.pool1 - vgg2.pool1)), axis=[1, 2, 3], keepdims=True)
    perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool2 - vgg2.pool2)), axis=[1, 2, 3], keepdims=True)
    perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool3 - vgg2.pool3)), axis=[1, 2, 3], keepdims=True)

    loss_test = get_l2_loss(_log(y_final_test), _log(_hdr_t))

    y_final_gamma = tf.log(1.0+10.0*y_final)/tf.log(1.0+10.0)
    _hdr_t_gamma = tf.log(1.0+10.0*_hdr_t)/tf.log(1.0+10.0)

    loss = tf.reduce_mean(tf.abs(y_final_gamma - _hdr_t_gamma), axis=[1, 2, 3], keepdims=True)
    y_final_gamma_pad_x = tf.pad(y_final_gamma, [[0, 0], [0, 1], [0, 0], [0, 0]], 'SYMMETRIC')
    y_final_gamma_pad_y = tf.pad(y_final_gamma, [[0, 0], [0, 0], [0, 1], [0, 0]], 'SYMMETRIC')
    tv_loss_x = tf.reduce_mean(tf.abs(y_final_gamma_pad_x[:, 1:] - y_final_gamma_pad_x[:, :-1]))
    tv_loss_y = tf.reduce_mean(tf.abs(y_final_gamma_pad_y[:, :, 1:] - y_final_gamma_pad_y[:, :, :-1]))
    tv_loss = tv_loss_x + tv_loss_y

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(
            tf.reduce_mean((loss + 0.001 * perceptual_loss + 0.1 * tv_loss)*loss_mask))

    t_vars = tf.trainable_variables()
    print('all layers:')
    for var in t_vars: print(var.name)

    tf.summary.scalar('loss', tf.reduce_mean(loss))
    tf.summary.image('hdr_t', _hdr_t)
    tf.summary.image('y', y_final)
    tf.summary.image('clipped_hdr_t', clipped_hdr_t)
    tf.summary.image('alpha', alpha)
    tf.summary.image('y_predict', y_predict)
    tf.summary.image('log_hdr_t', tf.log(1.0+10.0*_hdr_t)/tf.log(1.0+10.0))
    tf.summary.image('log_y', tf.log(1.0+10.0*y_final)/tf.log(1.0+10.0))
    tf.summary.image('log_clipped_hdr_t', tf.log(1.0+10.0*clipped_hdr_t)/tf.log(1.0+10.0))


    psnr = tf.zeros([])
    psnr_no_q = tf.zeros([])

    return loss, train_op, psnr, psnr_no_q, loss_test, vgg16_conv_layers, net, y_final_test, y_predict_test, alpha


b, h, w, c = ARGS.batch_size, 512, 512, 3

hdr = tf.placeholder(tf.float32, [None, None, None, c])
crf = tf.placeholder(tf.float32, [None, None])
t = tf.placeholder(tf.float32, [None])
is_training = tf.placeholder(tf.bool)

loss, train_op, psnr, psnr_no_q, loss_test, vgg16_conv_layers, net, y_final_test, y_predict_test, alpha = build_graph(hdr, crf, t, is_training)
saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

# ---

sess = tf.Session()
sess.run(tf.global_variables_initializer())
hallucination_net.load_vgg_weights(vgg16_conv_layers, 'vgg16_places365_weights.npy', sess)

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
        print(net.all_params)
        # tl.files.save_npz(net.all_params, name=os.path.join(ARGS.logdir_path, 'model'+str(it)+'.npz'), sess=sess)
        print('finish save')
    hdr_val, crf_val, invcrf_val, t_val = dataset_reader.read_batch_data()
    _, summary_val = sess.run([train_op, summary], {
        hdr: hdr_val,
        crf: crf_val,
        t: t_val,
        is_training: True,
    })
    if it == 0 or it % 10000 == 9999:
        summary_writer.add_summary(summary_val, it)
        logging.info('test')

