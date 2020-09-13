import tensorflow as tf


def get_tensor_shape(x):
    a = x.get_shape().as_list()
    b = [tf.shape(x)[i] for i in range(len(a))]
    def _select_one(aa, bb):
        if type(aa) is int:
            return aa
        else:
            return bb
    return [_select_one(aa, bb) for aa, bb in zip(a, b)]


def pass_net_nx(func, in_img, n):
    b, h, w, c, = get_tensor_shape(in_img)
    def _get_nx(x):
        s, r = x//n, x%n
        s = tf.cond(
            tf.equal(r, 0),
            lambda: s,
            lambda: s + 1,
        )
        return n*s
    nx_h = _get_nx(h)
    nx_w = _get_nx(w)
    def _get_rl_rr(x, nx):
        r = nx - x
        rl = r//2
        rr = r - rl
        return rl, rr
    in_img = tf.pad(in_img, [[0, 0], _get_rl_rr(h, nx_h), _get_rl_rr(w, nx_w), [0, 0]], mode='symmetric')
    in_img = tf.reshape(in_img, [b, nx_h, nx_w, c])
    out_img = func(in_img)
    out_img = tf.image.resize_image_with_crop_or_pad(out_img, h, w)
    return out_img


def sample_1d(
    img,   # [b, h, c]
    y_idx, # [b, n], 0 <= pos < h, dtpye=int32
):
    b, h, c = get_tensor_shape(img)
    b, n    = get_tensor_shape(y_idx)
    
    b_idx = tf.range(b, dtype=tf.int32) # [b]
    b_idx = tf.expand_dims(b_idx, -1)   # [b, 1]
    b_idx = tf.tile(b_idx, [1, n])      # [b, n]
    
    y_idx = tf.clip_by_value(y_idx, 0, h - 1) # [b, n]
    a_idx = tf.stack([b_idx, y_idx], axis=-1) # [b, n, 2]
    
    return tf.gather_nd(img, a_idx)


def interp_1d(
    img, # [b, h, c]
    y,   # [b, n], 0 <= pos < h, dtype=float32
):
    b, h, c = get_tensor_shape(img)
    b, n    = get_tensor_shape(y)
    
    y_0 = tf.floor(y) # [b, n]
    y_1 = y_0 + 1    
    
    _sample_func = lambda y_x: sample_1d(
        img,
        tf.cast(y_x, tf.int32)
    )
    y_0_val = _sample_func(y_0) # [b, n, c]
    y_1_val = _sample_func(y_1)
    
    w_0 = y_1 - y # [b, n]
    w_1 = y - y_0
    
    w_0 = tf.expand_dims(w_0, -1) # [b, n, 1]
    w_1 = tf.expand_dims(w_1, -1)
    
    return w_0*y_0_val + w_1*y_1_val


def apply_rf(
    x,  # [b, s...]
    rf, # [b, k]
):
    b, *s, = get_tensor_shape(x)
    b, k,  = get_tensor_shape(rf)
    x = interp_1d(
        tf.expand_dims(rf, -1),                              # [b, k, 1] 
        tf.cast((k - 1), tf.float32)*tf.reshape(x, [b, -1]), # [b, ?] 
    ) # [b, ?, 1]
    return tf.reshape(x, [b] + s)


def get_l2_loss(a, b):
    return tf.reduce_mean(tf.square(a - b))

def get_l2_loss_with_mask(a, b):
    return tf.reduce_mean(tf.square(a - b), axis=[1, 2, 3], keepdims=True)


def quantize(
    img, # [b, h, w, c]
    s=255, 
):
    _clip = lambda x: tf.clip_by_value(x, 0, 1)
    img = _clip(img)
    img = tf.round(s*img)/s
    img = _clip(img)
    return img


def rand_quantize(
    img, # [b, h, w, c]
    is_training,
):
    b, h, w, c, = get_tensor_shape(img)
    
    rand_bit  = tf.cast(tf.random_uniform([b, 1, 1, 1], minval=8, maxval=12, dtype=tf.int32), tf.float32)
    const_bit = tf.constant(8.0, tf.float32, [b, 1, 1, 1])
    
    bit = tf.cond(is_training, lambda: rand_bit, lambda: const_bit)
    s   = (2**bit) - 1
    
    return quantize(img, s)

