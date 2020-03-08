import tensorflow as tf

class Dequantization_net(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def inference(self, input_images):
        """Inference on a set of input_images.
        Args:
        """
        return self._build_model(input_images)

    def loss(self, predictions, targets):
        """Compute the necessary loss for training.
        Args:
        Returns:
        """
        return tf.reduce_mean(tf.square(predictions - targets))

    def down(self, x, outChannels, filterSize):
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, filterSize, 1, 'same'), 0.1)
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, filterSize, 1, 'same'), 0.1)
        return x

    def up(self, x, outChannels, skpCn):
        x = tf.image.resize_bilinear(x, 2*tf.shape(x)[1:3])
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, 3, 1, 'same'), 0.1)
        x = tf.nn.leaky_relu(tf.layers.conv2d(tf.concat([x, skpCn], -1), outChannels, 3, 1, 'same'), 0.1)
        return x

    def _build_model(self, input_images):
        print(input_images.get_shape().as_list())
        x = tf.nn.leaky_relu(tf.layers.conv2d(input_images, 16, 7, 1, 'same'), 0.1)
        s1 = tf.nn.leaky_relu(tf.layers.conv2d(x, 16, 7, 1, 'same'), 0.1)
        s2 = self.down(s1, 32, 5)
        s3 = self.down(s2, 64, 3)
        s4 = self.down(s3, 128, 3)
        x = self.down(s4, 256, 3)
        # x = self.down(s5, 512, 3)
        # x = self.up(x, 512, s5)
        x = self.up(x, 128, s4)
        x = self.up(x, 64, s3)
        x = self.up(x, 32, s2)
        x = self.up(x, 16, s1)
        x = tf.nn.tanh(tf.layers.conv2d(x, 3, 3, 1, 'same'))
        output = input_images + x
        return output