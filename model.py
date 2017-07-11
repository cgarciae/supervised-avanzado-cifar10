from tfinterface.supervised import SoftmaxClassifier
import tensorflow as tf
import tfinterface as ti

n_classes = 10

class Model(SoftmaxClassifier):

    def __init__(self, *args, **kwargs):

        self._learning_rates = kwargs.pop("learning_rates", [0.01, 0.001, 0.0003])
        self._boundaries = kwargs.pop("boundaries", [18000, 36000])

        # rotation
        self._rotation_angle = kwargs.pop("rotation_angle", 15.0)

        # model
        self._activation = kwargs.pop("activation", tf.nn.elu)
        self._dropout_rate = kwargs.pop("dropout_rate", 0.2)

        # densenet
        self._growth_rate = kwargs.pop("growth_rate", 12)
        self._compression = kwargs.pop("compression", 0.5)
        self._bottleneck = kwargs.pop("bottleneck", 4 * self._growth_rate)
        self._depth = kwargs.pop("depth", 100)
        self._depth -= 4
        self._n_layers = (
            self._depth / 6 if self._bottleneck else
            self._depth / 3
        )

        super(Model, self).__init__(*args, **kwargs)

    def get_labels(self, inputs):
        # one hot labels
        return tf.one_hot(inputs.labels, n_classes)

    def get_learning_rate(self, inputs):
        return tf.train.piecewise_constant(
            inputs.global_step,
            self._boundaries,
            self._learning_rates
        )

    def get_logits(self, inputs):

        ops = dict(
            bottleneck = self._bottleneck,
            compression=self._compression,
            activation=self._activation,
            padding="same",
            dropout = dict(rate = self._dropout_rate),
            batch_norm = dict(training = inputs.training, momentum = 0.9)
        )

        print("###############################")
        print("# Model")
        print("###############################")
        # cast
        net = tf.cast(self.inputs.features, tf.float32, "cast"); print("Input: {}".format(net))
        # net = tf.layers.batch_normalization(net, training=inputs.training); print("Batch Norm: {}".format(net))

        # big kernel
        net = tf.layers.conv2d(net, 2 * self._growth_rate, [3, 3], padding='same')
        print("Batch Norm Layer 24, 7x7: {}".format(net))

        # dense 1
        # self._n_layers = 6
        net = ti.layers.conv2d_dense_block(net, self._growth_rate, self._n_layers, **ops); print("DenseBlock(growth_rate={}, layers={}, bottleneck={}, compression={}): {}".format(self._growth_rate, self._n_layers, self._bottleneck, self._compression, net))
        net = tf.layers.average_pooling2d(net, [2, 2], strides=2); print("Average Pooling 2x2".format(net))


        # dense 2
        # self._n_layers = 12
        net = ti.layers.conv2d_dense_block(net, self._growth_rate, self._n_layers, **ops); print("DenseBlock(growth_rate={}, layers={}, bottleneck={}, compression={}): {}".format(self._growth_rate, self._n_layers, self._bottleneck, self._compression, net))
        net = tf.layers.average_pooling2d(net, [2, 2], strides=2); print("Average Pooling 2x2".format(net))

        # dense 3
        # self._n_layers = 24
        net = ti.layers.conv2d_dense_block(net, self._growth_rate, self._n_layers, **ops); print("DenseBlock(growth_rate={}, layers={}, bottleneck={}, compression={}): {}".format(self._growth_rate, self._n_layers, self._bottleneck, self._compression, net))

        # global average pooling
        shape = net.get_shape()[1]
        net = tf.layers.average_pooling2d(net, [shape, shape], strides=1); print("Global Average Pooling: {}".format(net))
        net = tf.contrib.layers.flatten(net); print("Flatten: {}".format(net))

        # dense
        net = tf.layers.dense(net, n_classes); print("Dense Layer({}): {}".format(n_classes, net))

        print("###############################\n")

        return net

    def get_summaries(self, inputs):
        return [
            tf.summary.scalar("learning_rate", self.learning_rate)
        ]

    def random_rotate_images(self, net):
        return tf.where(
            self.inputs.training,
            tf.contrib.image.rotate(
                net,
                tf.random_uniform(tf.shape(net)[:1], minval = -self._rotation_angle, maxval = self._rotation_angle)
            ),
            net
        )

    def get_update(self, *args, **kwargs):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return super(Model, self).get_update(*args, **kwargs)
