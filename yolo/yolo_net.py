import tensorflow as tf
import numpy as np
from yolo.config import batch_size, vgg16_path


class YOLO(object):
    def __init__(self):
        self.input_x = tf.placeholder(tf.float32, [None, 448, 448, 3])
        self.input_y = tf.placeholder(tf.float32, [None, 7, 7, 2, 25])
        self.traing_bn = tf.placeholder(tf.bool)
        self.vgg_dict = np.load(vgg16_path, encoding='latin1').item()
        self.logits = self.net(self.input_x, self.traing_bn)
        self.loss = self.losses(self.logits, self.input_y)
        self.learning_rate = 1e-3
        self.opt = self.train(self.loss)

    def net(self, input, training):
        x = self.conv_layer(input, 'conv1_1')
        x = self.conv_layer(x, 'conv1_2')
        x = tf.layers.max_pooling2d(x, 2, 2, 'same', name='pool1')

        x = self.conv_layer(x, 'conv2_1')
        x = self.conv_layer(x, 'conv2_2')
        x = tf.layers.max_pooling2d(x, 2, 2, 'same', name='pool2')

        x = self.conv_layer(x, 'conv3_1')
        x = self.conv_layer(x, 'conv3_2')
        x = self.conv_layer(x, 'conv3_3')
        x = tf.layers.max_pooling2d(x, 2, 2, 'same', name='pool3')

        x = self.conv_layer(x, 'conv4_1')
        x = self.conv_layer(x, 'conv4_2')
        x = self.conv_layer(x, 'conv4_3')
        x = tf.layers.max_pooling2d(x, 2, 2, 'same', name='pool4')

        x = self.conv_layer(x, 'conv5_1')
        x = self.conv_layer(x, 'conv5_2')
        x = self.conv_layer(x, 'conv5_3')
        x = tf.layers.max_pooling2d(x, 2, 2, 'same', name='pool5')

        x = tf.layers.conv2d(x, 512, 3, 1, 'same', activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.layers.conv2d(x, 512, 3, 1, 'same', activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.layers.conv2d(x, 512, 3, 1, 'same', activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.layers.max_pooling2d(x, 2, 2, 'same', name='pool6')

        x = tf.layers.conv2d(x, 512, 3, 1, 'same', activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.layers.conv2d(x, 512, 3, 1, 'same', activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.layers.conv2d(x, 512, 3, 1, 'same', activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.layers.max_pooling2d(x, 2, 2, 'same', name='pool7')

        x = tf.layers.dense(x, 1024)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, 1024)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, 512)
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)

        x = tf.layers.flatten(x)
        logits = tf.layers.dense(x, 7 * 7 * 2 * 25)

        return logits

    def conv_layer(self, input, layer_name):
        with tf.variable_scope(layer_name):
            weight_pretrain = self.vgg_dict[layer_name][0]
            bias_pretrain = self.vgg_dict[layer_name][1]
            conv = tf.nn.conv2d(input, weight_pretrain, [1, 1, 1, 1], 'SAME')
            output = tf.nn.relu(conv + bias_pretrain)
            return output

    def losses(self, logits, label):
        prediction = tf.reshape(logits, [-1, 7, 7, 2, 25])

        pred_class = prediction[..., :20]  # [-1,7,7,2,20]
        pred_confidence = prediction[..., 20]  # [-1,7,7,2]
        pred_loc = prediction[..., -4:]  # [-1,7,7,2,4]

        label_class = label[..., :20]
        label_confidence = label[..., 20]
        label_loc = label[..., -4:]  # [-1,7,7,2,4]

        # iou
        IOU = self.iou(pred_loc, label_loc)  # [-1,7,7,2]
        IOU = tf.reduce_max(IOU, axis=-1, keep_dims=True)  # [-1,7,7,1]
        IOU_label = tf.cast(IOU >= 0.5, tf.float32) * label_confidence  # [-1,7,7,2]
        noobj_mask = tf.ones_like(IOU_label) - IOU_label

        # obj confidence loss
        obj_loss = tf.reduce_sum(tf.square(pred_confidence - IOU_label) * IOU_label) / batch_size

        # no obj confidecne loss
        no_obj_loss = tf.reduce_sum(tf.square(pred_confidence - IOU_label) * noobj_mask) / batch_size

        # loc loss:
        x_loss = tf.reduce_sum(tf.square(pred_loc[..., 0] - label_loc[..., 0]) * label_confidence) / batch_size
        y_loss = tf.reduce_sum(tf.square(pred_loc[..., 1] - label_loc[..., 1]) * label_confidence) / batch_size
        w_loss = tf.reduce_sum(
            tf.square(
                tf.sqrt(tf.abs(pred_loc[..., 2])) - tf.sqrt(tf.abs(label_loc[..., 2]))) * label_confidence) / batch_size
        h_loss = tf.reduce_sum(
            tf.square(
                tf.sqrt(tf.abs(pred_loc[..., 3])) - tf.sqrt(tf.abs(label_loc[..., 3]))) * label_confidence) / batch_size
        loc_loss = x_loss + y_loss + w_loss + h_loss

        # class loss:
        class_loss = tf.reduce_sum(
            tf.square(pred_class - label_class) * tf.expand_dims(label_confidence, axis=-1)) / batch_size

        total_loss = class_loss + 5 * loc_loss + 2 * obj_loss + 0.1 * no_obj_loss

        tf.summary.scalar('obj loss', obj_loss)
        tf.summary.scalar('no obj loss', no_obj_loss)
        tf.summary.scalar('loc loss', loc_loss)
        tf.summary.scalar('class loss', class_loss)
        tf.summary.scalar('total loss', total_loss)

        return total_loss

    def iou(self, box1, box2):
        box1_trans = tf.stack((box1[..., 0] - 0.5 * box1[..., 2],
                               box1[..., 1] - 0.5 * box1[..., 3],
                               box1[..., 0] + 0.5 * box1[..., 2],
                               box1[..., 1] + 0.5 * box1[..., 3]), axis=-1)
        box2_trans = tf.stack((box2[..., 0] - 0.5 * box2[..., 2],
                               box2[..., 1] - 0.5 * box2[..., 3],
                               box2[..., 0] + 0.5 * box2[..., 2],
                               box2[..., 1] + 0.5 * box2[..., 3]), axis=-1)
        inter_w = tf.maximum(0., tf.minimum(box1_trans[..., 2], box2_trans[..., 2]) - tf.maximum(box1_trans[..., 0],
                                                                                                 box2_trans[..., 0]))
        inter_h = tf.maximum(0., tf.minimum(box1_trans[..., 3], box2_trans[..., 3]) - tf.maximum(box1_trans[..., 1],
                                                                                                 box2_trans[..., 1]))
        intersection = inter_w * inter_h
        union = box1[..., 2] * box1[..., 3] + box2[..., 2] * box2[..., 3] - intersection
        return tf.clip_by_value(intersection / union, 0., 1.)

    def train(self, loss):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        return opt
