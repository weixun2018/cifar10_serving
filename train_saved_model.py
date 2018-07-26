# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train_new.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'saved_model/cifar10/1',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def preprocess_image(image_buffer):
    img_decoded = tf.image.decode_jpeg(image_buffer, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [32, 32])
    reshape_img = tf.cast(img_resized, tf.float32)
    distorted_image = tf.random_crop(reshape_img, [24, 24, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([24, 24, 3])

    return float_image


def get_labels_list():
    with open('data/cifar10_data/cifar-10-batches-bin/batches.meta.txt') as f:
        lines = f.readlines()
        labels_list = [i[:-1] for i in lines if i != '\n']

        print('-----> labels number:', len(labels_list))

        return labels_list


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.

        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        # Input transformation.
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            'image/encoded': tf.FixedLenFeature(
                shape=[], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        jpegs = tf_example['image/encoded']
        imgs = tf.map_fn(preprocess_image, jpegs, tf.float32)
        y_ = tf.placeholder(tf.int32, [None])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(imgs)

        # Calculate loss.
        loss = cifar10.loss(logits, y_)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar10.train(loss, global_step)

        y = tf.nn.softmax(logits, name='y')
        values, indices = tf.nn.top_k(y, 1)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(
            tf.constant(get_labels_list()))
        prediction_class = table.lookup(tf.to_int64(indices))

        sess = tf.InteractiveSession()

        g = tf.get_default_graph()
        for ops in g.get_operations():
            if "DecodeJpeg" in ops.name:
                print(ops.name, ops.values())

        # Initialize an saver for store model checkpoints
        builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.train_dir)

        tf.global_variables_initializer().run()
        # start thread queue to speed up for data augmentation
        tf.train.start_queue_runners()

        start = time.time()
        for step in range(FLAGS.max_steps):
            images_train, labels_train = sess.run([images, labels])
            _loss, _ = sess.run([loss, train_op], feed_dict={imgs: images_train, y_: labels_train})
            if step % FLAGS.log_frequency == 0:
                duration = time.time() - start
                sec_per_batch = float(duration) / FLAGS.log_frequency
                print(
                    '{}: step:{:-6d}  loss:{:.2f}  {:.3f} sec/batch'.format(datetime.now(), step, _loss, sec_per_batch))
                start = time.time()

        print("Saving model...")
        # Build the signature_def_map.
        tensor_info_x = tf.saved_model.utils.build_tensor_info(jpegs)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(prediction_class)
        tensor_info_score = tf.saved_model.utils.build_tensor_info(values)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'class': tensor_info_y,
                         'score': tensor_info_score},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={
                                                 'predict_images':
                                                     prediction_signature
                                             },
                                             legacy_init_op=legacy_init_op)

    builder.save()


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    # tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
