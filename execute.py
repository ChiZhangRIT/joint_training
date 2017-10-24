from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pdb

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import pickle as pkl

import data_utils
import seq2seq_model
from ops import image_embedding
from ops import load_jpeg_with_tensorflow

# import seq2seq_model

try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.

gConfig = {}

def get_config(config_file='joint_training.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    # _conf_booleans = [ (key, bool(value)) for key,value in parser.items('booleans') ]
    _conf_booleans = [ (name, parser.getboolean('booleans', name))
                        for name in parser.options('booleans') ]
    return dict(_conf_ints + _conf_floats + _conf_strings + _conf_booleans)


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(10, 15), (10, 20), (15, 30), (15, 50)]


def read_data(session, im_filenames, source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
    im_filenames: text file containing all image paths
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

    Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (im, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    im_placeholder = tf.placeholder(shape=[1,299,299,3], dtype=tf.float32)
    inception_output_tensor = image_embedding.inception_v3(
                      im_placeholder,
                      trainable=False,
                      is_training=image_embedding.is_training(gConfig['mode']))

    with tf.gfile.GFile(im_filenames, mode="r") as im_file:
        with tf.gfile.GFile(source_path, mode="r") as source_file:
            with tf.gfile.GFile(target_path, mode="r") as target_file:
              im, source, target = im_file.readline(), source_file.readline(), target_file.readline()
              counter = 0
              image_tensor = load_jpeg_with_tensorflow.get_image_tensor(im.rstrip())
              tf.global_variables_initializer().run()
              coordinator = tf.train.Coordinator()
              threads = tf.train.start_queue_runners(coord=coordinator)

              while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                  print("  reading data line %d" % counter)
                  sys.stdout.flush()

                # extract image vector
                image = session.run(image_tensor)
                inception_output  = session.run(inception_output_tensor,
                                    feed_dict={im_placeholder:image})
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                  if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([inception_output, target_ids])
                    break
                im, source, target = im_file.readline(), source_file.readline(), target_file.readline()
    coordinator.request_stop()
    coordinator.join(threads)
    return data_set


def read_vector(session, im_filenames, source_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]
    with open('data/caffenet_4k/mscoco_train_fc6.pkl','r') as f:
        vectors = pkl.load(f)

    with tf.gfile.GFile(im_filenames, mode="r") as im_file:
        with tf.gfile.GFile(source_path, mode="r") as source_file:
            with tf.gfile.GFile(target_path, mode="r") as target_file:

                im, source, target = im_file.readline(), source_file.readline(), target_file.readline()
                counter = 0
                im_key = im.rstrip().split('/')[-1]  # 'COCO_train2014_000000390534.jpg'

                while source and target and (not max_size or counter < max_size):
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()

                    # extract image vector
                    vector = vectors[im_key]

                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    target_ids.append(data_utils.EOS_ID)
                    for bucket_id, (source_size, target_size) in enumerate(_buckets):
                        if len(source_ids) < source_size and len(target_ids) < target_size:
                            data_set[bucket_id].append([vector, target_ids])
                            break
                    im, source, target = im_file.readline(), source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only):

    """Create model and initialize or load parameters"""
    model = seq2seq_model.Seq2SeqModel( gConfig['enc_vocab_size'],
                                      gConfig['dec_vocab_size'], _buckets,
                                      gConfig['layer_size'],
                                      gConfig['embed_size'],
                                      gConfig['num_layers'],
                                      gConfig['max_gradient_norm'],
                                      gConfig['batch_size'],
                                      gConfig['learning_rate'],
                                      gConfig['learning_rate_decay_factor'],
                                      forward_only=forward_only,
                                      use_pretrained_embedding=gConfig['pretrained_embedding'],
                                      pretrained_embedding_path=gConfig['pretrained_embedding_path'])

    if 'pretrained_model' in gConfig:
        model.saver.restore(session,gConfig['pretrained_model'])
        return model

    ckpt = tf.train.get_checkpoint_state(gConfig['model_dir'])
    if ckpt and ckpt.model_checkpoint_path:
        session.run(tf.global_variables_initializer())
        # print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        # model.saver.restore(session, ckpt.model_checkpoint_path)
        print("Reading parameters from previous model ...")
        model.saver.restore(session, "model/joint_caption_1024units_noatt_50kvocab/seq2seq.ckpt-550000")

    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    # prepare dataset
    if not (tf.gfile.Exists(gConfig['working_dir'] + 'im_filename.txt') and tf.gfile.Exists(gConfig['working_dir'] + 'caption_a.txt') and tf.gfile.Exists(gConfig['working_dir'] + 'caption_b.txt')):
        print("Creating dataset...")
        # generate im_filename, caption_a, caption_b and save them to working_dir
        data_utils.parse_MSCOCO(gConfig['mscoco_path'],gConfig['working_dir'],permute=True)

    # prepare encoder/decoder inputs
    print("Preparing data in %s" % gConfig['working_dir'])
    enc_train, dec_train, _, _ = data_utils.prepare_custom_data(gConfig['working_dir'],gConfig['working_dir'] + 'caption_a.txt', gConfig['working_dir'] + 'caption_b.txt', gConfig['enc_vocab_size'],gConfig['dec_vocab_size'])

    # setup config to use BFC allocator
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    config.gpu_options.allow_growth=True  # "chunks"

    with tf.Session(config=config) as sess:
        # create model
        print("Creating %d layers of %d units." % (gConfig['num_layers'], gConfig['layer_size']))
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        model = create_model(sess, False)




        # Read data into buckets and compute their sizes.
        print ("Reading training data (limit: %d)."
               % gConfig['max_train_data_size'])
        train_set = read_vector(sess, gConfig['working_dir'] + 'im_filename.txt',
                              enc_train,
                              dec_train,
                              gConfig['max_train_data_size'])
        # train_set = read_vector(sess, gConfig['working_dir'] + 'toy_im_filename.txt',
        #                       gConfig['working_dir'] + 'toy_caption_a.txt.ids50000',
        #                       gConfig['working_dir'] + 'toy_caption_b.txt.ids50000',
        #                       gConfig['max_train_data_size'])
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(gConfig['log_dir'], graph=sess.graph)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        # pdb.set_trace()
        while model.global_step.eval() <= gConfig['max_num_steps']:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            step_loss_summary = tf.Summary()
            learning_rate_summary = tf.Summary()

            # Get a batch and make a step.
            start_time = time.time()
            decoder_hiddens, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, decoder_hiddens, decoder_inputs,
                                         target_weights, bucket_id, False)

            step_loss_value = step_loss_summary.value.add()
            step_loss_value.tag = "step loss"
            step_loss_value.simple_value = step_loss.astype(float)
            learning_rate_value = learning_rate_summary.value.add()
            learning_rate_value.tag = "learning rate"
            learning_rate_value.simple_value = model.learning_rate.eval().astype(float)
            # Write logs at every iteration
            summary_writer.add_summary(step_loss_summary, model.global_step.eval())
            summary_writer.add_summary(learning_rate_summary, model.global_step.eval())

            step_time += (time.time() - start_time) / gConfig['steps_per_checkpoint']
            loss += step_loss / gConfig['steps_per_checkpoint']
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % gConfig['steps_per_checkpoint'] == 0 or current_step == 1:
                # Print statistics for the previous epoch.
                if current_step == 1:
                    perplexity = math.exp(step_loss) if step_loss < 300 else float('inf')
                    print ("global step %d learning rate %.4f loss %.4f perplexity %.2f"
                            % (model.global_step.eval(), model.learning_rate.eval(),
                            step_loss, perplexity))
                else:
                    perplexity = math.exp(loss) if loss < 300 else float('inf')
                    print ("global step %d learning rate %.4f step-time %.2f loss %.4f perplexity "
                            "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                            step_time, loss, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(gConfig['model_dir'], "seq2seq.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                sys.stdout.flush()

            # pdb.set_trace()


if __name__ == '__main__':
    if len(sys.argv) - 1:
        gConfig = get_config(sys.argv[1])
    else:
        # get configuration from seq2seq.ini
        gConfig = get_config()

    if not tf.gfile.Exists(gConfig['model_dir']):
        tf.gfile.MakeDirs(gConfig['model_dir'])
    if not tf.gfile.Exists(gConfig['log_dir']):
        tf.gfile.MakeDirs(gConfig['log_dir'])
    if not tf.gfile.Exists(gConfig['result_dir']):
        tf.gfile.MakeDirs(gConfig['result_dir'])
    if not tf.gfile.Exists(gConfig['working_dir']):
        tf.gfile.MakeDirs(gConfig['working_dir'])

    print('\n>> Mode : %s\n' %(gConfig['mode']))

    if gConfig['mode'] == 'train':
        # start training
        train()
    else:
        print('mode cannot be recognized.')
