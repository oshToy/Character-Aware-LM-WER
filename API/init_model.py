from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from data_reader import load_data, DataReader, FasttextModel, DataReaderFastText

import model

flags = tf.flags

# data
flags.DEFINE_string('load_model',
                    r"E:\trained_models_wer\AMI_2020-04-12--23-10-25\epoch008_4813.5529.model",
                    'filename of the model to load')
# we need data only to compute vocabulary
flags.DEFINE_string('data_dir',
                    r"E://data_sets//AMI",
                    'data directory')
flags.DEFINE_string('fasttext_model_path',
                    "E:\\RNNLM Project\\Models\\Wikipedia\\Lower Case\\15 Epoch\\Wikipedia_epoch15.model",
                    'fasttext trained model path')
flags.DEFINE_integer('num_samples', 2000, 'how many words to generate')
flags.DEFINE_float('temperature', 0.2, 'sampling temperature')
flags.DEFINE_string('embedding', "kim fasttext", 'embedding method')
# model params
flags.DEFINE_integer('rnn_size', 650, 'size of LSTM internal state')
flags.DEFINE_integer('highway_layers', 2, 'number of highway layers')
flags.DEFINE_integer('char_embed_size', 15, 'dimensionality of character embeddings')
flags.DEFINE_string('kernels', '[1,2,3,4,5,6,7]', 'CNN kernel widths')
flags.DEFINE_string('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers', 2, 'number of layers in the LSTM')
flags.DEFINE_float('dropout', 0.5, 'dropout. 0 = no dropout')
flags.DEFINE_integer('batch_size', 1, 'number of sequences to train on in parallel')
flags.DEFINE_integer('num_unroll_steps', 35, 'number of timesteps to unroll for')

# optimization
flags.DEFINE_integer('max_word_length', 65, 'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed', 3435, 'random number generator seed')
flags.DEFINE_string('EOS', '+',
                    '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = flags.FLAGS



def run():
    ''' Loads trained model and evaluates it on test split '''

    if FLAGS.load_model is None:
        print('Please specify checkpoint file to load model from')
        return -1

    if not os.path.exists(FLAGS.load_model + '.meta'):
        print('Checkpoint file not found', FLAGS.load_model)
        return -1

    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length, words_list = \
        load_data(FLAGS.data_dir, FLAGS.max_word_length, FLAGS.num_unroll_steps, eos=FLAGS.EOS)

    fasttext_model = FasttextModel(fasttext_path=FLAGS.fasttext_model_path).get_fasttext_model()

    print('initialized test dataset reader')
    session = tf.Session()

    # tensorflow seed must be inside graph
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(seed=FLAGS.seed)

    ''' build inference graph '''
    with tf.variable_scope("Model"):
        m = model.inference_graph(
            char_vocab_size=char_vocab.size,
            word_vocab_size=word_vocab.size,
            char_embed_size=FLAGS.char_embed_size,
            batch_size=FLAGS.batch_size,
            num_highway_layers=FLAGS.highway_layers,
            num_rnn_layers=FLAGS.rnn_layers,
            rnn_size=FLAGS.rnn_size,
            max_word_length=max_word_length,
            kernels=eval(FLAGS.kernels),
            kernel_features=eval(FLAGS.kernel_features),
            num_unroll_steps=FLAGS.num_unroll_steps,
            dropout=0,
            embedding=FLAGS.embedding,
            fasttext_word_dim=300,
            acoustic_features_dim=4)
        # we need global step only because we want to read it from the model
        global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

    saver = tf.train.Saver()
    saver.restore(session, FLAGS.load_model)
    print('Loaded model from', FLAGS.load_model)

    ''' training starts here '''
    return session, m, fasttext_model, max_word_length, char_vocab, word_vocab


def run_t():
    return tf.app.run()


if __name__ == "__main__":
    tf.app.run()
