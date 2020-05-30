import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

def sentence_pre_process(sentence):
    return str(sentence).lower()


def sentence_embedding(fasttext_model, sentence, acoustics, max_word_length, char_vocab):
    input_2 = []
    char_input = []
    words = sentence.split(' ')
    for i in range(FLAGS.num_unroll_steps):
        word = ''
        if i < len(words):
            word = words[i]
            words_tf = fasttext_model.wv[word]
            input_2.append(np.reshape((np.concatenate((words_tf, acoustics[i])).T), (-1, 1)))
        else:
            words_tf = fasttext_model.wv[word]
            input_2.append(np.reshape((np.concatenate((words_tf, np.zeros(4))).T), (-1, 1)))

        word_char_input = np.zeros((1, 1, max_word_length))
        for i, c in enumerate('{' + word + '}'):
            word_char_input[0, 0, i] = char_vocab[c]

        char_input.append(word_char_input)
    return np.reshape(input_2, (FLAGS.num_unroll_steps, -1)).T, np.reshape(char_input, (1, FLAGS.num_unroll_steps, -1))


def get_sentence_wer(sentence, acoustics,  session, m, fasttext_model, max_word_length, char_vocab, rnn_state):
    sentence = sentence_pre_process(sentence)
    input_2, char_input = sentence_embedding(fasttext_model, sentence, acoustics, max_word_length, char_vocab)

    logits = session.run([
        m.logits
    ], {
        m.input2: input_2,
        m.input: char_input,
        m.initial_rnn_state: rnn_state
    })

    return logits

