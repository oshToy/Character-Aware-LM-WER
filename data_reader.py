from __future__ import print_function
from __future__ import division

from sklearn.preprocessing import MinMaxScaler
import ast
import os
import codecs
import collections
import numpy as np
import pandas as pd
from gensim.models import FastText



class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data(data_dir, max_word_length, num_unroll_steps, eos='+', datas=['train', 'valid'], batch_size=25):
    char_vocab = Vocab()
    char_vocab.feed(' ')  # blank is at index 0 in char vocab
    char_vocab.feed('{')  # start is at index 1 in char vocab
    char_vocab.feed('}')  # end   is at index 2 in char vocab

    word_vocab = Vocab()
    word_vocab.feed(' ')  # empty word for padding  at index 0 in word vocab
    word_vocab.feed('|')  # <unk> is at index 1 in word vocab

    actual_max_word_length = 0
    word_tokens = collections.defaultdict(list)
    char_tokens = collections.defaultdict(list)
    wers = {}
    words = {}
    acoustics = {}

    for fname in datas:
        wers[fname] = pd.Series(name='wer')
        words[fname] = list()
        acoustics[fname] = list()

        print('reading', fname)
        # with codecs.open(os.path.join(data_dir, fname + '.txt'), 'r', 'utf-8') as f:
        print(data_dir)
        for file in os.listdir(os.path.join(data_dir, fname)):
            df = pd.read_pickle(os.path.join(data_dir, fname, file))
            print(str(df.shape))
            df = df.dropna()
            # shuffle data in train
            if fname == 'train':
                df = df.sample(frac=1)
            if fname == 'test':
                df = df[:((len(df) // batch_size) * batch_size)]
            print(str(df.shape))
            wers[fname] = wers[fname].append(df['wer'])
            for line in df.iterrows():
                sent_acoustics = line[1]['acustic_tuple']
                sent = line[1]['sent']
                word_count_last_sent = 0
                sent = sent.strip()
                sent = sent.replace('}', '').replace('{', '').replace('|', '')
                sent = sent.replace('<unk>', ' | ')
                if eos:
                    sent = sent.replace(eos, '')
                sent_words = sent.split()
                for word_index in range(num_unroll_steps):
                    if word_index >= len(sent_words):
                        # Padding Zero UpTo max_sent_size
                        word = ' '
                        acoustics[fname].append([0, 0, 0, 0])
                    else:
                        word = sent_words[word_index]
                        acoustic_tuple = sent_acoustics[word_index]
                        feature_1, feature_2 = acoustic_tuple
                        feature_3 = feature_1 + feature_2
                        feature_4 = feature_1 - feature_2
                        acoustics[fname].append([feature_1, feature_2, feature_3, feature_4])

                    words[fname].append(word)

                    if len(word) > max_word_length - 2:  # space for 'start' and 'end' chars
                        word = word[:max_word_length - 2]

                    word_tokens[fname].append(word_vocab.feed(word))

                    char_array = [char_vocab.feed(c) for c in '{' + word + '}']
                    char_tokens[fname].append(char_array)

                    actual_max_word_length = max(actual_max_word_length, len(char_array))
                    word_count_last_sent += 1
                    if eos:
                        word_tokens[fname].append(word_vocab.feed(eos))

                        char_array = [char_vocab.feed(c) for c in '{' + eos + '}']
                        char_tokens[fname].append(char_array)
        wers[fname] = np.array(wers[fname])
    assert actual_max_word_length <= max_word_length
    assert len(words) == len(acoustics)
    print()
    print('actual longest token length is:', actual_max_word_length)
    print('size of word vocabulary:', word_vocab.size)
    print('size of char vocabulary:', char_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    char_tensors = {}
    for fname in datas:
        assert len(char_tokens[fname]) == len(word_tokens[fname])

        word_tensors[fname] = np.array(word_tokens[fname], dtype=np.int32)
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), actual_max_word_length], dtype=np.int32)

        for i, char_array in enumerate(char_tokens[fname]):
            char_tensors[fname][i, :len(char_array)] = char_array

    return word_vocab, char_vocab, word_tensors, char_tensors, actual_max_word_length, words, \
           wers, acoustics


def load_test_data(data_dir, max_word_length, num_unroll_steps, eos='+', datas=['test'], batch_size=25):
    char_vocab = Vocab()
    char_vocab.feed(' ')  # blank is at index 0 in char vocab
    char_vocab.feed('{')  # start is at index 1 in char vocab
    char_vocab.feed('}')  # end   is at index 2 in char vocab

    word_vocab = Vocab()
    word_vocab.feed(' ')   # empty word for padding  at index 0 in word vocab
    word_vocab.feed('|')  # <unk> is at index 1 in word vocab

    actual_max_word_length = 0
    word_tokens = collections.defaultdict(list)
    char_tokens = collections.defaultdict(list)
    wers = {}
    words = {}
    acoustics = {}
    files_name = {}
    kaldi_sents_index = {}
    for fname in datas:
        wers[fname] = pd.Series(name='wer')
        files_name[fname] = pd.Series(name='file_name')
        kaldi_sents_index[fname] = pd.Series(name='kaldi_sents_index')
        words[fname] = list()
        acoustics[fname] = list()

        print('reading', fname)
        # with codecs.open(os.path.join(data_dir, fname + '.txt'), 'r', 'utf-8') as f:
        print(data_dir)
        for file in os.listdir(os.path.join(data_dir, fname)):
            df = pd.read_pickle(os.path.join(data_dir, fname, file))
            print(str(df.shape))
            df = df.dropna()
            # shuffle data in train
            if fname == 'train':
                df = df.sample(frac=1)
            if fname == 'test':
                df = df[:((len(df) // batch_size) * batch_size)]
            print(str(df.shape))
            wers[fname] = wers[fname].append(df['wer'])
            files_name[fname] = files_name[fname].append(df['file_name'])
            kaldi_sents_index[fname] = kaldi_sents_index[fname].append(df['predict_sent_index'])
            for line in df.iterrows():
                sent_acoustics = line[1]['acustic_tuple']
                sent = line[1]['sent']
                word_count_last_sent = 0
                sent = sent.strip()
                sent = sent.replace('}', '').replace('{', '').replace('|', '')
                sent = sent.replace('<unk>', ' | ')
                if eos:
                    sent = sent.replace(eos, '')
                sent_words = sent.split()
                for word_index in range(num_unroll_steps):
                    if word_index >= len(sent_words):
                        # Padding Zero UpTo max_sent_size
                        word = ' '
                        acoustics[fname].append([0, 0, 0, 0])
                    else:
                        word = sent_words[word_index]
                        acoustic_tuple = sent_acoustics[word_index]
                        feature_1, feature_2 = acoustic_tuple
                        feature_3 = feature_1 + feature_2
                        feature_4 = feature_1 - feature_2
                        acoustics[fname].append([feature_1, feature_2, feature_3, feature_4])

                    words[fname].append(word)

                    if len(word) > max_word_length - 2:  # space for 'start' and 'end' chars
                        word = word[:max_word_length - 2]

                    word_tokens[fname].append(word_vocab.feed(word))

                    char_array = [char_vocab.feed(c) for c in '{' + word + '}']
                    char_tokens[fname].append(char_array)

                    actual_max_word_length = max(actual_max_word_length, len(char_array))
                    word_count_last_sent += 1
                    if eos:
                        word_tokens[fname].append(word_vocab.feed(eos))

                        char_array = [char_vocab.feed(c) for c in '{' + eos + '}']
                        char_tokens[fname].append(char_array)
        wers[fname] = np.array(wers[fname])
        files_name[fname] = np.array(files_name[fname])
        kaldi_sents_index[fname] = np.array(kaldi_sents_index[fname])
    assert actual_max_word_length <= max_word_length
    assert len(words) == len(acoustics)
    print()
    print('actual longest token length is:', actual_max_word_length)
    print('size of word vocabulary:', word_vocab.size)
    print('size of char vocabulary:', char_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    char_tensors = {}
    for fname in datas:
        assert len(char_tokens[fname]) == len(word_tokens[fname])

        word_tensors[fname] = np.array(word_tokens[fname] , dtype=np.int32)
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), actual_max_word_length], dtype=np.int32)

        for i, char_array in enumerate(char_tokens[fname]):
            char_tensors[fname][i, :len(char_array)] = char_array

    return word_vocab, char_vocab, word_tensors, char_tensors, actual_max_word_length, words,\
           wers, acoustics, files_name, kaldi_sents_index


class FasttextModel:
    def __init__(self,fasttext_path=None):
        self.fasttext_model = FastText.load(fasttext_path)

    def get_fasttext_model(self):
        return self.fasttext_model


class DataReaderFastText:
    FEATURES_PER_WORD = 4
    def __init__(self, words_list, batch_size, num_unroll_steps, model, data, acoustics):

        length = len(words_list[data])
        word_vector_size = model.vector_size

        # round down length to whole number of slices
        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        words_list[data] = words_list[data][:reduced_length]
        acoustics[data] = acoustics[data][:reduced_length]

        words_vectors_tensor = model.wv[words_list[data]]
        acoustics_tensor = np.array([np.array(xi) for xi in acoustics[data]])

        x_ft_batches = words_vectors_tensor.reshape([batch_size, -1, num_unroll_steps, word_vector_size])
        x_ft_batches = np.transpose(x_ft_batches, axes=(1, 0, 2, 3))

        x_acoustics_batches = acoustics_tensor.reshape([batch_size, -1, num_unroll_steps, self.FEATURES_PER_WORD])
        x_acoustics_batches = np.transpose(x_acoustics_batches, axes=(1, 0, 2, 3))
        original_x_acoustics_batches_shape = x_acoustics_batches.shape

        def min_max_scale_batch(batch):
            scaler = MinMaxScaler()
            batch = batch.reshape(-1, self.FEATURES_PER_WORD)
            return scaler.fit_transform(batch)

        x_acoustics_batches = x_acoustics_batches.reshape(x_acoustics_batches.shape[0], -1)
        x_acoustics_batches = np.apply_along_axis(func1d=min_max_scale_batch, axis=1, arr=x_acoustics_batches)
        x_acoustics_batches = x_acoustics_batches.reshape(original_x_acoustics_batches_shape)

        assert x_acoustics_batches.shape[0] == x_ft_batches.shape[0]
        self.x_acoustics_batches = x_acoustics_batches
        self.x_ft_batches = list(x_ft_batches)
        self.length = len(self.x_ft_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.word_vector_size = word_vector_size

    def iter(self):
        for x_ft, x_acoustics in zip(self.x_ft_batches, self.x_acoustics_batches):
            yield np.concatenate((x_ft.reshape(-1, self.word_vector_size), x_acoustics.reshape(-1, self.FEATURES_PER_WORD)), axis=1).T
            #TODO min max acoustic


class DataReader:

    def __init__(self, word_tensor, char_tensor, batch_size, num_unroll_steps, wers_ndarray):

        length = word_tensor.shape[0]  # max_words_in_sent(20) * wers_ndarray.shape[0]
        assert char_tensor.shape[0] == length

        max_word_length = char_tensor.shape[1]

        # round down length to whole number of slices
        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        char_tensor = char_tensor[:reduced_length, :]

        # Padding zeroes to wers
        for _ in range(batch_size - (len(wers_ndarray) % batch_size)):
            wers_ndarray = np.append(wers_ndarray, 0)

        print('wers shape ' + str(wers_ndarray.shape))

        x_batches = char_tensor.reshape([batch_size, -1, num_unroll_steps, max_word_length])
        y_batches = wers_ndarray.reshape([batch_size, -1])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))  # num of batches*sent on batch*words in sent*char_length
        y_batches = np.transpose(y_batches, axes=(1, 0))

        if x_batches.shape[0] != y_batches.shape[0]:
            y_batches = y_batches[:x_batches.shape[0], :]

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)

        assert len(self._x_batches) == len(self._y_batches)
        assert x_batches.shape[1] == y_batches.shape[1]
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps

    def iter(self):
        for x, y in zip(self._x_batches, self._y_batches):
            yield x, np.array(y).reshape(y.shape[0], 1)

class TestDataReader:

    def __init__(self, word_tensor, char_tensor, batch_size, num_unroll_steps, wers_ndarray, files_name_ndarray, kaldi_sents_index_ndarray):

        length = word_tensor.shape[0]  # max_words_in_sent(20) * wers_ndarray.shape[0]
        assert char_tensor.shape[0] == length

        max_word_length = char_tensor.shape[1]

        # round down length to whole number of slices
        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        char_tensor = char_tensor[:reduced_length, :]

        # Padding zeroes to wers
        for _ in range(batch_size - (len(wers_ndarray) % batch_size)):
            wers_ndarray = np.append(wers_ndarray, 0)
            files_name_ndarray = np.append(files_name_ndarray, 0)
            kaldi_sents_index_ndarray = np.append(kaldi_sents_index_ndarray, 0)

        print('wers shape ' + str(wers_ndarray.shape))
        print('files_name_ndarray shape ' + str(files_name_ndarray.shape))
        print('kaldi_sents_index_ndarray shape ' + str(kaldi_sents_index_ndarray.shape))

        x_batches = char_tensor.reshape([batch_size, -1, num_unroll_steps, max_word_length])
        y_batches = wers_ndarray.reshape([batch_size, -1])
        files_name_batches = files_name_ndarray.reshape([batch_size, -1])
        kaldi_sents_index_batches = kaldi_sents_index_ndarray.reshape([batch_size, -1])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))  # num of batches*sent on batch*words in sent*char_length
        y_batches = np.transpose(y_batches, axes=(1, 0))
        files_name_batches = np.transpose(files_name_batches, axes=(1, 0))
        kaldi_sents_index_batches = np.transpose(kaldi_sents_index_batches, axes=(1, 0))


        if x_batches.shape[0] != y_batches.shape[0]:
            y_batches = y_batches[:x_batches.shape[0], :]
            files_name_batches = files_name_batches[:x_batches.shape[0], :]
            kaldi_sents_index_batches = kaldi_sents_index_batches[:x_batches.shape[0], :]

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        self._files_name_batches = list(files_name_batches)
        self._kaldi_sents_index_batches = list(kaldi_sents_index_batches)

        assert len(self._x_batches) == len(self._y_batches) == len(self._files_name_batches) == len(self._kaldi_sents_index_batches)
        assert x_batches.shape[1] == y_batches.shape[1] == files_name_batches.shape[1] == kaldi_sents_index_batches.shape[1]
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps

    def iter(self):
        for x, y, files_name, kaldi_sents_index in zip(self._x_batches, self._y_batches, self._files_name_batches, self._kaldi_sents_index_batches):
            yield x, np.array(y).reshape(y.shape[0], 1), np.array(files_name).reshape(files_name.shape[0], 1),\
                  np.array(kaldi_sents_index).reshape(kaldi_sents_index.shape[0], 1)



if __name__ == '__main__':

    _, _, wt, ct, _, _ = load_data('data', 65, 25)
    print(wt.keys())

    count = 0
    for x, y in DataReader(wt['valid'], ct['valid'], 20, 35).iter():
        count += 1
        print(x, y)
        if count > 0:
            break
