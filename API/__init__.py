import tensorflow as tf
from API.init_model import run as init_model
from API.sentence_wer import get_sentence_wer
from API.rescore_kaldi_n_best import get_rescore_kaldi_n_best
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec


class EpochSaver(CallbackAny2Vec):

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        if self.epoch in [1, 5, 10, 15, 20]:
            output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
            model.save(output_path)
            print('model save')
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


class CharacterAwareLmWer:

    def __init__(self):
        self.session = None
        self.m = None
        self.fasttext_model = None
        self.max_word_length = None
        self.rnn_state = None
        self.char_vocab = None
        self.word_vocab = None
        self.load_model()

    def load_model(self):
        self.session, self.m, self.fasttext_model, self.max_word_length, self.char_vocab, self.word_vocab = init_model()
        self.rnn_state = self.session.run(self.m.initial_rnn_state)

    def sentence_wer(self, sentence, acoustics):
        # rnn outputs
        return get_sentence_wer(sentence, acoustics, self.session, self.m, self.fasttext_model, self.max_word_length,
                                self.char_vocab,
                                self.rnn_state)

    def rescore_kaldi_n_best(self, n_best_sentence, n_best_acoustics):
        # rnn outputs
        return get_rescore_kaldi_n_best(n_best_sentence, n_best_acoustics, self.session, self.m, self.fasttext_model, self.max_word_length,
                                        self.char_vocab,
                                        self.rnn_state)


if __name__ == '__main__':
    model = CharacterAwareLmWer()
    print(model.sentence_wer(sentence='hello world', acoustics=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]))
    print(model.rescore_kaldi_n_best(n_best_sentence=['hello world', 'hello', 'wow hiz'],
                                     n_best_acoustics=[
                                         [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                                         [[0.1, 0.2, 0.3, 0.5]],
                                         [[0.1, 0.2, 0.3, 0.6], [0.1, 0.6, 0.7, 0.8]]
                                     ]))
