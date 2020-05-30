from API.sentence_wer import get_sentence_wer
from blist import sorteddict

def get_rescore_kaldi_n_best(kaldi_n_best_sentence, kaldi_n_best_acoustics, session, m, fasttext_model, max_word_length, char_vocab, rnn_state):
    """

    :param kaldi_n_best: list of sentences
    :param session:
    :param m:
    :param fasttext_model:
    :param max_word_length:
    :param char_vocab:
    :param rnn_state:
    :return: sorted dict key: sent value: wer. the keys order are in rescore order.
    """
    rescore_kaldi_n_best = {}

    for sentence, acoustics in zip(kaldi_n_best_sentence, kaldi_n_best_acoustics):
        sentence_wer = get_sentence_wer(sentence, acoustics, session, m, fasttext_model, max_word_length, char_vocab, rnn_state)
        rescore_kaldi_n_best[sentence] = sentence_wer

    return sorteddict(rescore_kaldi_n_best).keys()


